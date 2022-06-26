import os
import time
import parser_util
import imageio
import losses
import extract_mesh

from torch import optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from datasets.load_scannet import load_scannet_data
from nerf_helpers import *
from optimization import FeatureArray, DeformationField, PoseArray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                feature_array=None,
                pose_array=None,
                deformation_field=None,
                c2w_array=None,
                raw_noise_std=0.,
                mode='density',
                truncation=0.05,
                sc_factor=1.0,
                eval_mode=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      feature_array: FeatureArray. Module with per-frame learnable latent
        correction codes.
      pose_array: PoseArray. Module with per-frame extrinsic corrections.
      deformation_field: DeformationField. Module for a global image-plane
        ray correction.
      c2w_array: array of shape [N_frames, 4, 4]. Camera-to-world matrices
        for every frame.
      raw_noise_std: float. Noise to apply to raw density/sdf values.
      mode: str. Implicit scene representation ('density' or 'sdf').
      truncation: float. Truncation distance in meters.
      sc_factor: float. Scale factor by which the scene is downscaled from
        metric space to fit into a [-1, 1] cube.
      eval_mode: bool. Flag for training/eval modes.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      depth_map: [num_ray]. Depth map.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      depth0: See depth_map. Output for coarse model.
      z_vals: [num_rays, num_samples]. Depth of each sample on each ray.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.
        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """

        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        def sdf2weights(sdf):
            weights = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)

            signs = sdf[:, 1:] * sdf[:, :-1]
            mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
            inds = torch.argmax(mask, axis=1)
            inds = inds[..., None]
            z_min = torch.gather(z_vals, 1, inds) # The first surface
            mask = torch.where(z_vals < z_min + sc_factor * truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

            weights = weights * mask
            return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        # Extract RGB of each sample position along each ray.
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

        if mode == 'density':
            # Predict density of each sample along each ray. Higher values imply
            # higher likelihood of being absorbed at this point.
            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

            # Compute weight for RGB of each sample along each ray.  A cumprod() is
            # used to express the idea of the ray not having reflected up to this
            # sample yet.
            # [N_rays, N_samples]
            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        elif mode == 'sdf':
            weights = sdf2weights(raw[..., 3])
        else:
            raise Exception('Unknown color integration mode' + mode)

        # Computed weighted color of each sample along each ray.
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = torch.sum(weights, -1)

        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, -1)

        # Disparity map is inverse depth.
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract non-normalized viewing direction.
    viewdirs = ray_batch[:, 8:11]

    # Extract lower, upper bound for ray distance.
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Extract frame features
    frame_ids = ray_batch[:, 11]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0., 1., steps=N_samples)

    # Space integration times linearly between 'near' and 'far'. Same
    # integration points will be used for all rays.
    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
          z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    raw, translation = network_query_fn(pts, viewdirs, feature_array, pose_array, frame_ids,
                                        deformation_field, c2w_array, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        if network_fine is not None:
            # Obtain all points to evaluate color, density at.
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                  z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # Make predictions with network_fine.
            raw, translation_fine = network_query_fn(pts, viewdirs, feature_array, pose_array, frame_ids,
                                                          deformation_field, c2w_array, network_fine)

        else:
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                  z_samples[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # Make predictions with network_fn.
            raw_fine, translation_fine = network_query_fn(pts, viewdirs, feature_array, pose_array, frame_ids,
                                                          deformation_field, c2w_array, network_fn)

            z_vals = torch.cat([z_vals, z_samples], -1)
            indices = torch.argsort(z_vals, -1)
            z_vals = torch.gather(z_vals, -1, indices)
            raw = torch.gather(torch.cat([raw, raw_fine], -2), -2, indices)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

        if translation is not None:
            translation = 0.5 * (translation + translation_fine)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}

    if not eval_mode:
        ret = {**ret, 'z_vals': z_vals}

    if retraw:
        ret['raw'] = raw
        if translation is not None:
            ret['translation'] = translation
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret

    
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal,
           chunk=1024 * 32, rays=None, frame_ids=None, c2w=None, ndc=True,
           near=0., far=1.,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      frame_ids: array of shape [batch_size, 1]. Id of the frame the ray
        belongs to. Used to apply the correct corrective code and pose correction.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      depth_map: [batch_size]. Predicted depth values for rays.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # provide ray directions as input
    viewdirs = torch.reshape(rays_d, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # (ray origin, ray direction, min dist, max dist, viewing direction, frame_id)
    rays = torch.cat([rays, viewdirs, frame_ids], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = NeRF(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views + args.frame_features, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0 and not args.share_coarse_fine:
        model_fine = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())
        models['model_fine'] = model_fine
    
    # Create feature array
    feature_array = None
    if args.frame_features > 0:
        feature_array = FeatureArray(args.num_training_frames, args.frame_features)
        grad_vars += [feature_array.data]
        models['feature_array'] = feature_array

    # Create pose array
    pose_array = None
    if args.optimize_poses:
        pose_array = PoseArray(args.num_training_frames)
        grad_vars += [pose_array.data]
        models['pose_array'] = pose_array

    # Create deformation field
    deformation_field = None
    if args.use_deformation_field:
        deformation_field = DeformationField()
        grad_vars += list(deformation_field.parameters())
        models['deformation_field'] = deformation_field
    
    def network_query_fn(inputs, viewdirs, feature_array, pose_array, frame_ids, deformation_field, c2w_array, network_fn):
        return run_network(
            inputs, viewdirs, feature_array, pose_array, frame_ids, deformation_field, c2w_array, network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk)
    
    optimizer = optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print('Resetting step to', start)

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        if feature_array is not None:
            feature_array.load_state_dict(ckpt['feature_array_state_dict'])
            
        if pose_array is not None:
            pose_array.load_state_dict(ckpt['pose_array_state_dict'])

        if deformation_field is not None:   
            deformation_field.load_state_dict(ckpt['deformation_field_state_dict'])
    
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'feature_array': feature_array,
        'pose_array': pose_array,
        'N_samples': args.N_samples,
        'network_fn': model,
        'deformation_field': deformation_field,
        'mode': args.mode,
        'raw_noise_std': args.raw_noise_std,
        'truncation': args.trunc,
        'sc_factor': args.sc_factor,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['deformation_field'] = None
    render_kwargs_test['pose_array'] = None

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models

def train():
    parser = parser_util.get_parser()
    args = parser.parse_args()

    basedir = args.basedir
    expname = args.expname

    # Create log dir and copy the config file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    

    print('Load data')
    

    if args.dataset_type == "scannet":
        images, depth_images, poses, hwf, frame_indices = load_scannet_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)

    near = args.near
    far = args.far

    # Create nerf model
    args.num_training_frames = len(images)
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    global_step = start
    feature_array = None
    if 'feature_array' in models:
        feature_array = models['feature_array']

    bds_dict = {
        'near' : near,
        'far' : far,
        'c2w_array': torch.tensor(poses, dtype=torch.float32)
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if 'optimizer' in models:
        optimizer = models['optimizer']
    else:
        optimizer = optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    

    def get_rays_rgbd():
        print('get rays')
        # get_camera_rays_np() returns rays_direction=[H, W, 3]
        # for each pixel in the image. The origin is assumed to be (0, 0, 0).
        # This stack() adds a new dimension.
        # rays: [N, H, W, 3]
        rays = np.stack([get_camera_rays_np(H, W, focal) for _ in range(poses.shape[0])], 0)  # [N, H, W, 3]
        print('done, concats')

        # Concatenate color and depth
        # rays = view direction (3) + RGB (3) + Depth (1)
        rays = np.concatenate([rays, images], -1)  # [N, H, W, 6]
        rays = np.concatenate([rays, depth_images], -1)  # [N, H, W, 7]

        # Concatenate frame ids
        ids = np.arange(rays.shape[0], dtype=np.float32)
        ids = ids[:, np.newaxis, np.newaxis, np.newaxis]
        ids = np.tile(ids, [1, rays.shape[1], rays.shape[2], 1])

        # rays = view direction (3) + RGB (3) + Depth (1) + Frame id (1)
        rays = np.concatenate([rays, ids], -1)  # [N, H, W, 8]

        rays = rays.reshape([-1, rays.shape[-1]])  # [N_rays, 8]
        return rays
    
    '''Prepare ray data'''
    # rays_rgbd = view direction (3) + RGB (3) + Depth (1) + Frame id (1)
    # rays_rgbd: [N_rays, 8]
    rays_rgbd = get_rays_rgbd()
    print('shuffle rays')
    np.random.shuffle(rays_rgbd)
    print('done')
    i_batch = 0

    # Batch size
    N_rand = args.N_rand
    N_iters = args.N_iters
    print('Begin')
    print('TRAIN views are', frame_indices)

    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        # Random over all images
        batch = rays_rgbd[i_batch:i_batch+N_rand].to(device) # [B, 2+1, 3*?]
        
        batch_rays = torch.stack([torch.zeros_like(batch[:, :3]), batch[:, :3]], 0)
        target_s = batch[:, 3:6]
        target_d = batch[:, 6:7]
        frame_ids = batch[:, 7:8].to(torch.int64)

        i_batch += N_rand
        if i_batch >= rays_rgbd.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgbd.shape[0])
            rays_rgbd = rays_rgbd[rand_idx]
            i_batch = 0
    
        #####  Core optimization loop  #####
        optimizer.zero_grad()
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                frame_ids=frame_ids,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        img_loss = losses.compute_loss(rgb, target_s, args.rgb_loss_type)
        psnr = mse2psnr(img2mse(rgb, target_s))
        loss = args.rgb_weight * img_loss

        if 'rgb0' in extras:
            img_loss0 = losses.compute_loss(extras['rgb0'], target_s, args.rgb_loss_type)
            loss += args.rgb_weight * img_loss0
            psnr0 = mse2psnr(img_loss0)

        # Depth loss
        depth_loss = losses.get_depth_loss(depth, target_d)
        loss += args.depth_weight * depth_loss

        if 'depth0' in extras:
            depth_loss0 = losses.get_depth_loss(extras['depth0'], target_d)
            loss += args.depth_weight * depth_loss0

        # Loss for free space / truncation samples
        z_vals = extras['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = extras['raw'][..., -1]

        truncation = args.trunc * args.sc_factor
        fs_loss, sdf_loss = losses.get_sdf_loss(z_vals, target_d, sdf, truncation, args.sdf_loss_type)
        loss += args.fs_weight * fs_loss + args.trunc_weight * sdf_loss

        if feature_array is not None:
            reg_features = 0.1 * torch.mean(torch.square(feature_array.data))
            loss += reg_features

        if 'translation' in extras:
            translation = extras['translation']
            reg_translation = 0.01 * torch.mean(torch.square(translation))
            loss += reg_translation
        
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        
        t = time.time()-time0

        #####           end            #####

        # Rest is logging

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_model_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            if render_kwargs_train['feature_array'] is not None:
                save_model_dict['feature_array_state_dict'] = render_kwargs_train['feature_array'].state_dict()
            
            if render_kwargs_train['pose_array'] is not None:
                save_model_dict['pose_array_state_dict'] = render_kwargs_train['pose_array'].state_dict()

            if render_kwargs_train['deformation_field'] is not None:
                save_model_dict['deformation_field_state_dict'] = render_kwargs_train['deformation_field'].state_dict()

            torch.save(save_model_dict, path)
            print('Saved checkpoints at', path)
    

        if i % args.i_print == 0 or i < 10:

            print("{}: loss:{:.4e}, FS:{:.4e}, SDF:{:.4e}, PSNR:{:.4f}".format(
                i, loss.cpu().data.numpy().item(), 
                fs_loss.cpu().data.numpy().item(), 
                sdf_loss.cpu().data.numpy().item(), 
                psnr0.cpu().data.numpy().item(),
            ))

            # print('iter time {:.05f}'.format(dt))
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('img_loss', img_loss, global_step)
            writer.add_scalar('depth_loss', depth_loss, global_step)
            writer.add_scalar('free_space_loss', fs_loss, global_step)
            writer.add_scalar('sdf_loss', sdf_loss, global_step)
            writer.add_scalar('psnr', psnr, global_step)
            writer.add_histogram('sdf', sdf, global_step)
            if args.N_importance > 0:
                writer.add_scalar('psnr0', psnr0, global_step)
        
        if i % args.i_img == 0 and i > 0:

            def get_logging_images(img_i):
                pose = torch.eye(4, 4)

                render_height = H // args.render_factor
                render_width = W // args.render_factor
                render_focal = focal / args.render_factor

                ids = img_i * torch.ones((render_height * render_width, 1), dtype=torch.float32)
                rgb, disp, acc, depth, extras = render(render_height, render_width, render_focal, chunk=args.chunk,
                                                    frame_ids=ids,
                                                    c2w=pose, eval_mode=True, **render_kwargs_train)

                depth = depth[..., None].cpu().data.numpy()

                if 'depth0' in extras:
                    extras['depth0'] = extras['depth0'][..., None]

                rgb = rgb.cpu().data.numpy()
                acc = acc.cpu().data.numpy()
                disp = disp.cpu().data.numpy()
                for key in extras:
                    extras[key] = extras[key].cpu().data.numpy()

                return rgb, disp, acc, depth, extras

            # Save a rendered training view to disk
            img_i = np.random.choice(args.num_training_frames)
            rgb, disp, acc, depth, extras = get_logging_images(img_i)
            frame_idx = frame_indices[img_i]

            trainimgdir = os.path.join(basedir, expname, 'tboard_train_imgs')
            os.makedirs(trainimgdir, exist_ok=True)
            imageio.imwrite(os.path.join(trainimgdir, 'rgb_{:06d}_{:04d}.png'.format(i, frame_idx)), to8b(rgb))
            imageio.imwrite(os.path.join(trainimgdir, 'depth_{:06d}_{:04d}.png'.format(i, frame_idx)),
                            to8b(depth / np.max(depth)))

        if i % args.i_mesh == 0 and i > 0:
            network_fn = render_kwargs_test['network_fine'] if render_kwargs_test['network_fine'] is not None else \
                         render_kwargs_test['network_fn']
            isolevel = 0.0 if args.mode == 'sdf' else 20.0
            mesh_savepath = os.path.join(args.basedir, args.expname, f'mesh_{i:06}.ply')
            extract_mesh.extract_mesh(render_kwargs_test['network_query_fn'], feature_array, network_fn, args,
                                      isolevel=isolevel, mesh_savepath=mesh_savepath)
        
        global_step += 1
    



if __name__ == '__main__':
    train()

