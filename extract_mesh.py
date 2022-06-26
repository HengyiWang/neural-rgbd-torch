# import load_network_model
import os
import scene_bounds

import numpy as np
import torch

import marching_cubes as mcubes
import trimesh

from torch.cuda.amp import autocast as autocast


def get_batch_query_fn(query_fn, feature_array, network_fn):

    fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :], viewdirs=torch.zeros_like(f[i0:i1]),
                                    feature_array=feature_array,
                                    pose_array=None,
                                    frame_ids=torch.zeros_like(f[i0:i1, 0], dtype=torch.int32),
                                    deformation_field=None,
                                    c2w_array=None,
                                    network_fn=network_fn)

    return fn


def extract_mesh(query_fn, feature_array, network_fn, args, voxel_size=0.01, isolevel=0.0, scene_name='', mesh_savepath=''):

    # Query network on dense 3d grid of points
    voxel_size *= args.sc_factor  # in "network space"

    tx, ty, tz = scene_bounds.get_scene_bounds(scene_name, voxel_size, True)

    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])

    fn = get_batch_query_fn(query_fn, feature_array, network_fn)

    chunk = 1024 * 64
    with autocast():
        raw = [fn(flat, i, i + chunk)[0].cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = raw[..., -1]

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(sigma, isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / args.sc_factor - args.translation

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    # Transform the mesh to Scannet's coordinate system
    gl_to_scannet = np.array([[1, 0, 0, 0],
                              [0, 0, -1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]]).astype(np.float32).reshape([4, 4])

    mesh.apply_transform(gl_to_scannet)

    if mesh_savepath == '':
        mesh_savepath = os.path.join(args.basedir, args.expname, f"mesh_vs{voxel_size / args.sc_factor.ply}")
    mesh.export(mesh_savepath)

    print('Mesh saved')


if __name__ == '__main__':
    pass