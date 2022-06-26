import torch
from nerf_helpers import img2mae, img2mse


def compute_loss(prediction, target, loss_type='l2'):
    if loss_type == 'l2':
        return img2mse(prediction, target)
    elif loss_type == 'l1':
        return img2mae(prediction, target)

    raise Exception('Unsupported loss type')


def get_masks(z_vals, target_d, truncation):

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))

    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))

    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))

    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, loss_type):

    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)

    fs_loss = compute_loss(predicted_sdf * front_mask, torch.ones_like(predicted_sdf) * front_mask, loss_type) * fs_weight
    sdf_loss = compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask, target_d * sdf_mask, loss_type) * sdf_weight

    return fs_loss, sdf_loss


def get_depth_loss(pred, gt, loss_type='l2'):
    depth_mask = torch.where(gt > 0, torch.ones_like(gt), torch.zeros_like(gt))
    eps = 1e-4
    num_pixel = depth_mask.nelement()
    num_valid = torch.count_nonzero(depth_mask) + eps 
    depth_valid_weight = num_pixel / num_valid

    return compute_loss(pred[..., None] * depth_mask, gt * depth_mask, loss_type) * depth_valid_weight