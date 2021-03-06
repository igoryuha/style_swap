import torch
import torch.nn.functional as F


def extract_image_patches_(image, kernel_size, strides):
    kh, kw = kernel_size
    sh, sw = strides
    patches = image.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)
    return patches


def style_swap(c_features, s_features, kernel_size, stride=1):

    s_patches = extract_image_patches_(s_features, [kernel_size, kernel_size], [stride, stride])
    s_patches_matrix = s_patches.reshape(s_patches.shape[0], -1)
    s_patch_wise_norm = torch.norm(s_patches_matrix, dim=1)
    s_patch_wise_norm = s_patch_wise_norm.reshape(-1, 1, 1, 1)
    s_patches_normalized = s_patches / (s_patch_wise_norm + 1e-8)
    # Computes the normalized cross-correlations.
    # At each spatial location, "K" is a vector of cross-correlations
    # between a content activation patch and all style activation patches.
    K = F.conv2d(c_features, s_patches_normalized, stride=stride)
    # Replace each vector "K" by a one-hot vector corresponding
    # to the best matching style activation patch.
    best_matching_idx = K.argmax(1, keepdim=True)
    one_hot = torch.zeros_like(K)
    one_hot.scatter_(1, best_matching_idx, 1)
    # At each spatial location, only the best matching style
    # activation patch is in the output, as the other patches
    # are multiplied by zero.
    F_ss = F.conv_transpose2d(one_hot, s_patches, stride=stride)
    overlap = F.conv_transpose2d(one_hot, torch.ones_like(s_patches), stride=stride)
    F_ss = F_ss / overlap
    return F_ss


def TVloss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def learning_rate_decay(optimizer, init_lr, global_step, decay_rate):

    lr = init_lr / (1. + global_step * decay_rate)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
