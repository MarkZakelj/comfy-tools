import numpy as np
import torch
import torchvision


def select_from_idx(elements: list, idx: list):
    return [elements[i] for i in idx if i < len(elements)]


# def resize_image_list(img_list: list, new_size: tuple):
#     """
#     :param img_list: list of images
#     :param new_size: (width, height)
#     :return:
#     """
#     new_list = []
#     for img in img_list:
#         if torch.is_tensor(img):
#             new_list.append(torchvision.resize)
#         new_list.append((img, new_size))


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()