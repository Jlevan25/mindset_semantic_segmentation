import time

import numpy as np
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


def one_hot_argmax(masks):
    batch, d, h, w = masks.shape
    preds = torch.zeros(batch * h * w, d)
    preds[torch.arange(len(preds)), masks.argmax(dim=1).reshape(-1)] = 1
    return preds.T.reshape(d, batch, h, w).transpose(1, 0)


def get_kernel_indexes1d(num_kernels, kernel_length, stride, repeat_axis):
    idx = np.tile(np.arange(kernel_length), num_kernels).reshape(-1, kernel_length)
    idx += stride * np.arange(num_kernels).reshape(-1, 1)
    return np.repeat(idx, kernel_length, axis=repeat_axis)


def get_kernel_indexes3d(kernel_shape, output_shape, stride):
    kernel_depth, kernel_h, kernel_w = kernel_shape
    output_h, output_w = output_shape

    k = np.arange(kernel_depth)
    k = np.repeat(k, kernel_h * kernel_w)
    k = np.tile(k, output_h * output_w)

    i = get_kernel_indexes1d(output_h, kernel_h, stride[0], repeat_axis=1)
    i = np.tile(i, kernel_depth)
    i = np.repeat(i, output_w, axis=0).reshape(-1)

    j = get_kernel_indexes1d(output_w, kernel_w, stride[1], repeat_axis=0)
    j = np.tile(j, kernel_depth)
    j = np.tile(j.reshape(-1), output_h)

    return k, i, j


@timer
def get_kernels(image, kernel_size, stride):
    h, w = image.shape[-2:]
    kernel_h, kernel_w = kernel_size[-2:]
    output_shape = (h - kernel_h) // stride[0] + 1, \
                   (w - kernel_w) // stride[1] + 1

    k, i, j = get_kernel_indexes3d(kernel_size, output_shape, stride)
    return image[..., k, i, j].reshape(*output_shape, *kernel_size), output_shape

#
# def _get_kernel_indexes1d(num_kernels, kernel_length, depth, stride, repeat_axis):
#     idx = np.tile(np.arange(kernel_length), num_kernels).reshape(-1, kernel_length)
#     idx += stride * np.arange(num_kernels).reshape(-1, 1)
#     idx = np.repeat(idx, kernel_length, axis=repeat_axis)
#     idx = np.tile(idx, depth)
#     return np.repeat(idx, num_kernels, axis=0).reshape(-1) if repeat_axis == 1\
#         else np.tile(idx.reshape(-1), num_kernels)
#
#
# def _get_kernel_indexes3d(kernel_shape, output_shape, stride):
#     kernel_depth, kernel_h, kernel_w = kernel_shape
#     output_h, output_w = output_shape
#
#     k = np.arange(kernel_depth)
#     k = np.repeat(k, kernel_h * kernel_w)
#     k = np.tile(k, output_w * output_h)
#
#     i = _get_kernel_indexes1d(output_w, kernel_h, kernel_depth, stride[0], repeat_axis=1)
#     j = _get_kernel_indexes1d(output_h, kernel_w, kernel_depth, stride[1], repeat_axis=0)
#
#     return k, i, j
