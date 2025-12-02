
import os
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb


def _pad_arr(arr, pad_size=10, pad_value=0):
    return np.pad(
        arr,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),   # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant',
        constant_values=pad_value)

def draw_per_panel_geo_imgs(surf_ncs, surf_mask, colors, pad_size=5, out_dir=''):
    n_surfs = surf_ncs.shape[0]
    reso = int(surf_ncs.shape[1] ** 0.5)

    framed_imgs = []

    _surf_ncs = surf_ncs.reshape(n_surfs, reso, reso, 3)
    _surf_mask = surf_mask.reshape(n_surfs, reso, reso, 1)

    for idx in range(n_surfs):
        mask_img = _surf_mask[idx, ...].astype(np.float32)
        _inv_mask_img = 1.0 - mask_img

        _padded_mask = _pad_arr(_inv_mask_img * 0.33, pad_size=pad_size, pad_value=1.0)

        _cur_color = colors[idx]
        if type(_cur_color) is str: _cur_color = to_rgb(_cur_color)

        _bg_img = np.zeros_like(_padded_mask.repeat(3, axis=-1)) + np.asarray(_cur_color)[None, None, :3]
        _bg_img = np.concatenate([_bg_img * _padded_mask, _padded_mask], axis=-1)

        _fg_img = np.concatenate([(np.clip(_surf_ncs[idx, ...], -1.0, 1.0) + 1.0) * 0.5, _surf_mask[idx, ...]], axis=-1)
        _fg_img = _pad_arr(_fg_img, pad_size=pad_size, pad_value=0.0)

        fused_img = _bg_img + _fg_img

        framed_imgs.append(fused_img)

        fused_pil_img = Image.fromarray((fused_img * 255).astype(np.uint8))

        os.makedirs(out_dir, exist_ok=True)
        fused_pil_img.save(os.path.join(out_dir, f'surf_{idx:02d}.png'))

    return framed_imgs