import numpy as np
import cv2
import math
def uicm(img):
    b, g, r = cv2.split(img)
    rg = r.astype(np.float32) - g.astype(np.float32)
    yb = 0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    return -0.0268 * np.sqrt(rg_mean**2 + yb_mean**2) + 0.1586 * np.sqrt(rg_std**2 + yb_std**2)
def eme(ch, block_size=8):
    h, w = ch.shape
    h_b, w_b = h // block_size, w // block_size
    if h_b == 0 or w_b == 0:
        return 0.0
    ch_crop = ch[:h_b*block_size, :w_b*block_size]
    blocks = ch_crop.reshape(h_b, block_size, w_b, block_size).transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    max_vals = np.max(blocks, axis=(1, 2))
    min_vals = np.min(blocks, axis=(1, 2))
    val = np.where((min_vals > 0) & (max_vals > min_vals), max_vals / (min_vals + 1e-4), 1.0)
    eme_val = 2.0 / (h_b * w_b) * np.sum(np.log(val + 1e-4))
    return eme_val
def uism(img):
    b, g, r = cv2.split(img)
    def sobel(c):
        sx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sx**2 + sy**2)
    rs, gs, bs = sobel(r), sobel(g), sobel(b)
    return 0.299 * eme(rs) + 0.587 * eme(gs) + 0.114 * eme(bs)
def plip_log(x):
    return 255.0 - 255.0 * ((255.0 - x) / 255.0)
def uiconm(img, block_size=8):
    b, g, r = cv2.split(img)
    rg = r.astype(np.float32) - g.astype(np.float32)
    yb = 0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32)
    h, w = rg.shape
    h_b, w_b = h // block_size, w // block_size
    if h_b == 0 or w_b == 0:
        return 0.0
    def ame(ch):
        ch_crop = ch[:h_b*block_size, :w_b*block_size]
        blocks = ch_crop.reshape(h_b, block_size, w_b, block_size).transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
        max_vals = np.max(blocks, axis=(1, 2))
        min_vals = np.min(blocks, axis=(1, 2))
        val = np.where((min_vals > 0) & (max_vals > min_vals), max_vals / (min_vals + 1e-4), 1.0)
        return 1.0 / (h_b * w_b) * np.sum(np.log(val + 1e-4))
    return 0.0282 * ame(rg) + 0.2953 * ame(yb)
def get_uiqm(img):
    c1, c2, c3 = 0.468, 0.274, 0.257
    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    uicm_val = uicm(bgr)
    uism_val = uism(bgr)
    uiconm_val = uiconm(bgr)
    uiqm_val = c1 * uicm_val + c2 * uism_val + c3 * uiconm_val
    return uiqm_val, uicm_val, uism_val, uiconm_val
