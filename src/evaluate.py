"""
PSNR, SSIM and Entropy for one or more methods.
PSNR / SSIM need a reference image (use --ref). Entropy does not, so without
--ref only Entropy is reported (use this for challenging-60).
With references (author dataset):
  python src/evaluate.py --ref data/reference-890 --methods Raw=data/raw-sub Proposed=results/proposed_890
Without references (challenging-60):
  python src/evaluate.py --methods Raw=test_data/challenging-60 Proposed=outputs
Install once:  pip install scikit-image pillow numpy matplotlib
"""
import argparse
import csv
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
from uiqm import get_uiqm
EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def normalize_stem(stem):
    """Normalize file stem so 'image_restored' matches 'image'."""
    return stem[:-9] if stem.endswith("_restored") else stem
def list_images(folder):
    images = {}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(EXT):
            stem = os.path.splitext(f)[0]
            if stem.endswith("_compare"):
                continue  
            norm_key = normalize_stem(stem)
            if norm_key not in images or stem.endswith("_restored"):
                images[norm_key] = os.path.join(folder, f)
    return images
def load(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)
def entropy(img):
    """Shannon entropy (bits) of the grayscale histogram. Higher = more detail/information."""
    gray = np.array(Image.fromarray(img).convert("L"))
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="reference folder (optional; needed for PSNR/SSIM)")
    ap.add_argument("--methods", nargs="+", required=True, help="Name=folder pairs")
    ap.add_argument("--out", default="results", help="folder for CSV and plots")
    ap.add_argument("--figure", type=int, default=0, help="side-by-side examples (needs --ref)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    refs = list_images(args.ref) if args.ref else None
    methods = {}
    for item in args.methods:
        name, folder = item.split("=", 1)
        methods[name] = list_images(folder)
    rows = []  
    for name, files in methods.items():
        for stem, path in files.items():
            if refs is not None and stem not in refs:
                continue  
            img = load(path)
            if refs is not None:
                ref = load(refs[stem])
                img = load(path, size=(ref.shape[1], ref.shape[0]))
                p = psnr(ref, img, data_range=255)
                s = ssim(ref, img, channel_axis=2, data_range=255)
            else:
                p = s = float("nan")
            uiqm_val, uicm_val, uism_val, uiconm_val = get_uiqm(img)
            rows.append((stem, name, p, s, entropy(img), uiqm_val, uicm_val, uism_val, uiconm_val))
    if not rows:
        raise SystemExit("No images matched. Check folder paths and that file names match the reference folder.")
    with open(os.path.join(args.out, "metrics_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "method", "PSNR", "SSIM", "Entropy", "UIQM", "UICM", "UISM", "UICONM"])
        for a, b, c, d, e, f_val, g_val, h_val, i_val in rows:
            w.writerow([a, b, f"{c:.4f}", f"{d:.4f}", f"{e:.4f}", f"{f_val:.4f}", f"{g_val:.4f}", f"{h_val:.4f}", f"{i_val:.4f}"])
    use_ref = refs is not None
    summary = []
    for name in methods:
        sel = [r for r in rows if r[1] == name]
        if not sel:
            continue
        P, S, E, UI, UC, US, UO = (np.array([r[i] for r in sel]) for i in (2, 3, 4, 5, 6, 7, 8))
        summary.append((name, len(sel), P.mean(), P.std(), S.mean(), S.std(), E.mean(), E.std(),
                        UI.mean(), UI.std(), UC.mean(), UC.std(), US.mean(), US.std(), UO.mean(), UO.std()))
    head = f"{'Method':<14}{'N':>5}"
    if use_ref:
        head += f"{'PSNR (dB)':>19}{'SSIM':>19}"
    head += f"{'Entropy (bits)':>19}{'UIQM':>19}{'UISM':>19}"
    print("\n" + head)
    print("-" * len(head))
    for name, n, pm, ps, sm, ss, em, es, uim, uis, ucm, ucs, usm, uss, uom, uos in summary:
        line = f"{name:<14}{n:>5}"
        if use_ref:
            line += f"{pm:>12.2f} ± {ps:<5.2f}{sm:>12.4f} ± {ss:<6.4f}"
        line += f"{em:>12.3f} ± {es:<5.3f}{uim:>12.3f} ± {uis:<5.3f}{usm:>12.3f} ± {uss:<5.3f}"
        print(line)
    with open(os.path.join(args.out, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "N", "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std", 
                    "Entropy_mean", "Entropy_std", "UIQM_mean", "UIQM_std", 
                    "UICM_mean", "UICM_std", "UISM_mean", "UISM_std", "UICONM_mean", "UICONM_std"])
        for r in summary:
            w.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])
    names = [s[0] for s in summary]
    if len(names) >= 2:
        a, b = names[-2], names[-1]
        da = {r[0]: r for r in rows if r[1] == a}
        db = {r[0]: r for r in rows if r[1] == b}
        common = sorted(set(da) & set(db))
        parts = []
        metrics = [("PSNR", 2), ("SSIM", 3)] if use_ref else []
        metrics.append(("Entropy", 4))
        for label, i in metrics:
            parts.append(f"{label} {sum(db[k][i] > da[k][i] for k in common)}/{len(common)}")
        print(f"\n{b} beats {a} on: " + ", ".join(parts))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = [("PSNR (dB)", 2), ("SSIM", 4)] if use_ref else []
        panels.append(("Entropy (bits)", 6))
        fig, ax = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), squeeze=False)
        for i, (title, idx) in enumerate(panels):
            ax[0][i].bar(names, [s[idx] for s in summary], yerr=[s[idx + 1] for s in summary], capsize=4)
            ax[0][i].set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "metrics_bar.png"), dpi=200)
        plt.close()
        if use_ref and args.figure > 0:
            stems = [s for s in refs if all(s in m for m in methods.values())][: args.figure]
            cols = list(methods) + ["Reference"]
            fig, axes = plt.subplots(len(stems), len(cols), figsize=(3 * len(cols), 2.4 * len(stems)), squeeze=False)
            for r, stem in enumerate(stems):
                ref = load(refs[stem])
                for c, name in enumerate(cols):
                    img = ref if name == "Reference" else load(methods[name][stem], (ref.shape[1], ref.shape[0]))
                    axes[r][c].imshow(img)
                    axes[r][c].axis("off")
                    if r == 0:
                        axes[r][c].set_title(name)
                    if name != "Reference":
                        pr = psnr(ref, img, data_range=255)
                        sm = ssim(ref, img, channel_axis=2, data_range=255)
                        axes[r][c].text(4, img.shape[0] - 6, f"PSNR {pr:.1f}  SSIM {sm:.3f}",
                                        color="white", fontsize=7, backgroundcolor="black")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "comparison.png"), dpi=200)
            plt.close()
    except ImportError:
        print("matplotlib not installed: skipped plots (pip install matplotlib)")
    print(f"\nSaved CSVs and plots in: {args.out}/")
if __name__ == "__main__":
    main()
