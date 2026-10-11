"""
Restore images with src/inference.py and sort the results automatically:
  results/proposed_<name>/<image>_restored.<ext>   <- the enhanced image (used for scoring)
  outputs/<image>_compare.<ext>                    <- the before/after comparison picture
Run from the project root. The target is a folder OR a single image:
  python src/batch_restore.py test_data/U45                # every image in the folder
  python src/batch_restore.py test_data/U45/12.png         # one image
  python src/batch_restore.py test_data/challenging-60 --name c60
<name> defaults to the folder name (U45 -> u45, challenging-60 -> challenging_60).
  --skip-done   skip images that already have a *_restored file in the results folder
  --recursive   also look inside sub-folders of a folder target
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
def is_result(p: Path) -> bool:
    return p.stem.endswith("_restored") or p.stem.endswith("_compare")
def find_new(folder: Path, stem: str, tag: str, since: float):
    """Files <stem>_<tag>.* in folder written at/after 'since'."""
    return [f for f in folder.glob(f"{stem}_{tag}.*")
            if f.is_file() and f.suffix.lower() in EXT and f.stat().st_mtime >= since - 1]
def move_to(files, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest / f.name
        if target.exists() and target.resolve() != f.resolve():
            target.unlink()
        if target.resolve() != f.resolve():
            shutil.move(str(f), str(target))
def derive_name(target: Path) -> str:
    base = target.parent.name if target.is_file() else target.name
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_") or "run"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="folder of images, or one image")
    ap.add_argument("--name", default=None, help="results sub-folder name: results/proposed_<name>")
    ap.add_argument("--inference", default="src/inference.py")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()
    target = Path(args.target).resolve()
    if not target.exists():
        raise SystemExit(f"Not found: {target}")
    base_out = Path(args.outputs).resolve()
    if target.is_dir():
        name = args.name or target.name
        out_dir = base_out / name
    else:
        out_dir = base_out
    res_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        images = [target]
        src_dir = target.parent
    else:
        src_dir = target
        pattern = "**/*" if args.recursive else "*"
        files = [p for p in sorted(src_dir.glob(pattern)) if p.is_file() and p.suffix.lower() in EXT]
        stray = [p for p in files if is_result(p)]
        if stray:
            print(f"Sorting {len(stray)} result file(s) found inside {src_dir.name}/")
            for p in stray:
                move_to([p], res_dir if p.stem.endswith("_restored") else out_dir)
        images = [p for p in files if not is_result(p)]
    stems = [p.stem for p in images]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        print(f"WARNING: duplicate file names (results would overwrite each other): {', '.join(dupes[:10])}")
    print(f"{len(images)} image(s) -> both restored and compare files in {out_dir.relative_to(Path.cwd().resolve()).as_posix() if out_dir.is_relative_to(Path.cwd().resolve()) else out_dir.name}/\n")
    failed, missing = [], []
    for i, img in enumerate(images, 1):
        stem = img.stem
        if args.skip_done and list(res_dir.glob(f"{stem}_restored.*")):
            print(f"[{i}/{len(images)}] {img.name}: already restored, skipping")
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, args.inference, str(img)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{i}/{len(images)}] {img.name}: inference FAILED")
            print("   " + (r.stderr.strip().splitlines() or ["(no error text)"])[-1])
            failed.append(img.name)
            continue
        restored = find_new(img.parent, stem, "restored", t0) + find_new(base_out, stem, "restored", t0)
        compare = find_new(img.parent, stem, "compare", t0) + find_new(base_out, stem, "compare", t0)
        move_to(restored, res_dir)
        move_to(compare, out_dir)
        if not restored:
            print(f"[{i}/{len(images)}] {img.name}: no {stem}_restored file found")
            missing.append(img.name)
        else:
            print(f"[{i}/{len(images)}] {img.name}: done in {time.time() - t0:.1f}s")
    got = sum(1 for s in stems if list(res_dir.glob(f"{s}_restored.*")))
    print(f"\nImages: {len(images)}   Restored images in {res_dir.name}: {got}")
    if failed:
        print(f"Inference failed for {len(failed)}: {', '.join(failed[:10])}")
    if missing:
        print(f"No restored file for {len(missing)}: {', '.join(missing[:10])}")
    if target.is_dir():
        try:
            rel = res_dir.relative_to(Path.cwd().resolve()).as_posix()
            src_rel = target.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            rel, src_rel = str(res_dir), str(target)
        folder_name = out_dir.name
        print(f"Next: python src/evaluate.py --methods Raw={src_rel} Proposed={rel} --out results_{folder_name}")
if __name__ == "__main__":
    main()
