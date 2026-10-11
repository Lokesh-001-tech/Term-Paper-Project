import os
import shutil
import subprocess
from pathlib import Path
import sys
def main():
    base_dir = Path("results").resolve()
    with_ref_dir = base_dir / "with_reference"
    without_ref_dir = base_dir / "without_reference"
    with_ref_dir.mkdir(parents=True, exist_ok=True)
    without_ref_dir.mkdir(parents=True, exist_ok=True)
    datasets_found = False
    for category_dir, has_ref in [(with_ref_dir, True), (without_ref_dir, False)]:
        if not category_dir.exists():
            continue
        for dataset_dir in category_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            raw_dir = dataset_dir / "raw"
            if not raw_dir.exists():
                print(f"Skipping {dataset_dir.name} - No 'raw' folder found inside it.")
                continue
            ref_dir = dataset_dir / "reference"
            if has_ref and not ref_dir.exists():
                print(f"Skipping {dataset_dir.name} - No 'reference' folder found inside it (required for with_reference).")
                continue
            datasets_found = True
            print(f"\n=======================================================")
            print(f" Processing Dataset: {dataset_dir.name} ({'With Reference' if has_ref else 'Without Reference'})")
            print(f"=======================================================")
            restored_dir = dataset_dir / "restored"
            restored_dir.mkdir(parents=True, exist_ok=True)
            print(f"-> Restoring images from {raw_dir.name}...")
            all_images = [p for p in raw_dir.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if not all_images:
                print("No images found in raw directory.")
                continue
            images = []
            for p in all_images:
                restored_path = restored_dir / f"{p.stem}_restored{p.suffix}"
                restored_path_png = restored_dir / f"{p.stem}_restored.png"
                if not restored_path.exists() and not restored_path_png.exists():
                    images.append(str(p))
            print(f"-> Found {len(images)} images left to restore out of {len(all_images)}.")
            for i in range(0, len(images), 10):
                chunk = images[i:i+10]
                cmd = [sys.executable, "src/inference.py"] + chunk + ["--out", str(restored_dir), "--no-compare"]
                if not has_ref:
                    cmd.append("--post")
                subprocess.run(cmd, check=True)
            print(f"-> Evaluating quality metrics...")
            eval_cmd = [
                sys.executable, "src/evaluate.py",
                "--methods", f"Proposed={str(restored_dir)}", f"Raw={str(raw_dir)}",
                "--out", str(dataset_dir)
            ]
            if has_ref:
                eval_cmd = [
                    sys.executable, "src/evaluate.py",
                    "--ref", str(ref_dir),
                    "--methods", f"Proposed={str(restored_dir)}", f"Raw={str(raw_dir)}",
                    "--out", str(dataset_dir)
                ]
            subprocess.run(eval_cmd, check=True)
            print(f"-> Done! Results saved in {dataset_dir.name}")
    if not datasets_found:
        print("No datasets found! Please create a folder like 'results/without_reference/MyDataset/raw/' and put images in it.")
if __name__ == "__main__":
    main()
