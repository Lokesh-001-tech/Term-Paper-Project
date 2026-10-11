Write-Host "Evaluating U45 (Already restored)"
python src/evaluate.py --methods Raw=test_data/U45 Proposed=outputs/U45 --out results/without_reference/U45

Write-Host "Evaluating challenging-60"
python src/evaluate.py --methods Raw=test_data/challenging-60 Proposed=outputs/challenging-60 --out results/without_reference/challenging-60

Write-Host "Running inference on raw-890... This will take about 1 hour."
python src/batch_restore.py data/raw-890

Write-Host "Evaluating raw-890 with references"
python src/evaluate.py --ref data/reference-890 --methods Raw=data/raw-890 Proposed=outputs/raw-890 --out results/with_reference/raw-890

Write-Host "All datasets processed and organized!"
