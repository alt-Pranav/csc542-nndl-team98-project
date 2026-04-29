#!/usr/bin/env bash
echo "Running ablation sweep..."
python scripts/run_ablation.py --ckpt baby_unet_best.pth --channels "range(0, 128)" --from-ch 0 --to-ch 128 --out-csv outputs/ablation_sweep.csv
echo "Running bottleneck visualization..."
python scripts/run_bottleneck_viz.py --ckpt baby_unet_best.pth --out-dir outputs/bottleneck_viz --max-ch 128 --num-samples 3
echo "Running sparsity scoring..."
python scripts/run_sparsity.py --ckpt baby_unet_best.pth --out-csv outputs/sparsity_scores.csv
echo "Done!"
