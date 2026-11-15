# Reproducing the slice-selection audit

The main manuscript compares seven image-driven policies at 160 training masks per split. The privileged reference-mask controls use a separate information regime. The additional same-source sample has 20 patients reserved by the original protocol. The two training splits share the official test patients, so the analysis pairs and resamples patients across fitted splits.

The anonymized bundle includes aggregate outputs and, under `manifests/`, the frozen patient roles and selected positions. Use the CREATIS releases described in [DATA.md](DATA.md) for images and reference masks. The scripts reconstruct checkpoints from those releases. Clinical time and verified external absences were outside the available audit.

## Environment and original runs

Use Python 3.12 and a CUDA-enabled PyTorch installation. Experiments used PyTorch 2.8.0+cu128 on RTX 4090D with the versions in `requirements.txt`. The figure script additionally uses ReportLab, and the external DICOM converter uses pydicom. Obtain ACDC through the first code block of [README.md](README.md), then run its original two-fold commands for the image-driven source, official image runs, budget tradeoff and acquisition-baseline audits. The strong-seed replay also uses the archived seed-2028 official image run, which the README shows how to generate. Use fresh output directories.

## Revision analyses

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for fold in 0 1; do
  orig="runs/sparse-image-cloud/fold-$fold"
  official="runs/sparse-official-cloud/image-$fold"
  baseline="runs/acquisition-baselines/fold-$fold"
  mixed="runs/sparse-factorial-cloud/fold-$fold"
  core="runs/review-core/fold-$fold"
  extra="runs/reviewer-weights/fold-$fold"
  common=(--data data/acdc-192 --registration registration-image-budget.json --original "$orig" --official-original "$official" --fold "$fold")
  python audit_external_transfer.py prepare "${common[@]}" --output "$core"
  python replay_audit_baselines.py "${common[@]}" \
    --assessment data/acdc-challenge-audit-192 --baselines "$baseline" --factorial "$mixed" --output "$extra"
  python audit_stratified_decisions.py score "${common[@]}" \
    --official data/acdc-challenge-audit-192 --core-weights "$core" --extra-weights "$extra" \
    --baseline-runs "$baseline" --mixed-runs "$mixed" --output "runs/reviewer-audit/fold-$fold.json" --sealed
  for seed in 2028 2029; do
    archive=()
    if [ "$seed" = 2028 ]; then
      archive=(--archived-2028 "runs/acquisition-seed-2028/fold-$fold")
    fi
    python audit_strong_seed_repeat.py run --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 \
      --official-original "$official" --acquisition-controls "$baseline" --registration registration-image-budget.json \
      --fold "$fold" --seed "$seed" --output "runs/strong-seed/seed-$seed-fold-$fold" "${archive[@]}"
  done
done
python audit_stratified_decisions.py aggregate \
  --folds runs/reviewer-audit/fold-0.json runs/reviewer-audit/fold-1.json --output runs/reviewer-audit/summary.json
python audit_candidate_stage.py --data data/acdc-192 --runs runs/sparse-image-cloud \
  --weights runs/reviewer-weights --output runs/reviewer-candidates/summary.json
python audit_strong_seed_repeat.py aggregate --runs runs/strong-seed --output runs/strong-seed-aggregate.json
python plot_decision_curves.py runs/reviewer-audit/summary.json decision-curves.pdf
```

`registration-review-audit.json` records the follow-up comparison and eligibility rules. The three score rules each use cutoffs fitted on the source calibration patients. Retrospective matched FPR uses assessment labels to locate a point on an empirical ROC curve. Keep the local per-patient outputs separate from the included aggregate results. The 2028 and 2029 matched-recall comparisons are seed-specific point estimates, and the three seeds share the same assessment patients.
