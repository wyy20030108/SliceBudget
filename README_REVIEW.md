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

## External verified-absence audit

HVSMR-2.0 provides full-volume chamber references for static axial MRI at a separate hospital. Use the cropped, non-normalized NIfTI variant and the technical CSV from the releases listed in [DATA.md](DATA.md). Preserve their original file names. The converter reorients the original axial grid and uses the native reference for negative eligibility. It writes a local reference-group map from the technical CSV for the 20 manual-reference sensitivity run.

```bash
python download_hvsmr.py --output data/hvsmr2-cropped
python prepare_hvsmr.py --raw data/hvsmr2-cropped --output data/hvsmr2-axial-192 \
  --technical-csv data/hvsmr2-cropped/hvsmr_technical.csv
for fold in 0 1; do
  CUDA_VISIBLE_DEVICES=$fold python audit_verified_negatives.py score \
    --data data/acdc-192 --external data/hvsmr2-axial-192 --coverage data/hvsmr2-axial-192/coverage.json \
    --registration registration-image-budget.json --fold "$fold" \
    --original "runs/sparse-image-cloud/fold-$fold" --official-original "runs/sparse-official-cloud/image-$fold" \
    --baseline-runs "runs/acquisition-baselines/fold-$fold" --core-weights "runs/review-core/fold-$fold" \
    --extra-weights "runs/reviewer-weights/fold-$fold" --source-audit "runs/reviewer-audit/fold-$fold.json" \
    --output "runs/verified-hvsmr2/fold-$fold.json"
done
python audit_verified_negatives.py aggregate --folds runs/verified-hvsmr2/fold-0.json runs/verified-hvsmr2/fold-1.json \
  --reference-groups data/hvsmr2-axial-192/reference_groups.json --output runs/verified-hvsmr2/summary.json
```

The fold files and reference-group map are used locally. The distributed JSON gives patient-macro rates, support counts, paired bootstrap intervals, and retrospective recall at 20% external FPR. The method retains the supplied ACDC thresholds for observed FPR and small-slice recall. Evaluation is limited to the two cavities in the static axial acquisition.

## Second review analyses

The anonymous reproduction archive contains a compressed numeric ledger for the common-support and source-calibration analysis. The ledger uses artificial patient keys and sorted within-support scores. It includes no source images, masks, original case keys, or slice coordinates. Extract the archive and run the analysis with NumPy, SciPy and scikit-learn. The saved JSON includes separate RV and LV results, area cells, nominal FPR curves, opposite-ventricle support, and paired contrasts. The original three-class ACDC table uses centered scores for both matched recall and standardized partial AUC.

```bash
OPENBLAS_NUM_THREADS=2 python audit_review_common_support.py replay \
  --ledger data/review2-common-support-ledger.json.gz --draws 4000 --output results/recomputed-common-support.json
```

To regenerate the ledger from the original dataset releases, use the frozen ACDC and HVSMR score outputs from the preceding sections. Run `extract_review_calibration.py` per fold with those original and replay paths. The `cache-acdc-native` and `compare` commands of `audit_review_common_support.py` combine the unresized official ACDC reference areas, external cache, frozen score records and calibration score records. Use the original and manual-source HVSMR grouping recorded in its preparation step.

For the candidate audit, reuse the original and strong-seed runs above. Run `audit_candidate_stage.py` with `--repeat-seed`, `--fold`, `--registration registration-image-budget.json` and `--seed-runs runs/strong-seed` in addition to the original candidate audit arguments. Pass the six replay JSONs to `--repeat-files` to regenerate the aggregated seed stability table. The analysis compares reconstructed acquisition rosters against previously scored ones. Its inference resamples patients within each fixed training fold.

The CHLA release's nearest-neighbor data are pinned to the original authors' repository revision `1e0d483045cbcd4d172156152c6ab7e8169528a4`. Fetch the corresponding source tar.gz and pass it to `prepare_chla.py --archive <archive> --out-root data/chla-positive-192`. The script verifies the archive digest. Run `audit_chla_positive.py score` with the same source arguments as the HVSMR score command, omitting `--coverage`, and choose either the `lv` or `rv` subdirectory under the external cache. Run `aggregate` on the two fold outputs for that class. The published release supplies contoured positives only for the screening scope used here.
