# Reproducing the slice-selection audit

The main manuscript compares seven image-driven policies at 160 training masks per split. The privileged reference-mask controls use a separate information regime. The additional same-source sample has 20 patients reserved by the original protocol. The two training splits share the official test patients, so the analysis pairs and resamples patients across fitted splits.

The anonymized bundle includes aggregate outputs and, under `manifests/`, frozen patient roles and selected positions from the earlier experiment. Use the CREATIS releases described in [DATA.md](DATA.md) for images and reference masks. The scripts reconstruct checkpoints from those releases. Human annotation time was unavailable.

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

### Boundary-state acquisition follow-up

The three design files record the sequence of experiments. The uncalibrated round keeps the 12 detector-calibration cases. The calibrated and bounded rounds divide those cases into six selection reliability and six detector cases. The training roster and 160-mask training budget are constant. The 50 official test patients are a retrospective assessment. Patient bootstrap intervals hold all fitted models fixed.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for fold in 0 1; do
  for seed in 2027 2028 2029; do
    id=seed-$seed-fold-$fold
    common=(--data data/acdc-192 --registration registration-image-budget.json)
    seed_fold=(--seed "$seed" --fold "$fold")
    v1=runs/boundary-coverage/development/$id
    v2=runs/calibrated-boundary/development/$id
    v3=runs/bounded-boundary/development/$id
    python run_boundary_coverage.py fit "${common[@]}" "${seed_fold[@]}" \
      --original runs/sparse-image-cloud --design registration-boundary-coverage.json --output "$v1"
    python run_calibrated_boundary_coverage.py "${common[@]}" "${seed_fold[@]}" \
      --prior "$v1" --design registration-calibrated-boundary.json --output "$v2"
    python run_bounded_boundary_coverage.py "${common[@]}" \
      --prior "$v2" --design registration-bounded-boundary.json --output "$v3"
    for method in boundary-coverage calibrated-boundary bounded-boundary; do
      python run_boundary_coverage.py official --fit "runs/$method/development/$id" \
        --assessment data/acdc-challenge-audit-192 --design "registration-$method.json" \
        --output "runs/$method/official/$id"
    done
  done
done
for method in boundary-coverage calibrated-boundary bounded-boundary; do
  python summarize_boundary_coverage.py --root "runs/$method" --design "registration-$method.json" \
    --output "results/$method-summary-recomputed.json"
done
```

The released summaries additionally record exact replay comparisons from the original archived runs. Add `--archive-root runs` to the summarizer to check those runs if available. The bounded runner verifies upstream checkpoints before reusing controls. The summaries report acquired states, paired intervals, and seed-specific estimates. Full MRI volumes, masks and fitted weights remain outside the release archive.

Run the matched-weight acquisition-only control after the calibrated stage. The optional saved initial teachers are read from `runs/reviewer-weights`. The control reconstructs warm-up from the shared seed if a compatible checkpoint is absent.

```bash
mkdir -p runs/boundary-equal-weight
for fold in 0 1; do
  python audit_boundary_equal_weight.py score --data data/acdc-192 \
    --registration registration-image-budget.json --design registration-calibrated-boundary.json \
    --v2 runs/calibrated-boundary/development --warmups runs/reviewer-weights \
    --fold "$fold" --output "runs/boundary-equal-weight/fold-$fold.json"
done
python audit_boundary_equal_weight.py aggregate \
  --inputs runs/boundary-equal-weight/fold-0.json runs/boundary-equal-weight/fold-1.json \
  --output results/boundary-equal-weight-summary-recomputed.json
```

### Review follow-up diagnostics

Use the V2 and V3 runs from the boundary-state workflow. The matched-slot control holds the eight bounded patient–phase slots fixed, picks the nearest alternative by image-distance loss, and repeats continuation training with fixed seeds. State reliability uses development patients outside each fold's state fitting panel and replays both acquisition lists. The detector audit reuses frozen checkpoints and verifies both the six-patient cutoffs and the scores on the official 50 before writing anonymous numeric ledgers.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
swap=runs/boundary-swap-control
state=runs/boundary-state-reliability
detector=runs/boundary-detector-audit
mkdir -p "$swap" "$state" "$detector"
for fold in 0 1; do
  python audit_boundary_state_reliability.py score --data data/acdc-192 \
    --registration registration-image-budget.json --design registration-calibrated-boundary.json \
    --v2 runs/calibrated-boundary/development --warmups runs/reviewer-weights \
    --fold "$fold" --output "$state/fold-$fold.json"
  for seed in 2027 2028 2029; do
    id=seed-$seed-fold-$fold
    python run_boundary_swap_control.py --data data/acdc-192 --registration registration-image-budget.json \
      --prior "runs/bounded-boundary/development/$id" --design registration-boundary-swap-control.json \
      --output "$swap/development/$id"
    python run_boundary_coverage.py official --fit "$swap/development/$id" \
      --assessment data/acdc-challenge-audit-192 --design registration-boundary-swap-control.json \
      --output "$swap/official/$id"
    python audit_boundary_detector.py score --source data/acdc-192 --assessment data/acdc-challenge-audit-192 \
      --fit "runs/bounded-boundary/development/$id" --archived "runs/bounded-boundary/official/$id" \
      --design registration-boundary-detector-audit.json --output "$detector/$id.json.gz"
  done
done
python summarize_boundary_swap_control.py --root "$swap" --design registration-boundary-swap-control.json \
  --assessment data/acdc-challenge-audit-192 --output "$swap/summary.json"
python audit_boundary_state_reliability.py aggregate --inputs "$state/fold-0.json" "$state/fold-1.json" \
  --output "$state/summary.json"
python audit_boundary_detector.py replay --files "$detector"/seed-*.json.gz \
  --design registration-boundary-detector-audit.json --output "$detector/summary.json"
```

The last command can replay from the numeric ledgers in Artifact A1 without images. Table 4 in the reviewer revision uses the original pooled, centered assessment at 20% FPR. The per-class matched ROC is invariant to a common subtraction in that class. Cutoff resampling re-estimates performance at the six-source-patient operating point, with the original fewer-than-five fallback. Both folds share the official 50 patients for all bootstrap draws. Patient intervals condition on the fixed fitted seeds.

### Common-support analysis

The anonymous reproduction archive contains a compressed numeric ledger for the common-support and source-calibration analysis. The ledger uses artificial patient keys and sorted within-support scores. It includes no source images, masks, original case keys, or slice coordinates. Extract the archive and run the analysis with NumPy, SciPy and scikit-learn. The saved JSON includes separate RV and LV results, area cells, nominal FPR curves, opposite-ventricle support, and paired contrasts. The original three-class ACDC table uses centered scores for both matched recall and standardized partial AUC.

```bash
OPENBLAS_NUM_THREADS=2 python audit_review_common_support.py replay \
  --ledger data/review2-common-support-ledger.json.gz --draws 4000 --output results/recomputed-common-support.json
```

### Independent cine validation of the frozen policies

Use the ACDC checkpoints from the bounded and matched-slot runs. This workflow reads source cutoff panels, verifies checkpoint digests and evaluates all source fits on the independent data. The primary comparison uses the ordinary array-aspect resize. The separate geometry diagnostic matches ACDC input pixel spacing and preserves phase indices and reference eligibility.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python download_cmr_multi.py --output data/cmr-multi-raw
python prepare_cmr_multi.py --raw data/cmr-multi-raw --output data/cmr-multi-192
python audit_cmr_multi_cycle_support.py --raw data/cmr-multi-raw --cache data/cmr-multi-192 \
  --output data/cmr-multi-192/cycle-eligibility.json
python prepare_cmr_multi_fov.py --raw data/cmr-multi-raw --source data/acdc-192 \
  --primary data/cmr-multi-192 --output data/cmr-multi-fov-192
for variant in native fov; do
  if [ "$variant" = native ]; then
    data=data/cmr-multi-192
    design=registration-external-cine.json
  else
    data=data/cmr-multi-fov-192
    design=registration-external-cine-fov.json
  fi
  out="runs/external-cine-$variant"
  for fold in 0 1; do
    for seed in 2027 2028 2029; do
      python audit_external_cine.py score --source data/acdc-192 --target "$data" \
        --bounded runs/bounded-boundary/development --swap runs/boundary-swap-control/development \
        --design "$design" --seed "$seed" --fold "$fold" --output "$out/seed-$seed-fold-$fold.json.gz"
    done
  done
  python audit_external_cine.py replay --files "$out"/seed-*.json.gz --design "$design" --output "$out/summary.json"
done
python audit_cmr_multi_phase_agreement.py --coverage data/cmr-multi-192/coverage.json \
  --files runs/external-cine-native/seed-*.json.gz --design registration-external-cine.json \
  --output runs/external-cine-native/phase-agreement.json
```

The cine archive includes the native and physical-spacing ledgers, role and geometry receipts, and both aggregate results. The replay commands use those ledgers without re-running inference. Three fitted seeds and both folds share a single resample of target patients. Only cutoff-sensitivity intervals resample the six ACDC source patients in each fold. A blank class-specific endpoint denotes insufficient verified reference support.

### Submission diagnostics for cine

The geometry audit measures a paired change in the acquisition contrast under fixed weights, cutoffs, and patient support. Its ventricular-only comparison uses the same cases and class groups with and without subtracting the frozen cutoffs. This diagnostic is separate from the registered primary operating point.

```bash
python audit_cine_geometry_interaction.py --native runs/external-cine-native \
  --physical runs/external-cine-fov --native-protocol registration-external-cine.json \
  --physical-protocol registration-external-cine-fov.json \
  --design registration-cine-geometry-interaction.json --output runs/cine-geometry-interaction/summary.json
```

The exploratory myocardial audit uses 16 cycle-supported blank planes from 12 patients, with only three adjacent planes. Generate `cycle-eligibility.json` for both input caches with `audit_cmr_multi_cycle_support.py`. Frozen detector cutoffs are verified against the original cine ledgers. GPU inference preserves each original case batch boundary and retains only the selected numeric scores.

```bash
out=runs/cine-myocardium-cycle
for fold in 0 1; do
  for seed in 2027 2028 2029; do
    python audit_cmr_multi_myocardium_cycle.py score \
      --native data/cmr-multi-192 --fov data/cmr-multi-fov-192 \
      --bounded runs/bounded-boundary/development --swap runs/boundary-swap-control/development \
      --native-archive runs/external-cine-native --fov-archive runs/external-cine-fov \
      --design registration-cmr-multi-myocardium-cycle.json --fold "$fold" --seed "$seed" \
      --output "$out/seed-$seed-fold-$fold.json.gz"
  done
done
python audit_cmr_multi_myocardium_cycle.py replay --files "$out"/seed-*.json.gz \
  --design registration-cmr-multi-myocardium-cycle.json --output "$out/summary.json"
```

To regenerate the ledger from the original dataset releases, use the frozen ACDC and HVSMR score outputs from the preceding sections. Run `extract_review_calibration.py` per fold with those original and replay paths. The `cache-acdc-native` and `compare` commands of `audit_review_common_support.py` combine the unresized official ACDC reference areas, external cache, frozen score records and calibration score records. Use the original and manual-source HVSMR grouping recorded in its preparation step.

For the candidate audit, reuse the original and strong-seed runs above. Run `audit_candidate_stage.py` with `--repeat-seed`, `--fold`, `--registration registration-image-budget.json` and `--seed-runs runs/strong-seed` in addition to the original candidate audit arguments. Pass the six replay JSONs to `--repeat-files` to regenerate the aggregated seed stability table. The analysis compares reconstructed acquisition rosters against previously scored ones. Its inference resamples patients within each fixed training fold.

The CHLA release's nearest-neighbor data are pinned to the original authors' repository revision `1e0d483045cbcd4d172156152c6ab7e8169528a4`. Fetch the corresponding source tar.gz and pass it to `prepare_chla.py --archive <archive> --out-root data/chla-positive-192`. The script verifies the archive digest. Run `audit_chla_positive.py score` with the same source arguments as the HVSMR score command, omitting `--coverage`, and choose either the `lv` or `rv` subdirectory under the external cache. Run `aggregate` on the two fold outputs for that class. The published release supplies contoured positives only for the screening scope used here.
