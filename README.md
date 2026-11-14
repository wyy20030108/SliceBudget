# 复查 ACDC 标注分配证据

源码和 `results/` 中的聚合 JSON 用于科研复核。数据下载和许可说明见 [DATA.md](DATA.md)。 [ACDC 数据入口](https://www.creatis.insa-lyon.fr/Challenge/acdc/) · [ACDC 引用](https://doi.org/10.1109/TMI.2018.2837502)

使用支持 CUDA 的 PyTorch 环境并安装 `requirements.txt`。本轮使用两张 RTX 4090D 和 PyTorch 2.8+cu128，Linux 环境将 OpenMP、MKL 和 OpenBLAS 线程数各设为 2。用空结果目录保存每次运行。

```bash
python download_acdc.py data/acdc-raw
python acdc_data.py --raw-root data/acdc-raw --out-root data/acdc-192 --size 192
python download_acdc_official_test.py --raw data/official-test-raw --cache data/acdc-challenge-audit-192
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

每种模型只从训练患者学习，在相互分开的患者上校准和评分。下面展示完整单卡顺序。`--fold 0` 和 `--fold 1` 可分别并行执行。

```bash
for fold in 0 1; do
  python run_matched_slice_budget.py --data data/acdc-192 --registration registration-sparse-v1.json --fold "$fold" --output "runs/sparse-matched-cloud/fold-$fold"
  python run_image_slice_budget.py --data data/acdc-192 --registration registration-image-budget.json --fold "$fold" --output "runs/sparse-image-cloud/fold-$fold"
  python run_matched_slice_budget.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-sparse-v1.json --fold "$fold" --output "runs/sparse-official-cloud/truth-$fold"
  python run_image_slice_budget.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --fold "$fold" --output "runs/sparse-official-cloud/image-$fold"
  python audit_budget_tradeoff.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --prior "runs/sparse-image-cloud/fold-$fold" --output "runs/sparse-factorial-cloud/fold-$fold" --fold "$fold"
  python audit_supervision_regime.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --original "runs/sparse-image-cloud/fold-$fold" --distribution "runs/sparse-factorial-cloud/fold-$fold" --output "runs/sparse-supervised-cloud/fold-$fold" --fold "$fold"
done
python summarize_annotation_evidence.py --runs runs --output runs/annotation-evidence-aggregate.json
python audit_target_reliability.py --data data/acdc-192 --runs runs/sparse-image-cloud --output runs/annotation-target-audit.json
```

固定 160 张训练掩膜时可在两种已冻结的切片名单上比较 UOT 代码路径。预算敏感性实验给每种策略加上完全相同的 80 张空间补充掩膜。两项实验复用上述流程的校准患者和官方测试批次。

```bash
for fold in 0 1; do
  python audit_prcv_bridge.py run --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --original "runs/sparse-image-cloud/fold-$fold" --official-original "runs/sparse-official-cloud/image-$fold" --output "runs/extension/prcv-bridge-fold-$fold" --fold "$fold"
  python audit_mask_budget.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --original "runs/sparse-image-cloud/fold-$fold" --output "runs/extension/budget-fold-$fold" --fold "$fold"
done
python audit_prcv_bridge.py aggregate --folds runs/extension/prcv-bridge-fold-0 runs/extension/prcv-bridge-fold-1 --output runs/prcv-bridge-aggregate.json
python summarize_budget_audit.py --experiment runs/extension --original runs/sparse-official-cloud --output runs/mask-budget-aggregate.json
```

复现时，实际数据使用应遵守其研究许可。论文或补充包如使用这些数据，引用 Bernard 等的原始论文并按会方匿名要求处理。报告的生成式 AI 参与、可接受的使用范围和正式投稿由作者按 PAKDD 规则核实。 [PAKDD 作者指南](https://www.pakdd2027.org/pages/calls/authors-kit)
