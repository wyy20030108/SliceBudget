# SliceBudget

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

### 等预算追加对照

在原 ACDC 协议及 160 张掩膜预算下，追加均匀随机选片及图像差异选片。后者比较候选切片与同一患者同一期的初始切片，取中心化并归一化的 16×16 块均值特征欧氏距离最大的候选。追加对照在主实验后设计，结果仍使用同一批 ACDC 测试患者。随机选片每折固定一个清单。

```bash
for fold in 0 1; do
  python audit_acquisition_baselines.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget.json --prior "runs/sparse-image-cloud/fold-$fold" --official-prior "runs/sparse-official-cloud/image-$fold" --output "runs/acquisition-baselines/fold-$fold" --fold "$fold"
done
python summarize_acquisition_baselines.py --runs runs/acquisition-baselines --official-prior runs/sparse-official-cloud --output runs/acquisition-baselines/summary-public.json
```

### 随机种子和检测操作点复查

第二个训练种子复用原两折患者，并重新训练、选择切片和校准阈值，评估结果另行报告。匹配误报率的分析使用原已冻结模型，在官方测试集按所有真缺失或邻近真缺失切片计算经验 ROC。所用测试标签仅确定回顾性匹配点。原可用阈值由 ACDC 校准患者独立确定。

```bash
for fold in 0 1; do
  python run_image_slice_budget.py --data data/acdc-192 --official-audit data/acdc-challenge-audit-192 --registration registration-image-budget-seed-2028.json --output "runs/acquisition-seed-2028/fold-$fold" --fold "$fold"
  common=(--data data/acdc-192 --registration registration-image-budget.json --original "runs/sparse-image-cloud/fold-$fold" --official-original "runs/sparse-official-cloud/image-$fold" --fold "$fold")
  weights="runs/decision-replay/fold-$fold"
  python audit_external_transfer.py prepare "${common[@]}" --output "$weights"
  for mode in score argmax; do
    python audit_decision_curves.py "$mode" "${common[@]}" --assessment data/acdc-challenge-audit-192 --weights "$weights" --output "runs/decision-curves/fold-$fold"
  done
done
python summarize_seed_repeat.py --original runs/sparse-official-cloud --repeat runs/acquisition-seed-2028 --output runs/acquisition-seed-2028/summary-public.json
python audit_decision_curves.py aggregate --folds runs/decision-curves/fold-0 runs/decision-curves/fold-1 --output runs/decision-curves/summary.json
python plot_decision_curves.py runs/decision-curves/summary.json decision-curves.pdf
```

绘图步骤需要 ReportLab。图由公开在本仓库的汇总值重建。

### 独立中心验证

外部影像只用于评分。沿用上述两折的 ACDC 选片名单、训练设置和类别阈值。重建四种策略的教师模型时核验其 ACDC 校准阈值是否与原运行一致。Sunnybrook 按原始人工轮廓评价左室腔舒张末期和收缩末期，并单独评价舒张末期左室心肌。York 儿科队列在所有时相均有轮廓的共同切片上汇总左室腔面积，按其最大值和最小值估计两个时相，分别评价左室腔和左室心肌。外部数据上的未勾画切片视为状态未知，评估限于对应类别有人工轮廓的阳性切片。检出召回使用源域阈值，外部误报率及部分 AUC 在结果中记为 `null`。数据来源及适用范围见 [DATA.md](DATA.md)。

```bash
python -m pip install pydicom==3.0.1
python prepare_sunnybrook.py --download --archives data/sunnybrook-sourceforge --out-root data/sunnybrook-positive-192
python prepare_sunnybrook.py --archives data/sunnybrook-sourceforge --include-ed-myocardium --out-root data/sunnybrook-positive-myo-ed-192
python prepare_york_cine.py --download --raw data/york-source --cache data/york-audit-192
for fold in 0 1; do
  common=(--data data/acdc-192 --registration registration-image-budget.json --original "runs/sparse-image-cloud/fold-$fold" --official-original "runs/sparse-official-cloud/image-$fold" --fold "$fold")
  weights="runs/sunnybrook-transfer/fold-$fold"
  python audit_external_transfer.py prepare "${common[@]}" --output "$weights"
  python audit_external_transfer.py run "${common[@]}" --weights "$weights" --output "$weights" --external data/sunnybrook-positive-192 --classes 3 --positive-only --cohort-name Sunnybrook --source-url https://www.cardiacatlas.org/sunnybrook-cardiac-data/
  python audit_external_transfer.py run "${common[@]}" --weights "$weights" --output "runs/sunnybrook-myo-ed/fold-$fold" --external data/sunnybrook-positive-myo-ed-192 --classes 2 --phases 0 --positive-only --cohort-name Sunnybrook --source-url https://www.cardiacatlas.org/sunnybrook-cardiac-data/
  python audit_external_transfer.py run "${common[@]}" --weights "$weights" --output "runs/york-lv/fold-$fold" --external data/york-audit-192 --classes 3 --positive-only --cohort-name York --source-url https://data.nvision.eecs.yorku.ca/datasets/MRI/
  python audit_external_transfer.py run "${common[@]}" --weights "$weights" --output "runs/york-myo/fold-$fold" --external data/york-audit-192 --classes 2 --positive-only --cohort-name York --source-url https://data.nvision.eecs.yorku.ca/datasets/MRI/
  python audit_external_transfer.py reference "${common[@]}" --weights "$weights" --output "runs/acdc-lv-reference/fold-$fold" --assessment data/acdc-challenge-audit-192 --classes 3
  python audit_external_transfer.py reference "${common[@]}" --weights "$weights" --output "runs/acdc-myo-ed-reference/fold-$fold" --assessment data/acdc-challenge-audit-192 --classes 2 --phases 0
  python audit_external_transfer.py reference "${common[@]}" --weights "$weights" --output "runs/acdc-myo-reference/fold-$fold" --assessment data/acdc-challenge-audit-192 --classes 2
done
for sample in sunnybrook-transfer sunnybrook-myo-ed york-lv york-myo acdc-lv-reference acdc-myo-ed-reference acdc-myo-reference; do
  python audit_external_transfer.py aggregate --omit-case-identifiers --folds "runs/$sample/fold-0" "runs/$sample/fold-1" --output "runs/$sample/summary-public.json"
done
```

论文及补充材料应引用原始数据论文，数据来源见 [DATA.md](DATA.md)。投稿按会方双盲要求处理仓库链接和作者信息，并按出版政策如实披露生成式 AI 工具的使用范围。 [DASFAA 征稿](https://dasfaa2027.github.io/CallforResearchTrack/list.htm) · [Springer 写作政策](https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation)
