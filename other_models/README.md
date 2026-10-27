# 其他模型：独立基线与图 3 风格绘图

```text
other_models/
  rf_bayesian_oof.py
  cnn_bayesian_oof.py
  catboost_bayesian_oof.py
  xgb_bayesian_oof.py
  baseline_workflow.py
  cnn_baseline.py
  plot_baseline_comparison.py
  results/      # 各模型结果与历史数据
  plot_data/    # 绘图时汇总生成的源数据
  figures/      # 绘图时生成的图及 QA 报告
```

## 分别训练

从项目根目录运行，或进入 `other_models` 后直接运行对应脚本文件名。

```bash
# 一次安装全部训练与绘图依赖
python -m pip install -r requirements.txt

# RF
python other_models/rf_bayesian_oof.py

# CatBoost
python other_models/catboost_bayesian_oof.py

# CNN
python other_models/cnn_bayesian_oof.py
```

四种模型的入口及其共用代码、CNN 定义和绘图脚本均在本目录。原始输入表仍共用上一级 `data/`，不重复复制。三个文件均可独立启动；公共模块 `baseline_workflow.py` 仅复用数据处理、优化和结果导出，不会自动启动其他模型。XGBoost 的训练算法和参数保持不变，仅调整搬移后的导入和路径；其已有结果迁移到 `results/xgb_bayesian_oof*`，无需为本次改动重跑。

## 与 XGBoost 一致的协议

默认读取主模型共享的原始 `../data/Raman_spectroscopy_data.xlsx`；可用 `--data` 指定原始 Excel/CSV，相对路径均以训练脚本目录为准。

1. Sn/Bi 双标签完整行，样本顺序不变；400–1800 cm⁻¹ 全波段按波数排序。
2. SNV → SG 一阶导数（窗口 9、多项式阶数 2）→ StandardScaler → 各自模型。
3. 不做选波段、不加描述符、不堆叠。SNV/SG 逐样本计算，StandardScaler 在每个训练分区独立拟合。
4. 外层五折用于 OOF（seed=42），内层五折用于贝叶斯优化（seed=42+外层折号）。默认 20 个初始点＋40 次优化，目标为内层平均 MSE 最小。
5. 指标统一为 R²、RMSE、RPD，另保存 MAE；RPD 使用样本标准差 `ddof=1`。OOF 汇总通过全部留出预测统一计算，不是各折指标平均值。

RF 优化树数、深度、分裂/叶节点样本数和特征比例；CatBoost 优化迭代数、深度、学习率、L2 和随机强度；CNN 优化卷积通道数、学习率、权重衰减和 batch size。整数参数在拟合时取整。CNN 学习率和权重衰减在 log10 坐标搜索。

CNN 默认 CPU 训练、每次拟合 150 个 epoch。`--epochs 200` 可修改训练轮数，与 `--init-points 20 --n-iter 40` 的贝叶斯搜索次数是不同概念。完整 CNN 实验约需 3000 次内部模型拟合，可能较慢；不使用外层测试标签提前停止。

## 独立结果文件夹

```text
other_models/results/
  rf/
  catboost/
  cnn/
```

目录已有内容时自动新增带时间戳的同级目录，不覆盖旧结果。每种模型保存：

- `Sn_oof_predictions.csv`、`Bi_oof_predictions.csv`：逐样本实测值和 OOF 预测。
- `oof_predictions.csv`：两种金属合并的一行一条样本的预测表。
- `fold_performance.csv`：逐折 Train/Test 的 R²、RMSE、RPD、MAE。
- `global_summary.csv`：训练折均值与 pooled OOF 指标。
- `per_fold_predictions.csv`：全部逐折训练与测试预测；训练部分重复出现，不可整表混作 OOF。
- `optimization_results.json`：完整优化历史、最优参数。
- `baseline_results.xlsx`：上述主要指标与预测的 Excel 汇总。
- `run_metadata.json`：数据哈希、波数、参数和运行状态。
- `plot_data/`：直接可用于作图的 OOF、逐折指标、汇总 CSV 副本。

`--save-fold-models` 可额外保存外层模型（不是全数据部署模型）。未完成运行的 metadata 不会标为 `complete`。

## 绘图

所有训练完成后，在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python other_models/plot_baseline_comparison.py
```

默认读取四种模型最新的已完成运行，并加入原有 RaChor Final Fusion 结果；最新按 metadata 修改时间确定，不按性能挑选。也可手动指定某次运行：

```bash
python other_models/plot_baseline_comparison.py --rf-dir other_models/results/rf --cnn-dir other_models/results/cnn --catboost-dir other_models/results/catboost --xgb-dir other_models/results/xgb_bayesian_oof
```

仅绘制部分已完成模型可用 `--models RF CatBoost`；不加入 RaChor 可用 `--no-rachor`。绘图脚本的相对路径以 RaChor 目录为准，启动位置不限。缺少结果时会报错，不生成虚构占位图。

沿用图 3 的浅绿箱体、蓝色散点、五种折符号、字体、等值虚线和拟合实线，输出两组 PDF/SVG：

- `other_models/figures/baseline_fold_performance.*`：Sn/Bi 的训练与留出折 R²（四面板）。显示全部五折点，纵轴根据实际数据扩展，不截掉低分或负分。
- `other_models/figures/baseline_oof_predictions.*`：每种模型对两种金属的 pooled OOF 实测–预测图，标注 R²、RMSE、RPD。每种金属的模型使用相同坐标范围，保留全部样本。

先核对各模型的样本、真实标签、折号，以及逐折指标是否与 OOF 数据一致。源数据另存到 `other_models/plot_data/`，包含 OOF 长表、折指标、统一计算的指标及来源清单。RPD 均使用 `ddof=1`，不沿用旧绘图函数里的总体标准差算法。

绘图导出前检查面板对齐，导出后检查 PDF 字体与碰撞，报告在结果目录的 `qa/`。自动检查不能代替最终人工视觉审阅。模型结果尚未生成时只能交付代码，不能验证最终图形。

旧合集入口 `compare_models.py`、旧说明和旧合集依赖文件已移除；此前训练产生的结果保留。当前 RaChor 的全数据标准化口径与这些独立基线的折内标准化不同，来源清单会记录这一点。
