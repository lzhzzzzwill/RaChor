# 原版贝叶斯 XGBoost 的 Sn/Bi 改写

从项目根目录运行：

```bash
python -m pip install -r requirements.txt
python other_models/xgb_bayesian_oof.py
```

默认输入 `data/Raman_spectroscopy_data.xlsx`（相对脚本目录为 `../data/Raman_spectroscopy_data.xlsx`），也支持原始 CSV。所有相对文件参数以脚本目录为基准：

```bash
python other_models/xgb_bayesian_oof.py --data ../data/my_raw_spectra.csv --output-dir results/xgb_run2
```

只预测 `118Sn (KED)`、`209Bi (KED)`，采用双标签完整行并保持样本顺序。只选择数字波数列，范围包含 400–1800 cm⁻¹ 两端，按波数排序；Type 仅用于结果标注，不作为输入。不选波段、不追加描述符。

处理顺序是全谱截取 → SNV → SG 一阶导数（窗口 9，阶数 2）→ StandardScaler → XGBRegressor。SNV、SG 对每条光谱独立执行；StandardScaler 位于 sklearn Pipeline 中，在每个内部或外部训练分区单独拟合，延续原版代码的 scaler 用法。它与当前 RaChor 预处理脚本在完整数据集上标准化的口径有所不同。

保留原代码的贝叶斯优化参数范围和负平均 MSE 目标，默认使用 20 个初始点＋40 次优化。每个外层训练集单独优化：外层五折用于 OOF 预测，内层五折用于选参。外层 seed=42，内层 seed=42+外层折号。完整运行约需 3000 次内部拟合及 10 次外层拟合。

输出在 `other_models/results/xgb_bayesian_oof/`：

- `Sn_oof_predictions.csv`、`Bi_oof_predictions.csv`：样本编号、原表行号、折编号、实测值、OOF 预测值。
- `oof_predictions.csv`：两种金属的合并预测表，每条样本一行。
- `global_summary.csv`：训练折均值和全体 OOF 统一计算的 R²、RMSE、RPD，另外保留原版的 MAE。
- `fold_performance.csv`：逐折训练/测试指标。
- `per_fold_predictions.csv`：保留原版 Train/Test 逐折记录格式；训练样本会在不同外层折重复出现，不能将该表的所有行混在一起当作 OOF。
- `optimization_results.json`：每个目标、每个外层折的优化历史及最优参数。
- `xgb_bayesian_results.xlsx`：以上主要预测和指标的 Excel 汇总。
- `run_metadata.json`：输入哈希及处理参数，全部完成后才标记 `complete`。

RPD 与当前代码相同，为 `std(y_true, ddof=1) / RMSE`。OOF 汇总指标通过拼接每条样本唯一的外层留出预测计算，不是各折测试指标的均值。

可添加 `--save-fold-models` 保存每个外层模型及波数信息；这些是外层折模型，不是全数据拟合的部署模型。加载后需先按记录的波数截取、SNV、SG，再调用其中的 pipeline。

如果输出目录已有文件，脚本会自动使用带时间戳后缀的新目录，保留旧结果。代码未自动运行训练，也不会改动原有四模型对比脚本或 GitHub 仓库。
