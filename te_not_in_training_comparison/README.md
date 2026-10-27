# 完整 MoE 模型测试

输入：../data/te_not_in_training.xlsx
模型：../../APP/artifacts_complete/spectroscopy_pipeline.joblib
预测：../data/te_not_in_training_predictions.xlsx

全部 4 行均完成预测。预测自动排除实测值列，使用训练时的波数对齐、400–1800 cm⁻¹ 裁剪、SNV、Savitzky–Golay 一阶导数（窗口 9，阶数 2）、保存的标准化器及完整 MoE。

all_samples_comparison.csv 保留全部 4 行。comparison_data.csv、metrics.json 和两张对比图仅使用实测值完整的 Y8、Y118（n=2）；Y167、Y171 的 Sn、Bi 实测值均为空，未参与误差计算。

Y8 与 Y118 的光谱完全相同，但实测标签不同；因此相同的模型预测无法同时匹配两组标签。必须先核对原始光谱与实测标签的对应关系，且当前数据只有 3 条不同光谱，不宜据此判断泛化性能。

运行时出现 sklearn 的矩阵计算 RuntimeWarning；本次最终 4 行预测均为有限数值。警告的具体原因尚未由本次测试确定。
