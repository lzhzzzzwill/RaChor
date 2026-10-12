from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# 1. 准备数据 (如果没有文件，生成模拟数据)
# ==========================================
try:
    # 尝试读取你的数据路径
    data_path = Path(__file__).resolve().parent / "filter_snv_sg_scaled_del_4_model.csv"
    df = pd.read_csv(data_path)


    # 简单的列索引转换函数
    def col_letter_to_index(col_letter):
        col_letter = col_letter.strip().upper()
        idx = 0
        for ch in col_letter:
            idx = idx * 26 + (ord(ch) - ord('A') + 1)
        return idx - 1


    x_start = col_letter_to_index("B")
    x_end = col_letter_to_index("AAF")
    X = df.iloc[:, x_start:x_end + 1].values
    # 假设目标列是这两列，请根据实际情况修改
    target_metals = ["118Sn (KED)", "209Bi (KED)"]
    y = df[target_metals].values
    print(f"成功加载数据: {X.shape}")

except Exception as e:
    print(f"未找到数据文件或读取出错 ({e})，使用模拟数据演示...")
    np.random.seed(42)
    n_samples = 150
    n_features = 200
    X = np.random.randn(n_samples, n_features)
    # 模拟两个金属，与X线性相关但带噪声
    coef = np.random.randn(n_features, 2)
    y = X @ coef + np.random.normal(0, 0.5, size=(n_samples, 2))
    target_metals = ["Metal-1", "Metal-2"]

# ==========================================
# 2. 嵌套交叉验证 (Nested CV) - 只用 PLS
# ==========================================
# 定义外层 5 折
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# 定义内层 5 折 (在 GridSearchCV 中使用)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储最终的 OOF 预测结果
y_oof = np.zeros_like(y)

print("开始嵌套交叉验证 (Outer 5-Fold, Inner 5-Fold)...")

# 外层循环
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 定义 PLS 管道 (标准化 + PLS)
    pls_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSRegression())
    ])

    # 定义超参数网格 (调整 n_components)
    # 注意：n_components 不能超过特征数或样本数
    max_comp = min(X_train.shape[0], X_train.shape[1], 30)
    param_grid = {
        'pls__n_components': [2, 5, 8, 10, 12, 15, 20]  # 在这里调整参数范围
    }
    # 过滤掉超过限制的组件数
    param_grid['pls__n_components'] = [c for c in param_grid['pls__n_components'] if c <= max_comp]

    # 内层循环：使用 GridSearchCV 自动调参
    grid = GridSearchCV(
        estimator=pls_pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='neg_mean_squared_error',  # 或 'r2'
        n_jobs=-1,
        verbose=0
    )

    # 在外层训练集上进行内层搜索和训练
    grid.fit(X_train, y_train)

    # 使用最佳参数的模型预测外层测试集 (OOF Predict)
    y_pred_fold = grid.predict(X_test)
    y_oof[test_idx] = y_pred_fold

    print(f"Fold {fold_idx + 1}/5 | Best Components: {grid.best_params_['pls__n_components']}")

# ==========================================
# 3. 绘图 (矢量图风格)
# ==========================================
plt.style.use('default')
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 7), dpi=300)

# 定义样式颜色
scatter_face = '#a6cee3'  # 浅蓝填充
scatter_edge = '#1f78b4'  # 深蓝边框
line_color = '#555555'  # 虚线颜色

for i, metal_name in enumerate(target_metals):
    ax = axes[i]
    y_true_col = y[:, i]
    y_pred_col = y_oof[:, i]

    # 计算范围
    min_val = min(y_true_col.min(), y_pred_col.min())
    max_val = max(y_true_col.max(), y_pred_col.max())
    margin = (max_val - min_val) * 0.1
    lims = [min_val - margin, max_val + margin]

    # 1. 画对角虚线
    ax.plot(lims, lims, ls='--', c=line_color, lw=1.5, zorder=1)

    # 2. 画散点 (OOF Prediction)
    ax.scatter(y_true_col, y_pred_col,
               c=scatter_face,
               edgecolors=scatter_edge,
               s=40,
               alpha=0.7,
               linewidth=1.0,
               zorder=2)

    # 3. 坐标轴美化 (仿照参考图)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 隐藏上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 加粗左侧和下方边框
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # 手动画坐标轴末端的三角形箭头
    # Y轴箭头
    ax.plot(lims[0], lims[1], marker='^', color='black', markersize=6,
            transform=ax.transData, clip_on=False, zorder=10)
    # X轴箭头
    ax.plot(lims[1], lims[0], marker='>', color='black', markersize=6,
            transform=ax.transData, clip_on=False, zorder=10)

    # 隐藏刻度数字 (参考图风格)
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置标签
    display_name = "Metal-1" if i == 0 else "Metal-2"
    ax.set_xlabel(display_name, fontsize=14, labelpad=5, color='black')

    # 可选：计算并打印指标看看效果
    r2 = r2_score(y_true_col, y_pred_col)
    print(f"Metrics {display_name}: R2 = {r2:.3f}")

plt.tight_layout()

# 保存为 SVG 矢量图
output_file = "pls_nested_cv_oof.svg"
plt.savefig(output_file, format='svg', transparent=True, bbox_inches='tight')
print(f"图片已保存至: {output_file}")
plt.show()