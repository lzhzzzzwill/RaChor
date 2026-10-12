from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


# 1. 核心需求：Excel 列名字母转数字索引的函数
def excel_col2num(col_str):
    """
    将 Excel 列字母（如 'B', 'BZF'）转换为 0-based 索引。
    例如：'A' -> 0, 'B' -> 1, 'Z' -> 25, 'AA' -> 26
    """
    num = 0
    for c in col_str:
        num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1  # 减 1 是为了适配 pandas 的 0 索引系统


# 2. 映射关系字典：根据数字查找对应的塑料类型
# 注意：由于图二有些模糊，我根据每10个一类的规律提取了部分。
# 请对照你的原图二，将这个字典补全或修改准确！
plastic_mapping = {
    range(1, 11): "PP",
    range(11, 21): "POM",
    range(21, 31): "PE",
    range(31, 41): "PVC",
    range(41, 51): "SMMA",
    range(51, 61): "PAR",
    range(61, 71): "PPO",  # 需核对
    range(71, 81): "EPS",  # 需核对
    range(81, 91): "PTFE",  # 需核对
    range(91, 101): "ABS",  # 需核对
    range(101, 111): "TPU",  # 需核对
    range(111, 121): "SD",
    range(121, 131): "LDPE",
    range(131, 141): "POE",  # 需核对
    range(141, 151): "SMAC",  # 需核对
    range(151, 161): "PC",  # 需核对
    range(161, 171): "PA6",  # 需核对
    range(171, 181): "PBT",
    range(181, 191): "PPA",
    range(191, 201): "MABS",
}


def get_plastic_type(sample_id):
    """从样本名（如 Y11）提取数字并返回对应的塑料类型"""
    match = re.search(r'\d+', str(sample_id))
    if not match:
        return "Unknown"

    num = int(match.group())
    for num_range, p_type in plastic_mapping.items():
        if num in num_range:
            return p_type
    return "Unknown"


# 3. 主程序：读取数据并绘图
def main():
    file_path = Path(__file__).resolve().parent / "original.xlsx"
    base_out_dir = "Spectra_Plots"
    os.makedirs(base_out_dir, exist_ok=True)

    print(f"正在读取数据: {file_path} ...")
    df = pd.read_excel(file_path)

    start_idx = excel_col2num('B')
    end_idx = excel_col2num('BZF')
    wavelengths = df.columns[start_idx: end_idx + 1]

    # 【核心修改点】：引入新的连续编号计数器，从 1 开始
    new_sample_id = 1

    for index, row in df.iterrows():
        orig_sample_id = str(row.iloc[0]).strip()  # 获取原始 ID (如 Y11)

        # 过滤掉不是以 Y 开头的无效行
        if not orig_sample_id.startswith('Y'):
            continue

        # 根据“原始 ID”获取塑料类别
        plastic_type = get_plastic_type(orig_sample_id)

        # 创建类别文件夹
        type_dir = os.path.join(base_out_dir, plastic_type)
        os.makedirs(type_dir, exist_ok=True)

        # 提取光谱数据
        spectra_data = row.iloc[start_idx: end_idx + 1].values

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, spectra_data, color='blue', linewidth=1)

        # 标题中使用新的样本编号，方便你对应论文
        plt.title(f"Sample {new_sample_id}")
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.grid(True, linestyle='--', alpha=0.6)

        # 【核心修改点】：保存的文件名使用 new_sample_id，如 1.svg, 2.svg
        save_path = os.path.join(type_dir, f"{new_sample_id}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"已生成: {new_sample_id}.svg (来源于原始数据 {orig_sample_id})，存入 {plastic_type} 文件夹")

        # 成功画完一张图后，新编号加 1
        new_sample_id += 1

    print(f"\n全部处理完成！所有图片已保存在 '{base_out_dir}' 目录中。")


if __name__ == "__main__":
    main()