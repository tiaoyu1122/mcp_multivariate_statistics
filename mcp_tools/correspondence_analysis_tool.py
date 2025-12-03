import os
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pydantic import Field
from sklearn.preprocessing import StandardScaler

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_correspondence_analysis(
    observed_frequencies: List[int] = Field(..., description="观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)"),
    row_labels: List[str] = Field(..., description="行标签列表"),
    col_labels: List[str] = Field(..., description="列标签列表"),
) -> dict:
    """
    执行对应分析 (Correspondence Analysis)
    
    参数:
    - observed_frequencies: 观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)
    - row_labels: 行标签列表
    - col_labels: 列标签列表
    
    返回:
    - 包含对应分析结果的字典
    """
    try:
        # 输入验证
        if not observed_frequencies:
            raise ValueError("观测频数表不能为空")
        
        n_rows = len(row_labels)
        if n_rows == 0:
            raise ValueError("行数不能为0")
            
        n_cols = len(col_labels)
        if n_cols == 0:
            raise ValueError("列数不能为0")
            
        if len(observed_frequencies) != n_rows * n_cols:
            raise ValueError("观测频数表长度与行列标签数量不匹配")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_frequencies = []
        for i in range(n_rows):
            row_data = observed_frequencies[i * n_cols:(i + 1) * n_cols]
            reshaped_frequencies.append(row_data)
        
        # 转换为numpy数组
        frequency_matrix = np.array(reshaped_frequencies, dtype=float)  # shape: (n_rows, n_cols)
        
        # 检查是否有负值
        if np.any(frequency_matrix < 0):
            raise ValueError("频数表中不能包含负值")
        
        # 计算总频数
        total_frequency = np.sum(frequency_matrix)
        
        # 计算行和列的边际频率
        row_marginals = np.sum(frequency_matrix, axis=1)  # 每行的总和
        col_marginals = np.sum(frequency_matrix, axis=0)  # 每列的总和
        
        # 检查边际频率是否为零
        if np.any(row_marginals == 0):
            raise ValueError("某些行的边际频率为零")
            
        if np.any(col_marginals == 0):
            raise ValueError("某些列的边际频率为零")
        
        # 计算概率矩阵 (除以总频数)
        probability_matrix = frequency_matrix / total_frequency
        
        # 计算行和列的边际概率
        row_probs = row_marginals / total_frequency
        col_probs = col_marginals / total_frequency
        
        # 计算期望频数矩阵
        expected_matrix = np.outer(row_marginals, col_marginals) / total_frequency
        
        # 计算标准化残差矩阵
        residuals_matrix = (frequency_matrix - expected_matrix) / np.sqrt(expected_matrix)
        
        # 计算惯量 (Inertia)
        inertia_matrix = ((frequency_matrix - expected_matrix)**2) / expected_matrix
        total_inertia = np.sum(inertia_matrix) / total_frequency
        
        # 进行奇异值分解 (SVD)
        # 构造矩阵: D_r^(-1/2) * (P - rc^T) * D_c^(-1/2)
        # 其中 D_r 和 D_c 是行和列边际概率的对角矩阵
        D_r_sqrt_inv = np.diag(1.0 / np.sqrt(row_probs))
        D_c_sqrt_inv = np.diag(1.0 / np.sqrt(col_probs))
        
        # 中心矩阵
        center_matrix = probability_matrix - np.outer(row_probs, col_probs)
        
        # SVD分解矩阵
        svd_matrix = D_r_sqrt_inv @ center_matrix @ D_c_sqrt_inv
        
        # 执行奇异值分解
        U, singular_values, Vt = np.linalg.svd(svd_matrix, full_matrices=False)
        
        # 计算惯量比例
        inertia_ratios = (singular_values**2) / total_inertia if total_inertia > 0 else np.zeros_like(singular_values)
        
        # 确定维度数 (最多为 min(n_rows, n_cols) - 1)
        n_dimensions = min(n_rows, n_cols) - 1
        
        # 计算行和列的坐标
        # 行坐标: row_coords = D_r^(-1/2) * U * S
        row_coordinates = D_r_sqrt_inv @ U[:, :n_dimensions] @ np.diag(singular_values[:n_dimensions])
        
        # 列坐标: col_coords = D_c^(-1/2) * V^T * S
        col_coordinates = D_c_sqrt_inv @ Vt[:n_dimensions, :].T @ np.diag(singular_values[:n_dimensions])
        
        # 计算质量 (Quality of representation)
        row_quality = np.zeros((n_rows, n_dimensions))
        col_quality = np.zeros((n_cols, n_dimensions))
        
        for i in range(n_rows):
            row_norm_sq = np.sum(row_coordinates[i, :]**2)
            if row_norm_sq > 0:
                row_quality[i, :] = row_coordinates[i, :]**2 / row_norm_sq
        
        for j in range(n_cols):
            col_norm_sq = np.sum(col_coordinates[j, :]**2)
            if col_norm_sq > 0:
                col_quality[j, :] = col_coordinates[j, :]**2 / col_norm_sq
        
        # 生成对应分析图 (Biplot)
        plt.figure(figsize=(12, 10))
        
        # 绘制行点
        plt.scatter(row_coordinates[:, 0], row_coordinates[:, 1], 
                   marker='o', s=100, c='blue', alpha=0.7, label='行点')
        
        # 绘制列点
        plt.scatter(col_coordinates[:, 0], col_coordinates[:, 1], 
                   marker='s', s=100, c='red', alpha=0.7, label='列点')
        
        # 添加标签
        for i, label in enumerate(row_labels):
            plt.annotate(label, (row_coordinates[i, 0], row_coordinates[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, color='blue')
        
        for j, label in enumerate(col_labels):
            plt.annotate(label, (col_coordinates[j, 0], col_coordinates[j, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
        
        # 添加原点参考线
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel(f'维度1 ({inertia_ratios[0]:.2%} 惯量)')
        plt.ylabel(f'维度2 ({inertia_ratios[1]:.2%} 惯量)' if n_dimensions > 1 else '维度2 (0.00% 惯量)')
        plt.title('对应分析图 (Correspondence Analysis Biplot)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存对应分析图
        ca_plot_filename = f"correspondence_analysis_plot_{uuid.uuid4().hex}.png"
        ca_plot_filepath = os.path.join(OUTPUT_DIR, ca_plot_filename)
        plt.savefig(ca_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        ca_plot_url = f"{PUBLIC_FILE_BASE_URL}/{ca_plot_filename}"
        
        # 生成惯量贡献图
        plt.figure(figsize=(10, 6))
        dimensions = range(1, min(10, len(inertia_ratios) + 1))  # 最多显示前10个维度
        plt.bar(dimensions, inertia_ratios[:len(dimensions)], color='skyblue')
        plt.xlabel('维度')
        plt.ylabel('惯量比例')
        plt.title('各维度惯量贡献')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存惯量贡献图
        inertia_plot_filename = f"inertia_contribution_plot_{uuid.uuid4().hex}.png"
        inertia_plot_filepath = os.path.join(OUTPUT_DIR, inertia_plot_filename)
        plt.savefig(inertia_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        inertia_plot_url = f"{PUBLIC_FILE_BASE_URL}/{inertia_plot_filename}"
        
        # 组织结果
        result = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "total_frequency": float(total_frequency),
            "total_inertia": float(total_inertia),
            "n_dimensions": n_dimensions,
            "singular_values": singular_values[:n_dimensions].tolist(),
            "inertia_ratios": inertia_ratios[:n_dimensions].tolist(),
            "row_coordinates": row_coordinates.tolist(),
            "col_coordinates": col_coordinates.tolist(),
            "row_quality": row_quality.tolist(),
            "col_quality": col_quality.tolist(),
            "frequency_matrix": frequency_matrix.tolist(),
            "expected_matrix": expected_matrix.tolist(),
            "residuals_matrix": residuals_matrix.tolist(),
            "plots": {
                "correspondence_analysis_plot_url": ca_plot_url,
                "inertia_contribution_plot_url": inertia_plot_url
            },
            "interpretation": _get_interpretation(inertia_ratios, row_coordinates, col_coordinates, row_labels, col_labels)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"对应分析失败: {str(e)}") from e


def _get_interpretation(inertia_ratios: np.ndarray, row_coordinates: np.ndarray, col_coordinates: np.ndarray,
                       row_labels: List[str], col_labels: List[str]) -> str:
    """
    根据对应分析结果提供解释
    """
    n_dimensions = len(inertia_ratios)
    total_explained = np.sum(inertia_ratios[:min(2, n_dimensions)])  # 前两个维度解释的惯量
    
    interpretation = f"对应分析完成，共提取{n_dimensions}个维度。"
    interpretation += f"前两个维度分别解释了{inertia_ratios[0]:.2%}和{inertia_ratios[1]:.2%}的惯量。"
    interpretation += f"前两个维度累计解释了{total_explained:.2%}的总惯量。"
    
    # 根据解释惯量的比例给出建议
    if total_explained >= 0.7:
        interpretation += "前两个维度能够较好地展示行和列之间的关系结构。"
    elif total_explained >= 0.5:
        interpretation += "前两个维度基本能够展示行和列之间的关系结构。"
    else:
        interpretation += "前两个维度对关系结构的解释程度较低，建议查看更高维度或检查数据质量。"
    
    # 解释行点和列点的关系
    interpretation += "\n\n关系解释：\n"
    
    # 找到距离原点较远的点（具有较大影响力的点）
    row_distances = np.sqrt(np.sum(row_coordinates**2, axis=1))
    col_distances = np.sqrt(np.sum(col_coordinates**2, axis=1))
    
    # 找到前25%的点作为重要点
    row_threshold = np.percentile(row_distances, 75)
    col_threshold = np.percentile(col_distances, 75)
    
    interpretation += "影响力较大的行点："
    important_rows = []
    for i, distance in enumerate(row_distances):
        if distance >= row_threshold:
            important_rows.append((row_labels[i], distance))
    
    # 按距离排序
    important_rows.sort(key=lambda x: x[1], reverse=True)
    for label, distance in important_rows[:3]:  # 显示前3个
        interpretation += f" {label}"
    
    interpretation += "\n影响力较大的列点："
    important_cols = []
    for j, distance in enumerate(col_distances):
        if distance >= col_threshold:
            important_cols.append((col_labels[j], distance))
    
    # 按距离排序
    important_cols.sort(key=lambda x: x[1], reverse=True)
    for label, distance in important_cols[:3]:  # 显示前3个
        interpretation += f" {label}"
    
    interpretation += "\n\n注：越远离原点的点在对应分析中影响力越大。"
    interpretation += "在同一区域的行点和列点表示它们之间存在较强的关系。"
    
    return interpretation