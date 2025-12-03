import os
import uuid
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pydantic import Field

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_categorical_independence_test(
    observed_frequencies: List[int] = Field(..., description="观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)"),
    row_labels: List[str] = Field(..., description="行标签列表"),
    col_labels: List[str] = Field(..., description="列标签列表"),
) -> dict:
    """
    执行卡方独立性检验，用于检验两个分类变量是否相互独立
    
    参数:
    - observed_frequencies: 观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)
    - row_labels: 行标签列表
    - col_labels: 列标签列表
    
    返回:
    - 包含检验结果的字典
    """
    try:
        # 输入验证
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        
        if n_rows == 0 or n_cols == 0:
            raise ValueError("观测频数表不能为空")
        
        # 检查数据长度是否匹配
        expected_length = n_rows * n_cols
        if len(observed_frequencies) != expected_length:
            raise ValueError(f"观测频数数据长度({len(observed_frequencies)})与行列标签定义的长度({expected_length})不匹配")
        
        # 将一维数组还原为嵌套列表
        observed_frequencies_nested = []
        idx = 0
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                row.append(observed_frequencies[idx])
                idx += 1
            observed_frequencies_nested.append(row)
        
        # 转换为numpy数组
        observed = np.array(observed_frequencies_nested)
        
        # 计算行总计、列总计和总总计
        row_totals = np.sum(observed, axis=1)
        col_totals = np.sum(observed, axis=0)
        grand_total = np.sum(observed)
        
        # 计算期望频数
        # E_ij = (行总计_i * 列总计_j) / 总总计
        expected = np.outer(row_totals, col_totals) / grand_total
        
        # 执行卡方检验
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
        
        # 计算Cramer's V效应量（适用于大的列联表）
        # Cramer's V = sqrt(chi2 / (n * min(r-1, c-1)))
        n = grand_total
        cramers_v = np.sqrt(chi2_stat / (n * min(n_rows - 1, n_cols - 1)))
        
        # 计算Phi系数（仅适用于2x2列联表）
        phi_coefficient = None
        if n_rows == 2 and n_cols == 2:
            phi_coefficient = np.sqrt(chi2_stat / n)
        
        # 生成频数表格热力图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制观测频数热力图
        im1 = axes[0].imshow(observed, cmap='Blues', aspect='auto')
        axes[0].set_title('观测频数')
        axes[0].set_xticks(range(n_cols))
        axes[0].set_yticks(range(n_rows))
        axes[0].set_xticklabels(col_labels, rotation=45)
        axes[0].set_yticklabels(row_labels)
        
        # 在每个单元格中添加数值
        for i in range(n_rows):
            for j in range(n_cols):
                axes[0].text(j, i, str(observed[i, j]), 
                            ha="center", va="center", color="black")
        
        plt.colorbar(im1, ax=axes[0])
        
        # 绘制期望频数热力图
        im2 = axes[1].imshow(expected, cmap='Reds', aspect='auto')
        axes[1].set_title('期望频数')
        axes[1].set_xticks(range(n_cols))
        axes[1].set_yticks(range(n_rows))
        axes[1].set_xticklabels(col_labels, rotation=45)
        axes[1].set_yticklabels(row_labels)
        
        # 在每个单元格中添加数值
        for i in range(n_rows):
            for j in range(n_cols):
                axes[1].text(j, i, f'{expected[i, j]:.2f}', 
                            ha="center", va="center", color="black")
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        # 保存频数表格图
        plot_filename = f"independence_test_heatmap_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_rows": n_rows,
            "number_of_columns": n_cols,
            "row_labels": row_labels,
            "column_labels": col_labels,
            "observed_frequencies": observed.tolist(),
            "expected_frequencies": expected.tolist(),
            "row_totals": row_totals.tolist(),
            "column_totals": col_totals.tolist(),
            "grand_total": int(grand_total),
            "test_statistic": {
                "chi_square": float(chi2_stat),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value)
            },
            "effect_size": {
                "cramers_v": float(cramers_v),
                "phi_coefficient": float(phi_coefficient) if phi_coefficient is not None else None
            },
            "significant": p_value < 0.05,
            "heatmap_url": plot_url,
            "interpretation": _get_interpretation(p_value, cramers_v)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"独立性检验计算失败: {str(e)}") from e


def _get_interpretation(p_value: float, cramers_v: float) -> str:
    """
    根据p值和效应量提供结果解释
    """
    interpretation = ""
    
    if p_value < 0.05:
        interpretation += "两个分类变量之间存在显著关联。"
    else:
        interpretation += "两个分类变量之间无显著关联，可以认为是相互独立的。"
    
    # 解释Cramer's V效应量
    if cramers_v < 0.1:
        interpretation += "关联程度很弱。"
    elif cramers_v < 0.3:
        interpretation += "关联程度较弱。"
    elif cramers_v < 0.5:
        interpretation += "关联程度中等。"
    else:
        interpretation += "关联程度较强。"
    
    return interpretation