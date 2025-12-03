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


def perform_box_m_test(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行Box's M检验，用于检验多个多元正态分布总体的协方差矩阵是否相等
    
    Box's M检验比Bartlett检验更为常用，特别是在SPSS等统计软件中。
    
    参数:
    - groups_data: 多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)
    - group_names: 组名称列表
    - var_names: 变量名称列表
    
    返回:
    - 包含检验结果的字典
    """
    try:
        # 输入验证
        n_groups = len(group_names)
        n_vars = len(var_names)
        
        if n_groups < 2:
            raise ValueError("至少需要两组数据进行比较")
            
        if n_vars == 0:
            raise ValueError("变量不能为空")
        
        # 将一维数组还原为嵌套列表
        # 首先计算每组的样本数
        total_length = len(groups_data)
        if total_length % (n_groups * n_vars) != 0:
            raise ValueError("数据长度与组数和变量数不匹配")
        
        n_samples_per_group = total_length // (n_groups * n_vars)
        
        # 重新组织数据为嵌套列表: groups_data_nested[group][var][sample]
        groups_data_nested = []
        idx = 0
        for g in range(n_groups):
            group_data = []
            for v in range(n_vars):
                var_data = []
                for s in range(n_samples_per_group):
                    var_data.append(groups_data[idx])
                    idx += 1
                group_data.append(var_data)
            groups_data_nested.append(group_data)
        
        # 验证数据已正确重组
        n_samples = [n_samples_per_group] * n_groups  # 每组样本数相同
        
        # 转换为numpy数组并转置为 (n_samples, n_vars)
        groups_arrays = []
        for group_data in groups_data_nested:
            group_array = np.array(group_data).T  # shape: (n_samples, n_vars)
            groups_arrays.append(group_array)
        
        # 计算每组的协方差矩阵
        cov_matrices = [np.cov(group_array, rowvar=False) for group_array in groups_arrays]
        
        # 计算各组的自由度
        dfs = [n - 1 for n in n_samples]
        
        # 计算合并的协方差矩阵
        n_total = sum(n_samples)
        pooled_cov = np.zeros((n_vars, n_vars))
        for i in range(n_groups):
            pooled_cov += dfs[i] * cov_matrices[i]
        pooled_cov /= (n_total - n_groups)
        
        # 计算Box's M统计量
        # M = (n_total - n_groups) * ln|Σ_pooled| - Σ(n_i - 1) * ln|Σ_i|
        log_det_pooled = np.log(np.linalg.det(pooled_cov))
        sum_log_dets = sum(dfs[i] * np.log(np.linalg.det(cov_matrices[i])) 
                          for i in range(n_groups))
        
        box_m_stat = (n_total - n_groups) * log_det_pooled - sum_log_dets
        
        # 计算修正因子C
        # C = 1 - (2*p^2 + 3*p - 1) / (6*(p+1)*(k-1)) * (Σ(1/(n_i-1)) - 1/(n_total-k))
        p = n_vars  # 变量数
        k = n_groups  # 组数
        
        c_factor = 1 - (2*p*p + 3*p - 1) / (6*(p+1)*(k-1)) * (
            sum(1.0/dfs[i] for i in range(k)) - 1.0/(n_total - k))
        
        # 修正后的统计量
        corrected_m_stat = box_m_stat / c_factor
        
        # 近似卡方分布的自由度
        df = p * (p + 1) // 2 * (k - 1)
        
        # 计算p值
        p_value = 1 - stats.chi2.cdf(corrected_m_stat, df)
        
        # 生成协方差矩阵热力图
        fig, axes = plt.subplots(1, n_groups + 1, figsize=(5 * (n_groups + 1), 5))
        if n_groups == 1:
            axes = [axes]
        
        # 绘制每组的协方差矩阵热力图
        for i in range(n_groups):
            im = axes[i].imshow(cov_matrices[i], cmap='coolwarm', aspect='auto')
            axes[i].set_title(f'{group_names[i]} 协方差矩阵')
            axes[i].set_xticks(range(p))
            axes[i].set_yticks(range(p))
            axes[i].set_xticklabels(var_names, rotation=45)
            axes[i].set_yticklabels(var_names)
            plt.colorbar(im, ax=axes[i])
        
        # 绘制合并的协方差矩阵
        im = axes[-1].imshow(pooled_cov, cmap='coolwarm', aspect='auto')
        axes[-1].set_title('合并协方差矩阵')
        axes[-1].set_xticks(range(p))
        axes[-1].set_yticks(range(p))
        axes[-1].set_xticklabels(var_names, rotation=45)
        axes[-1].set_yticklabels(var_names)
        plt.colorbar(im, ax=axes[-1])
        
        plt.tight_layout()
        
        # 保存协方差矩阵图
        plot_filename = f"box_m_covariance_matrices_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_groups": n_groups,
            "number_of_variables": n_vars,
            "group_names": group_names,
            "variable_names": var_names,
            "sample_sizes": {group_names[i]: n_samples[i] for i in range(n_groups)},
            "test_statistic": {
                "box_m_stat": float(box_m_stat),
                "corrected_box_m_stat": float(corrected_m_stat)
            },
            "degrees_of_freedom": df,
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "covariance_plot_url": plot_url,
            "interpretation": _get_interpretation(p_value)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"Box's M检验计算失败: {str(e)}") from e


def _get_interpretation(p_value: float) -> str:
    """
    根据p值提供结果解释
    """
    if p_value < 0.05:
        return "各组协方差矩阵存在显著差异，违反了协方差矩阵齐性的假设。"
    else:
        return "各组协方差矩阵无显著差异，满足协方差矩阵齐性的假设。"