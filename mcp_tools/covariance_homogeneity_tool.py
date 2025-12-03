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


def perform_covariance_homogeneity_test(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行协方差矩阵齐性检验(Bartlett检验)，用于检验多个多元正态分布总体的协方差矩阵是否相等
    
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
            raise ValueError("变量数量不能为0")
        
        # 检查数据长度是否是组数和变量数的倍数
        if len(groups_data) % (n_groups * n_vars) != 0:
            raise ValueError("数据长度必须是组数和变量数乘积的整数倍")
        
        # 计算每组的样本数
        total_samples = len(groups_data) // (n_groups * n_vars)
        n_samples = [total_samples] * n_groups  # 假设每组样本数相同
        
        # 将一维数组还原为多层嵌套列表
        groups_data_nested = []
        for g_idx in range(n_groups):
            group_data = []
            for v_idx in range(n_vars):
                # 计算当前组当前变量在数组中的起始位置
                start_idx = (g_idx * n_vars + v_idx) * total_samples
                end_idx = start_idx + total_samples
                var_data = groups_data[start_idx:end_idx]
                group_data.append(var_data)
            groups_data_nested.append(group_data)
        
        # 检查每组内所有变量的数据长度是否一致
        for g_idx, group_data in enumerate(groups_data_nested):
            if len(group_data) != n_vars:
                raise ValueError(f"第{g_idx+1}组数据的变量数量与其它组不一致")
                
            for v_idx, var_data in enumerate(group_data):
                if len(var_data) != n_samples[g_idx]:
                    raise ValueError(f"第{g_idx+1}组中变量 {var_names[v_idx]} 的数据长度不一致")
        
        # 转换为numpy数组并转置为 (n_samples, n_vars)
        groups_arrays = []
        for group_data in groups_data_nested:
            group_array = np.array(group_data).T  # shape: (n_samples, n_vars)
            groups_arrays.append(group_array)
        
        # 计算每组的协方差矩阵
        cov_matrices = [np.cov(group_array, rowvar=False) for group_array in groups_arrays]
        
        # Bartlett协方差矩阵齐性检验
        # 实现Bartlett检验的统计量计算
        n_total = sum(n_samples)
        p = n_vars  # 变量数
        
        # 计算合并的协方差矩阵
        pooled_cov = np.zeros((p, p))
        for i in range(n_groups):
            pooled_cov += (n_samples[i] - 1) * cov_matrices[i]
        pooled_cov /= (n_total - n_groups)
        
        # 计算Bartlett检验统计量
        # 先计算对数行列式
        log_det_pooled = np.log(np.linalg.det(pooled_cov))
        sum_log_dets = sum((n_samples[i] - 1) * np.log(np.linalg.det(cov_matrices[i])) 
                          for i in range(n_groups))
        
        # Bartlett检验统计量
        # C = 1 - (2*p^2 + 3*p - 1) / (6*(p+1)*(k-1)) * (sum(1/(n_i-1)) - 1/(n-k))
        c_factor = 1 - (2*p*p + 3*p - 1) / (6*(p+1)*(n_groups-1)) * (
            sum(1.0/(n_samples[i]-1) for i in range(n_groups)) - 1.0/(n_total-n_groups))
        
        # 检验统计量
        bartlett_stat = ((n_total - n_groups) * log_det_pooled - sum_log_dets) / c_factor
        
        # 检验统计量近似服从自由度为 p*(p+1)/2 * (k-1) 的卡方分布
        df = p * (p + 1) // 2 * (n_groups - 1)
        p_value = 1 - stats.chi2.cdf(bartlett_stat, df)
        
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
        plot_filename = f"covariance_matrices_{uuid.uuid4().hex}.png"
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
                "bartlett_stat": float(bartlett_stat),
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
        raise RuntimeError(f"协方差矩阵齐性检验计算失败: {str(e)}") from e


def _get_interpretation(p_value: float) -> str:
    """
    根据p值提供结果解释
    """
    if p_value < 0.05:
        return "各组协方差矩阵存在显著差异，违反了协方差矩阵齐性的假设。"
    else:
        return "各组协方差矩阵无显著差异，满足协方差矩阵齐性的假设。"