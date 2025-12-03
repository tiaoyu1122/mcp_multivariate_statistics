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

def perform_hotelling_t2_test(
    group1_data: List[float] = Field(..., description="第一组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    group2_data: List[float] = Field(..., description="第二组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行Hotelling's T²检验，用于检验两个多维正态分布总体的均值向量是否相等
    
    参数:
    - group1_data: 第一组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - group2_data: 第二组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    
    返回:
    - 包含检验结果的字典
    """
    try:
        # 输入验证
        n_vars = len(var_names)
        
        # 检查数据长度是否是变量数的倍数
        if len(group1_data) % n_vars != 0:
            raise ValueError("第一组数据长度必须是变量数量的整数倍")
        
        if len(group2_data) % n_vars != 0:
            raise ValueError("第二组数据长度必须是变量数量的整数倍")
        
        n1 = len(group1_data) // n_vars
        n2 = len(group2_data) // n_vars
        
        if n_vars == 0:
            raise ValueError("变量数量不能为0")
        
        # 将一维数组还原为嵌套列表
        group1_data_nested = []
        group2_data_nested = []
        
        for i in range(n_vars):
            var1_data = group1_data[i * n1:(i + 1) * n1]
            var2_data = group2_data[i * n2:(i + 1) * n2]
            group1_data_nested.append(var1_data)
            group2_data_nested.append(var2_data)
        
        # 检查每组内所有变量的数据长度是否一致
        for i, var_data in enumerate(group1_data_nested):
            if len(var_data) != n1:
                raise ValueError(f"第一组中变量 {var_names[i]} 的数据长度不一致")
        
        for i, var_data in enumerate(group2_data_nested):
            if len(var_data) != n2:
                raise ValueError(f"第二组中变量 {var_names[i]} 的数据长度不一致")
        
        # 转换为numpy数组
        group1_array = np.array(group1_data_nested)  # shape: (n_vars, n1)
        group2_array = np.array(group2_data_nested)  # shape: (n_vars, n2)
        
        # 转置为 (n_samples, n_vars)
        group1_array = group1_array.T  # shape: (n1, n_vars)
        group2_array = group2_array.T  # shape: (n2, n_vars)
        
        # 计算各组均值向量
        mean1 = np.mean(group1_array, axis=0)
        mean2 = np.mean(group2_array, axis=0)
        
        # 计算各组协方差矩阵
        cov1 = np.cov(group1_array, rowvar=False)
        cov2 = np.cov(group2_array, rowvar=False)
        
        # 计算合并协方差矩阵
        pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
        
        # 计算均值差向量
        diff_mean = mean1 - mean2
        
        # 计算Hotelling's T²统计量
        # T² = n1*n2/(n1+n2) * (mean1 - mean2)^T * S_pooled^(-1) * (mean1 - mean2)
        try:
            inv_pooled_cov = np.linalg.inv(pooled_cov)
            t2_stat = (n1 * n2 / (n1 + n2)) * diff_mean.T @ inv_pooled_cov @ diff_mean
        except np.linalg.LinAlgError:
            raise ValueError("协方差矩阵奇异，无法计算逆矩阵")
        
        # 转换为F统计量
        # F = (n1+n2-p-1)/(p*(n1+n2-2)) * T²
        p = n_vars  # 变量数
        f_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * t2_stat
        
        # 计算p值
        # F统计量服从F分布，自由度为(p, n1+n2-p-1)
        p_value = 1 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)
        
        # 计算效应量 (Cohen's d 等效形式)
        # 这里使用Mahalanobis距离作为效应量
        try:
            effect_size = np.sqrt(t2_stat / (n1 + n2))
        except:
            effect_size = None
        
        # 生成均值对比图
        plt.figure(figsize=(10, 6))
        
        x_pos = np.arange(len(var_names))
        plt.bar(x_pos - 0.2, mean1, 0.4, label='组1', alpha=0.8)
        plt.bar(x_pos + 0.2, mean2, 0.4, label='组2', alpha=0.8)
        
        plt.xlabel('变量')
        plt.ylabel('均值')
        plt.title('两组样本均值对比')
        plt.xticks(x_pos, var_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存均值对比图
        plot_filename = f"hotelling_t2_means_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_variables": n_vars,
            "sample_sizes": {
                "group1": n1,
                "group2": n2
            },
            "variable_names": var_names,
            "means": {
                "group1": mean1.tolist(),
                "group2": mean2.tolist()
            },
            "test_statistic": {
                "t2": float(t2_stat),
                "f": float(f_stat)
            },
            "degrees_of_freedom": {
                "numerator": p,
                "denominator": n1 + n2 - p - 1
            },
            "p_value": float(p_value),
            "effect_size": float(effect_size) if effect_size is not None else None,
            "significant": p_value < 0.05,
            "mean_plot_url": plot_url,
            "interpretation": _get_interpretation(p_value, effect_size)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"Hotelling's T²检验计算失败: {str(e)}") from e


def _get_interpretation(p_value: float, effect_size: float) -> str:
    """
    根据p值和效应量提供结果解释
    """
    interpretation = ""
    
    if p_value < 0.05:
        interpretation += "两组样本在多变量均值上存在显著差异。"
    else:
        interpretation += "两组样本在多变量均值上无显著差异。"
    
    if effect_size is not None:
        if effect_size < 0.2:
            interpretation += "效应量较小。"
        elif effect_size < 0.5:
            interpretation += "效应量中等。"
        else:
            interpretation += "效应量较大。"
    
    return interpretation