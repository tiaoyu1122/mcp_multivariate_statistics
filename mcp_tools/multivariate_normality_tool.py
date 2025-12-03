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

def perform_multivariate_normality_test(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行多元正态性检验，检验多变量数据是否符合多元正态分布
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    
    返回:
    - 包含检验结果的字典
    """
    try:
        # 输入验证
        n_vars = len(var_names)
        
        # 检查数据长度是否是变量数的倍数
        if len(data) % n_vars != 0:
            raise ValueError("数据长度必须是变量数量的整数倍")
        
        n_samples = len(data) // n_vars
        
        if n_vars == 0:
            raise ValueError("变量数量不能为0")
        
        # 将一维数组还原为嵌套列表
        data_nested = []
        for i in range(n_vars):
            var_data = data[i * n_samples:(i + 1) * n_samples]
            data_nested.append(var_data)
        
        # 检查所有变量的数据长度是否一致
        for i, var_data in enumerate(data_nested):
            if len(var_data) != n_samples:
                raise ValueError(f"变量 {var_names[i]} 的数据长度不一致")
        
        # 转换为numpy数组
        data_array = np.array(data_nested)  # shape: (n_vars, n_samples)
        
        # 转置为 (n_samples, n_vars)
        data_array = data_array.T  # shape: (n_samples, n_vars)
        
        # 计算均值向量和协方差矩阵
        mean_vector = np.mean(data_array, axis=0)
        cov_matrix = np.cov(data_array, rowvar=False)
        
        # 方法1: Mardia多变量正态性检验
        mardia_results = _mardia_multivariate_normality_test(data_array, mean_vector, cov_matrix)
        
        # 方法2: Henze-Zirkler检验
        hz_results = _henze_zirkler_test(data_array, mean_vector, cov_matrix)
        
        # 方法3: Royston检验
        royston_results = _royston_test(data_array)
        
        # 生成可视化图表
        plot_urls = _generate_visualizations(data_array, var_names)
        
        # 组织结果
        result = {
            "number_of_variables": n_vars,
            "sample_size": n_samples,
            "variable_names": var_names,
            "mean_vector": mean_vector.tolist(),
            "covariance_matrix": cov_matrix.tolist(),
            "tests": {
                "mardia": mardia_results,
                "henze_zirkler": hz_results,
                "royston": royston_results
            },
            "plots": plot_urls,
            "overall_normality": _assess_overall_normality(mardia_results, hz_results, royston_results)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"多元正态性检验计算失败: {str(e)}") from e


def _mardia_multivariate_normality_test(data: np.ndarray, mean_vector: np.ndarray, cov_matrix: np.ndarray) -> dict:
    """
    Mardia多变量正态性检验
    
    参数:
    - data: 样本数据 (n_samples, n_vars)
    - mean_vector: 均值向量
    - cov_matrix: 协方差矩阵
    
    返回:
    - 检验结果字典
    """
    n_samples, n_vars = data.shape
    
    # 计算马哈拉诺比斯距离平方
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    diff = data - mean_vector
    mahalanobis_sq = np.sum((diff @ inv_cov_matrix) * diff, axis=1)
    
    # Mardia skewness (偏度)
    b1p = np.sum(np.outer(mahalanobis_sq, mahalanobis_sq)) / (6 * n_samples)
    
    # Mardia kurtosis (峰度)
    b2p = np.mean(mahalanobis_sq)
    
    # 计算检验统计量
    # 偏度检验统计量 (近似卡方分布)
    skewness_stat = n_samples * b1p / 6
    skewness_df = n_vars * (n_vars + 1) * (n_vars + 2) / 6
    skewness_p = 1 - stats.chi2.cdf(skewness_stat, skewness_df)
    
    # 峰度检验统计量 (近似正态分布)
    kurtosis_stat = (b2p - n_vars) * np.sqrt(n_samples / (8 * n_vars * (n_vars + 2)))
    kurtosis_p = 2 * (1 - stats.norm.cdf(np.abs(kurtosis_stat)))
    
    return {
        "skewness": {
            "statistic": float(skewness_stat),
            "degrees_of_freedom": int(skewness_df),
            "p_value": float(skewness_p),
            "significant": skewness_p < 0.05
        },
        "kurtosis": {
            "statistic": float(kurtosis_stat),
            "p_value": float(kurtosis_p),
            "significant": kurtosis_p < 0.05
        }
    }


def _henze_zirkler_test(data: np.ndarray, mean_vector: np.ndarray, cov_matrix: np.ndarray) -> dict:
    """
    Henze-Zirkler多元正态性检验
    
    参数:
    - data: 样本数据 (n_samples, n_vars)
    - mean_vector: 均值向量
    - cov_matrix: 协方差矩阵
    
    返回:
    - 检验结果字典
    """
    n_samples, n_vars = data.shape
    
    # 计算beta参数
    beta = 1.0 / (n_samples ** (1.0 / (n_vars + 4.0)))
    
    # 计算样本协方差矩阵的行列式
    det_cov = np.linalg.det(cov_matrix)
    
    if det_cov <= 0:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "error": "协方差矩阵奇异"
        }
    
    # 计算各点的马哈拉诺比斯距离平方
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    diff = data - mean_vector
    mahalanobis_sq = np.sum((diff @ inv_cov_matrix) * diff, axis=1)
    
    # 计算样本特征函数
    phi_t = np.mean(np.exp(1j * (data @ mean_vector)))
    
    # 计算理论特征函数 (多元正态分布)
    # 对于多元正态分布，特征函数为 exp(i*t'*mu - 0.5*t'*Sigma*t)
    # 这里我们使用Henze-Zirkler的近似公式
    
    # 简化实现，使用近似方法
    term1 = (1.0 + beta**2)**(-n_vars/2.0) * np.exp(-beta**2 / (2.0 * (1.0 + beta**2)) * mahalanobis_sq)
    hz_stat = (1.0 / n_samples) * np.sum(term1) - 2.0 * (1.0 + 2.0*beta**2)**(-n_vars/2.0) + (1.0 + 4.0*beta**2)**(-n_vars/2.0)
    hz_stat = (1.0 / (beta**n_vars)) * hz_stat
    
    # 使用蒙特卡洛方法近似p值 (简化实现)
    # 实际应用中会使用更复杂的近似方法
    # 这里我们使用一个简化的近似公式
    mu = 1.0 - (n_vars * (n_vars + 2) * beta**2) / (n_samples * (1.0 + 2.0 * beta**2)**(n_vars/2.0 + 1.0))
    sigma = (2.0 * n_vars * (n_vars + 2) * beta**2) / (n_samples * (1.0 + 4.0 * beta**2)**(n_vars/2.0 + 1.0))
    
    if sigma > 0:
        standardized_stat = (hz_stat - mu) / np.sqrt(sigma)
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(standardized_stat)))
    else:
        p_value = None
    
    return {
        "statistic": float(hz_stat) if not np.isnan(hz_stat) else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": p_value < 0.05 if p_value is not None else None
    }


def _royston_test(data: np.ndarray) -> dict:
    """
    Royston多元正态性检验 (基于各变量的单变量Shapiro-Wilk检验)
    
    参数:
    - data: 样本数据 (n_samples, n_vars)
    
    返回:
    - 检验结果字典
    """
    n_samples, n_vars = data.shape
    
    # 对每个变量执行Shapiro-Wilk检验
    p_values = []
    for i in range(n_vars):
        try:
            stat, p_val = stats.shapiro(data[:, i])
            p_values.append(p_val)
        except:
            # 如果样本量太大，Shapiro-Wilk检验可能不适用
            p_values.append(None)
    
    # 使用Fisher方法组合p值
    valid_p_values = [p for p in p_values if p is not None]
    
    if len(valid_p_values) > 0:
        # Fisher's method: chi² = -2 * sum(ln(p_i))
        chi2_stat = -2 * np.sum(np.log(valid_p_values))
        df = 2 * len(valid_p_values)
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return {
            "statistic": float(chi2_stat),
            "degrees_of_freedom": df,
            "p_value": float(combined_p),
            "individual_p_values": [float(p) if p is not None else None for p in p_values],
            "significant": combined_p < 0.05
        }
    else:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "error": "无法计算单变量正态性检验"
        }


def _generate_visualizations(data: np.ndarray, var_names: List[str]) -> dict:
    """
    生成多元正态性检验的可视化图表
    
    参数:
    - data: 样本数据 (n_samples, n_vars)
    - var_names: 变量名称列表
    
    返回:
    - 图表URL字典
    """
    n_samples, n_vars = data.shape
    plot_urls = {}
    
    # 1. Q-Q图 (对每个变量)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(min(n_vars, 4)):  # 最多显示4个变量的Q-Q图
        ax = axes[i]
        stats.probplot(data[:, i], dist="norm", plot=ax)
        ax.set_title(f'{var_names[i]} Q-Q图')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(min(n_vars, 4), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    qq_plot_filename = f"multivariate_normality_qq_{uuid.uuid4().hex}.png"
    qq_plot_filepath = os.path.join(OUTPUT_DIR, qq_plot_filename)
    plt.savefig(qq_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_urls["qq_plot"] = f"{PUBLIC_FILE_BASE_URL}/{qq_plot_filename}"
    
    # 2. 散点图矩阵 (如果有2-4个变量)
    if 2 <= n_vars <= 4:
        fig, axes = plt.subplots(n_vars-1, n_vars-1, figsize=(10, 10))
        if n_vars == 2:
            axes = np.array([[axes]])
        
        for i in range(n_vars-1):
            for j in range(n_vars-1):
                if i <= j:
                    ax = axes[i, j] if n_vars > 2 else axes[0, 0]
                    if i == j:
                        # 对角线：直方图
                        ax.hist(data[:, i+1], bins=20, alpha=0.7, edgecolor='black')
                        ax.set_title(var_names[i+1])
                    else:
                        # 非对角线：散点图
                        ax.scatter(data[:, j], data[:, i+1], alpha=0.7)
                        ax.set_xlabel(var_names[j])
                        ax.set_ylabel(var_names[i+1])
                        ax.grid(True, alpha=0.3)
                elif n_vars > 2:
                    axes[i, j].set_visible(False)
        
        plt.tight_layout()
        
        scatter_plot_filename = f"multivariate_normality_scatter_{uuid.uuid4().hex}.png"
        scatter_plot_filepath = os.path.join(OUTPUT_DIR, scatter_plot_filename)
        plt.savefig(scatter_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_urls["scatter_plot"] = f"{PUBLIC_FILE_BASE_URL}/{scatter_plot_filename}"
    
    return plot_urls


def _assess_overall_normality(mardia_results: dict, hz_results: dict, royston_results: dict) -> dict:
    """
    综合评估多元正态性
    
    参数:
    - mardia_results: Mardia检验结果
    - hz_results: Henze-Zirkler检验结果
    - royston_results: Royston检验结果
    
    返回:
    - 综合评估结果
    """
    # 收集所有有效的p值
    p_values = []
    
    # Mardia检验 (偏度和峰度)
    if mardia_results["skewness"]["p_value"] is not None:
        p_values.append(mardia_results["skewness"]["p_value"])
    
    if mardia_results["kurtosis"]["p_value"] is not None:
        p_values.append(mardia_results["kurtosis"]["p_value"])
    
    # Henze-Zirkler检验
    if hz_results["p_value"] is not None:
        p_values.append(hz_results["p_value"])
    
    # Royston检验
    if royston_results["p_value"] is not None:
        p_values.append(royston_results["p_value"])
    
    # 综合判断
    if len(p_values) > 0:
        # 计算平均p值
        avg_p_value = np.mean(p_values)
        is_normal = avg_p_value >= 0.05
        
        # 判断一致性
        significant_count = sum(1 for p in p_values if p < 0.05)
        consensus = significant_count <= len(p_values) / 2  # 多数检验认为不显著则认为正态
        
        return {
            "average_p_value": float(avg_p_value),
            "is_normal": bool(is_normal),
            "consensus": bool(consensus),
            "interpretation": _get_normality_interpretation(avg_p_value, consensus)
        }
    else:
        return {
            "average_p_value": None,
            "is_normal": None,
            "consensus": None,
            "interpretation": "无法综合评估多元正态性"
        }


def _get_normality_interpretation(avg_p_value: float, consensus: bool) -> str:
    """
    根据平均p值和一致性提供解释
    
    参数:
    - avg_p_value: 平均p值
    - consensus: 检验结果是否一致
    
    返回:
    - 解释文本
    """
    if avg_p_value is None:
        return "无法综合评估多元正态性"
    
    interpretation = ""
    
    if avg_p_value >= 0.05:
        interpretation += "数据符合多元正态分布。"
    else:
        interpretation += "数据不符合多元正态分布。"
    
    if not consensus:
        interpretation += "但不同检验方法的结果存在分歧。"
    
    return interpretation