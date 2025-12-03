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


def perform_manova(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行多元方差分析(MANOVA)，用于检验多个组在多个因变量上的均值向量是否存在显著差异
    
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
        
        # 计算总样本数
        n_total = sum(n_samples)
        
        # 计算总体均值向量
        all_data = np.vstack(groups_arrays)
        grand_mean = np.mean(all_data, axis=0)
        
        # 计算组间平方和矩阵 (H矩阵)
        H = np.zeros((n_vars, n_vars))
        for i, group_array in enumerate(groups_arrays):
            group_mean = np.mean(group_array, axis=0)
            diff = group_mean - grand_mean
            H += n_samples[i] * np.outer(diff, diff)
        
        # 计算组内平方和矩阵 (E矩阵)
        E = np.zeros((n_vars, n_vars))
        for i, group_array in enumerate(groups_arrays):
            group_mean = np.mean(group_array, axis=0)
            for j in range(n_samples[i]):
                diff = group_array[j] - group_mean
                E += np.outer(diff, diff)
        
        # 计算Wilks' Lambda统计量
        # Λ = |E| / |H + E|
        try:
            wilks_lambda = np.linalg.det(E) / np.linalg.det(H + E)
        except np.linalg.LinAlgError:
            raise ValueError("无法计算Wilks' Lambda统计量，矩阵可能奇异")
        
        # 计算 Pillai's Trace 统计量
        # V = tr(H(H+E)^(-1))
        try:
            pillai_trace = np.trace(H @ np.linalg.inv(H + E))
        except np.linalg.LinAlgError:
            pillai_trace = None
        
        # 计算 Lawley-Hotelling Trace 统计量
        # U = tr(HE^(-1))
        try:
            hotelling_trace = np.trace(H @ np.linalg.inv(E))
        except np.linalg.LinAlgError:
            hotelling_trace = None
        
        # 计算 Roy's Largest Root 统计量
        #  наибольшее собственное значение HE^(-1)
        try:
            eigvals = np.linalg.eigvals(H @ np.linalg.inv(E))
            roy_largest_root = np.max(eigvals)
        except np.linalg.LinAlgError:
            roy_largest_root = None
        
        # 近似F统计量 (基于Wilks' Lambda)
        # 参考: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Multivariate_Analysis_of_Variance-MANOVA.pdf
        p = n_vars  # 因变量个数
        k = n_groups  # 组数
        N = n_total  # 总样本数
        
        # Wilks' Lambda的F近似
        if p == 1:
            # 当只有一个因变量时，等价于单因素方差分析
            df1 = k - 1
            df2 = N - k
            f_stat = ((N - k) / (k - 1)) * (1 - wilks_lambda) / wilks_lambda
        elif k == 2:
            # 当只有两组时，等价于Hotelling's T²检验
            df1 = p
            df2 = N - p - 1
            f_stat = ((N - p - 1) / p) * (1 - wilks_lambda) / wilks_lambda
        else:
            # 一般情况下的近似
            df1 = p * (k - 1)
            # 使用Bartlett近似
            b = (N - 1 - (p + k) / 2) / (N - 1 - p - k + 3/2)
            f_stat = ((N - k - p + 1) / (p * (k - 1))) * (1 - wilks_lambda**b) / wilks_lambda**b
            df2 = (N - k - p + 1) * (N - k - p + 3) / (N - k - 1/2)
        
        # 计算p值
        try:
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        except:
            p_value = None
        
        # 计算各组均值
        group_means = []
        for i, group_array in enumerate(groups_arrays):
            group_means.append(np.mean(group_array, axis=0).tolist())
        
        # 生成均值对比图
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(var_names))
        bar_width = 0.8 / n_groups
        
        for i in range(n_groups):
            plt.bar(x_pos + i * bar_width, group_means[i], bar_width, 
                   label=group_names[i], alpha=0.8)
        
        plt.xlabel('变量')
        plt.ylabel('均值')
        plt.title('各组样本均值对比')
        plt.xticks(x_pos + bar_width * (n_groups - 1) / 2, var_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存均值对比图
        plot_filename = f"manova_means_{uuid.uuid4().hex}.png"
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
            "group_means": {group_names[i]: group_means[i] for i in range(n_groups)},
            "test_statistics": {
                "wilks_lambda": float(wilks_lambda),
                "pillai_trace": float(pillai_trace) if pillai_trace is not None else None,
                "hotelling_trace": float(hotelling_trace) if hotelling_trace is not None else None,
                "roy_largest_root": float(roy_largest_root) if roy_largest_root is not None else None,
                "approximate_f": float(f_stat) if f_stat is not None else None
            },
            "degrees_of_freedom": {
                "numerator": float(df1) if df1 is not None else None,
                "denominator": float(df2) if df2 is not None else None
            },
            "p_value": float(p_value) if p_value is not None else None,
            "significant": p_value < 0.05 if p_value is not None else None,
            "mean_plot_url": plot_url,
            "interpretation": _get_interpretation(p_value)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"MANOVA计算失败: {str(e)}") from e


def _get_interpretation(p_value: float) -> str:
    """
    根据p值提供结果解释
    """
    if p_value is None:
        return "无法计算显著性水平。"
    
    if p_value < 0.05:
        return "各组在多变量均值上存在显著差异。"
    else:
        return "各组在多变量均值上无显著差异。"