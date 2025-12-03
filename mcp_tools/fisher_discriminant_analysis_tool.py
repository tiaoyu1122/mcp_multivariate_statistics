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


def perform_fisher_discriminant_analysis(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行线性判别分析(LDA)，用于寻找最能区分不同组的线性组合
    
    参数:
    - groups_data: 多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)
    - group_names: 组名称列表
    - var_names: 变量名称列表
    
    返回:
    - 包含判别分析结果的字典
    """
    try:
        # 输入验证
        n_groups = len(group_names)
        n_vars = len(var_names)
        
        if n_groups < 2:
            raise ValueError("至少需要两组数据进行判别分析")
            
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
        
        # 计算各组均值向量
        group_means = []
        for i, group_array in enumerate(groups_arrays):
            group_means.append(np.mean(group_array, axis=0))
        
        # 计算总体均值向量
        all_data = np.vstack(groups_arrays)
        grand_mean = np.mean(all_data, axis=0)
        
        # 计算组间散度矩阵 (S_B)
        S_B = np.zeros((n_vars, n_vars))
        for i in range(n_groups):
            diff = group_means[i] - grand_mean
            S_B += n_samples[i] * np.outer(diff, diff)
        
        # 计算组内散度矩阵 (S_W)
        S_W = np.zeros((n_vars, n_vars))
        for i, group_array in enumerate(groups_arrays):
            group_mean = group_means[i]
            for j in range(n_samples[i]):
                diff = group_array[j] - group_mean
                S_W += np.outer(diff, diff)
        
        # 检查组内散度矩阵是否可逆
        try:
            inv_S_W = np.linalg.inv(S_W)
        except np.linalg.LinAlgError:
            raise ValueError("组内散度矩阵奇异，无法计算判别函数")
        
        # 计算判别函数系数
        # 通过求解 S_W^(-1) * S_B 的特征值和特征向量
        try:
            # 计算特征值和特征向量
            eigenvals, eigenvecs = np.linalg.eig(inv_S_W @ S_B)
            
            # 确保只使用实数部分，避免复数
            eigenvals = np.real(eigenvals)
            eigenvecs = np.real(eigenvecs)
            
            # 按特征值降序排列
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # 选择前 min(n_groups-1, n_vars) 个判别函数
            n_discriminants = min(n_groups - 1, n_vars)
            eigenvals = eigenvals[:n_discriminants]
            eigenvecs = eigenvecs[:, :n_discriminants]
            
        except np.linalg.LinAlgError:
            raise ValueError("无法计算判别函数系数")
        
        # 计算每个判别函数的解释方差比例
        total_eigenval = np.sum(eigenvals)
        explained_variance_ratio = eigenvals / total_eigenval if total_eigenval > 0 else np.zeros_like(eigenvals)
        
        # 计算标准化判别函数系数
        # 首先计算每个变量的标准差
        pooled_std = np.sqrt(np.diag(S_W) / (n_total - n_groups))
        standardized_coefficients = eigenvecs * pooled_std[:, np.newaxis]
        
        # 计算组在判别函数上的得分
        discriminant_scores = []
        for i, group_array in enumerate(groups_arrays):
            scores = group_array @ eigenvecs
            discriminant_scores.append(scores)
        
        # 计算组中心在判别函数上的得分
        group_centroids = []
        for i in range(n_groups):
            centroid_scores = group_means[i] @ eigenvecs
            group_centroids.append(centroid_scores)
        
        # 计算判别效果检验指标
        # Wilks' Lambda 检验每个判别函数的显著性
        wilks_lambda_values = []
        chi2_stats = []
        f_stats = []
        p_values = []
        
        # 对每个判别函数进行显著性检验
        for i in range(n_discriminants):
            # 计算第i个判别函数及之后所有判别函数的Wilks' Lambda
            remaining_eigenvals = eigenvals[i:]
            wilks_lambda = np.prod(1 / (1 + remaining_eigenvals))
            wilks_lambda_values.append(wilks_lambda)
            
            # 使用似然比检验计算卡方统计量
            # -2 * ln(Lambda) ~ chi2(df)
            chi2_stat = -2 * np.log(wilks_lambda)
            df_chi2 = (n_vars - i) * (n_groups - 1 - i)
            chi2_stats.append(float(chi2_stat))
            
            # 转换为F统计量
            if df_chi2 > 0:
                # F = (chi2_stat / df_chi2) * (n_total - n_groups - (n_vars + n_groups - 1)/2)
                df_f1 = float(df_chi2)
                df_f2 = n_total - n_groups - (n_vars + n_groups - 1)/2
                if df_f2 > 0:
                    f_stat = (chi2_stat / df_chi2) * df_f2
                    f_stats.append(float(f_stat))
                    # 计算p值，确保参数类型正确
                    try:
                        p_val = 1 - stats.f.cdf(float(f_stat), int(df_f1), int(df_f2))
                        p_values.append(float(p_val))
                    except Exception:
                        p_values.append(None)
                else:
                    f_stats.append(None)
                    p_values.append(None)
            else:
                f_stats.append(None)
                p_values.append(None)
        
        # 计算分类准确率相关指标
        # 通过计算组间和组内的马氏距离来评估分类效果
        mahalanobis_distances = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                diff = group_means[i] - group_means[j]
                try:
                    # 计算马氏距离
                    mahalanobis_dist = np.sqrt(diff.T @ inv_S_W @ diff)
                    mahalanobis_distances.append({
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "mahalanobis_distance": float(mahalanobis_dist)
                    })
                except np.linalg.LinAlgError:
                    mahalanobis_distances.append({
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "mahalanobis_distance": None
                    })
        
        # 计算结构相关系数（判别函数与原始变量的相关性）
        structure_coefficients = []
        for i in range(n_discriminants):
            # 计算每个判别函数与原始变量的相关性
            discr_scores = all_data @ eigenvecs[:, i]
            correlations = []
            for j in range(n_vars):
                corr, _ = stats.pearsonr(all_data[:, j], discr_scores)
                correlations.append(corr)
            structure_coefficients.append(correlations)
        
        # 生成判别函数可视化图
        if n_discriminants >= 2:
            plt.figure(figsize=(12, 5))
            
            # 绘制前两个判别函数的散点图
            plt.subplot(1, 2, 1)
            colors = plt.cm.get_cmap('tab10', n_groups)
            for i in range(n_groups):
                scores = discriminant_scores[i]
                plt.scatter(scores[:, 0], scores[:, 1], 
                           label=group_names[i], alpha=0.7, color=colors(i))
                # 绘制组中心
                plt.scatter(group_centroids[i][0], group_centroids[i][1], 
                           marker='x', s=200, color=colors(i), linewidth=3)
            
            plt.xlabel(f'判别函数 1 (解释方差: {explained_variance_ratio[0]*100:.1f}%)')
            plt.ylabel(f'判别函数 2 (解释方差: {explained_variance_ratio[1]*100:.1f}%)')
            plt.title('判别函数散点图')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制判别函数系数热力图
            plt.subplot(1, 2, 2)
            im = plt.imshow(eigenvecs, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.yticks(range(n_vars), var_names)
            plt.xticks(range(n_discriminants), [f'函数{i+1}' for i in range(n_discriminants)])
            plt.title('判别函数系数')
            
            # 在每个单元格中添加数值
            for i in range(n_vars):
                for j in range(n_discriminants):
                    plt.text(j, i, f'{eigenvecs[i, j]:.3f}', 
                            ha="center", va="center", 
                            color="white" if abs(eigenvecs[i, j]) > np.max(np.abs(eigenvecs))/2 else "black")
            
            plt.tight_layout()
        else:
            # 只有一个判别函数的情况
            plt.figure(figsize=(10, 6))
            
            # 绘制第一个判别函数的分布图
            plt.subplot(1, 2, 1)
            for i in range(n_groups):
                scores = discriminant_scores[i][:, 0]
                plt.hist(scores, alpha=0.7, label=group_names[i], bins=20)
            
            plt.xlabel('判别函数得分')
            plt.ylabel('频率')
            plt.title(f'判别函数 1 (解释方差: {explained_variance_ratio[0]*100:.1f}%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制判别函数系数条形图
            plt.subplot(1, 2, 2)
            bars = plt.barh(var_names, eigenvecs[:, 0])
            plt.xlabel('判别函数系数')
            plt.title('判别函数系数')
            
            # 为条形图添加颜色
            for i, bar in enumerate(bars):
                if eigenvecs[i, 0] >= 0:
                    bar.set_color('blue')
                else:
                    bar.set_color('red')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        # 保存判别分析图
        plot_filename = f"discriminant_analysis_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_groups": n_groups,
            "number_of_variables": n_vars,
            "number_of_discriminant_functions": int(n_discriminants),
            "group_names": group_names,
            "variable_names": var_names,
            "sample_sizes": {group_names[i]: n_samples[i] for i in range(n_groups)},
            "eigenvalues": eigenvals.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "discriminant_function_coefficients": eigenvecs.tolist(),
            "standardized_discriminant_coefficients": standardized_coefficients.tolist(),
            "group_centroids": {group_names[i]: group_centroids[i].tolist() for i in range(n_groups)},
            "discriminant_effectiveness_tests": {
                "wilks_lambda": wilks_lambda_values,
                "chi_square_statistics": chi2_stats,
                "f_statistics": f_stats,
                "p_values": p_values
            },
            "mahalanobis_distances": mahalanobis_distances,
            "structure_coefficients": structure_coefficients,
            "discriminant_plot_url": plot_url,
            "interpretation": _get_interpretation(explained_variance_ratio, n_discriminants)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"判别分析计算失败: {str(e)}") from e


def _get_interpretation(explained_variance_ratio: np.ndarray, n_discriminants: int) -> str:
    """
    根据解释方差比例提供结果解释
    """
    interpretation = f"共计算出 {n_discriminants} 个判别函数。\n"
    
    for i in range(min(3, n_discriminants)):  # 最多显示前3个
        interpretation += f"判别函数 {i+1} 解释了 {explained_variance_ratio[i]*100:.1f}% 的方差。\n"
    
    if n_discriminants > 0 and explained_variance_ratio[0] > 0.7:
        interpretation += "第一个判别函数能够很好地分离不同组别。\n"
    elif n_discriminants > 0 and explained_variance_ratio[0] > 0.5:
        interpretation += "第一个判别函数对组别分离有一定效果。\n"
    else:
        interpretation += "判别函数对组别分离效果较弱。\n"
    
    interpretation += "\n该工具使用线性判别分析（LDA），也称为Fisher判别法。它通过最大化组间差异与组内差异的比值来寻找最优的线性判别函数。"
    
    return interpretation