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


def perform_stepwise_discriminant_analysis(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("wilks", description="逐步选择方法: 'wilks'(Wilks' Lambda), 'f'(F检验)"),
    significance_level_enter: float = Field(0.05, description="变量进入模型的显著性水平"),
    significance_level_remove: float = Field(0.10, description="变量移出模型的显著性水平"),
) -> dict:
    """
    执行逐步判别分析，通过逐步选择变量构建最优判别函数
    
    参数:
    - groups_data: 多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)
    - group_names: 组名称列表
    - var_names: 变量名称列表
    - method: 逐步选择方法 ('wilks' 或 'f')
    - significance_level_enter: 变量进入模型的显著性水平 (默认0.05)
    - significance_level_remove: 变量移出模型的显著性水平 (默认0.10)
    
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
        
        if method not in ["wilks", "f"]:
            raise ValueError("method 参数必须是 'wilks' 或 'f' 之一")
        
        # 将训练数据一维数组还原为嵌套列表
        # 首先计算每组的样本数
        total_groups_length = len(groups_data)
        if total_groups_length % (n_groups * n_vars) != 0:
            raise ValueError("训练数据长度与组数和变量数不匹配")
        
        n_samples_per_group = total_groups_length // (n_groups * n_vars)
        
        # 重新组织训练数据为嵌套列表: groups_data_nested[group][var][sample]
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
        
        # 计算组间散度矩阵 (S_B) 和组内散度矩阵 (S_W)
        S_B = np.zeros((n_vars, n_vars))
        S_W = np.zeros((n_vars, n_vars))
        
        for i in range(n_groups):
            # 组间散度
            diff = group_means[i] - grand_mean
            S_B += n_samples[i] * np.outer(diff, diff)
            
            # 组内散度
            group_mean = group_means[i]
            for j in range(n_samples[i]):
                diff = groups_arrays[i][j] - group_mean
                S_W += np.outer(diff, diff)
        
        # 检查组内散度矩阵是否可逆
        try:
            inv_S_W = np.linalg.inv(S_W)
        except np.linalg.LinAlgError:
            raise ValueError("组内散度矩阵奇异，无法计算判别函数")
        
        # 执行逐步选择
        selected_vars, steps = stepwise_selection(
            all_data, groups_arrays, group_names, 
            method, significance_level_enter, significance_level_remove
        )
        
        if len(selected_vars) == 0:
            raise ValueError("逐步选择未选择任何变量")
        
        # 对选中的变量进行Fisher判别分析
        n_selected_vars = len(selected_vars)
        selected_indices = selected_vars  # stepwise_selection已经返回了索引列表
        
        # 提取选中变量的数据
        selected_all_data = all_data[:, selected_indices]
        selected_groups_arrays = [group_array[:, selected_indices] for group_array in groups_arrays]
        
        # 重新计算选中变量的统计量
        selected_group_means = []
        for group_array in selected_groups_arrays:
            selected_group_means.append(np.mean(group_array, axis=0))
        
        selected_grand_mean = np.mean(selected_all_data, axis=0)
        
        # 计算选中变量的组间和组内散度矩阵
        selected_S_B = np.zeros((n_selected_vars, n_selected_vars))
        selected_S_W = np.zeros((n_selected_vars, n_selected_vars))
        
        for i in range(n_groups):
            # 组间散度
            diff = selected_group_means[i] - selected_grand_mean
            selected_S_B += n_samples[i] * np.outer(diff, diff)
            
            # 组内散度
            group_mean = selected_group_means[i]
            for j in range(n_samples[i]):
                diff = selected_groups_arrays[i][j] - group_mean
                selected_S_W += np.outer(diff, diff)
        
        # 检查选中变量的组内散度矩阵是否可逆
        try:
            selected_inv_S_W = np.linalg.inv(selected_S_W)
        except np.linalg.LinAlgError:
            raise ValueError("选中变量的组内散度矩阵奇异，无法计算判别函数")
        
        # 计算判别函数系数
        try:
            # 计算特征值和特征向量
            eigenvals, eigenvecs = np.linalg.eig(selected_inv_S_W @ selected_S_B)
            
            # 按特征值降序排列
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # 选择前 min(n_groups-1, n_selected_vars) 个判别函数
            n_discriminants = min(n_groups - 1, n_selected_vars)
            eigenvals = eigenvals[:n_discriminants]
            eigenvecs = eigenvecs[:, :n_discriminants]
            
        except np.linalg.LinAlgError:
            raise ValueError("无法计算判别函数系数")
        
        # 计算每个判别函数的解释方差比例
        total_eigenval = np.sum(eigenvals)
        explained_variance_ratio = eigenvals / total_eigenval if total_eigenval > 0 else np.zeros_like(eigenvals)
        
        # 计算组在判别函数上的得分
        discriminant_scores = []
        for i, group_array in enumerate(selected_groups_arrays):
            scores = group_array @ eigenvecs
            discriminant_scores.append(scores)
        
        # 计算组中心在判别函数上的得分
        group_centroids = []
        for i in range(n_groups):
            centroid_scores = selected_group_means[i] @ eigenvecs
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
            chi2_stat = -2 * np.log(wilks_lambda)
            df_chi2 = (n_selected_vars - i) * (n_groups - 1 - i)
            chi2_stats.append(chi2_stat)
            
            # 转换为F统计量
            if df_chi2 > 0:
                df_f1 = df_chi2
                df_f2 = n_total - n_groups - (n_selected_vars + n_groups - 1)/2
                if df_f2 > 0:
                    f_stat = (chi2_stat / df_chi2) * df_f2
                    f_stats.append(f_stat)
                    # 计算p值
                    p_val = 1 - stats.f.cdf(f_stat, df_f1, df_f2)
                    p_values.append(p_val)
                else:
                    f_stats.append(None)
                    p_values.append(None)
            else:
                f_stats.append(None)
                p_values.append(None)
        
        # 生成可视化图
        plt.figure(figsize=(12, 5))
        
        # 绘制逐步选择过程
        plt.subplot(1, 2, 1)
        step_numbers = [step["step"] for step in steps]
        f_values = [step.get("f_value", 0) for step in steps]
        
        colors = ['green' if step["action"] == "added" else 'red' for step in steps]
        plt.bar(step_numbers, f_values, color=colors)
        plt.xlabel('步骤')
        plt.ylabel('F值')
        plt.title('逐步选择过程')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        added_patch = plt.Rectangle((0,0),1,1,fc='green')
        removed_patch = plt.Rectangle((0,0),1,1,fc='red')
        plt.legend([added_patch, removed_patch], ['变量加入', '变量移除'])
        
        # 绘制判别函数散点图（如果有至少2个选中变量）
        plt.subplot(1, 2, 2)
        if n_discriminants >= 2:
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
        else:
            # 只有一个判别函数的情况
            group_indices = range(n_groups)
            means = [group_centroids[i][0] for i in range(n_groups)]
            plt.bar(group_indices, means)
            plt.xlabel('组别')
            plt.ylabel('判别函数得分')
            plt.xticks(group_indices, group_names, rotation=45)
            plt.title('各组判别函数得分')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存逐步判别分析图
        plot_filename = f"stepwise_discriminant_analysis_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_groups": n_groups,
            "number_of_total_variables": n_vars,
            "number_of_selected_variables": n_selected_vars,
            "selected_variables": selected_vars,
            "group_names": group_names,
            "variable_names": var_names,
            "sample_sizes": {group_names[i]: n_samples[i] for i in range(n_groups)},
            "steps": steps,
            "number_of_discriminant_functions": int(n_discriminants),
            "eigenvalues": eigenvals.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "group_centroids": {group_names[i]: group_centroids[i].tolist() for i in range(n_groups)},
            "discriminant_effectiveness_tests": {
                "wilks_lambda": wilks_lambda_values,
                "chi_square_statistics": chi2_stats,
                "f_statistics": f_stats,
                "p_values": p_values
            },
            "stepwise_plot_url": plot_url,
            "interpretation": _get_interpretation(explained_variance_ratio, n_discriminants, selected_vars, steps)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"逐步判别分析计算失败: {str(e)}") from e


def stepwise_selection(data, groups_arrays, group_names, method, sl_enter, sl_remove):
    """
    执行逐步选择
    
    参数:
    - data: 所有数据
    - groups_arrays: 各组数据列表
    - group_names: 组名称列表
    - method: 选择方法
    - sl_enter: 进入显著性水平
    - sl_remove: 移除显著性水平
    
    返回:
    - selected_vars: 选中的变量索引列表
    - steps: 步骤记录
    """
    n_vars = data.shape[1]
    n_groups = len(groups_arrays)
    n_samples = [group.shape[0] for group in groups_arrays]
    n_total = sum(n_samples)
    
    # 初始化
    selected_indices = []
    steps = []
    step_count = 0
    
    # 所有变量索引
    all_indices = list(range(n_vars))
    
    while True:
        changed = False
        
        # 前向选择
        if len(selected_indices) < n_vars:
            excluded_indices = [i for i in all_indices if i not in selected_indices]
            best_f = 0
            best_var = None
            
            for var_idx in excluded_indices:
                # 临时添加变量
                temp_indices = selected_indices + [var_idx]
                temp_f = calculate_f_statistic(data, groups_arrays, temp_indices)
                
                if temp_f > best_f and temp_f > stats.f.ppf(1 - sl_enter, n_groups - 1, n_total - n_groups):
                    best_f = temp_f
                    best_var = var_idx
            
            if best_var is not None:
                selected_indices.append(best_var)
                changed = True
                step_count += 1
                steps.append({
                    "step": step_count,
                    "action": "added",
                    "variable_index": best_var,
                    "f_value": best_f
                })
        
        # 后向消除
        if len(selected_indices) > 1:
            worst_f = float('inf')
            worst_var = None
            
            for var_idx in selected_indices:
                # 临时移除变量
                temp_indices = [i for i in selected_indices if i != var_idx]
                if len(temp_indices) > 0:
                    temp_f = calculate_f_statistic(data, groups_arrays, temp_indices)
                    
                    if temp_f < worst_f and temp_f < stats.f.ppf(1 - sl_remove, n_groups - 1, n_total - n_groups):
                        worst_f = temp_f
                        worst_var = var_idx
            
            if worst_var is not None:
                selected_indices.remove(worst_var)
                changed = True
                step_count += 1
                steps.append({
                    "step": step_count,
                    "action": "removed",
                    "variable_index": worst_var,
                    "f_value": worst_f
                })
        
        if not changed:
            break
    
    return selected_indices, steps


def calculate_f_statistic(data, groups_arrays, selected_indices):
    """
    计算Wilks' Lambda对应的F统计量
    """
    try:
        n_vars = len(selected_indices)
        if n_vars == 0:
            return 0
            
        n_groups = len(groups_arrays)
        n_samples = [group.shape[0] for group in groups_arrays]
        n_total = sum(n_samples)
        
        # 提取选中变量的数据
        selected_data = data[:, selected_indices]
        selected_groups_arrays = [group_array[:, selected_indices] for group_array in groups_arrays]
        
        # 计算各组均值向量
        group_means = []
        for group_array in selected_groups_arrays:
            group_means.append(np.mean(group_array, axis=0))
        
        grand_mean = np.mean(selected_data, axis=0)
        
        # 计算组间散度矩阵 (S_B) 和组内散度矩阵 (S_W)
        S_B = np.zeros((n_vars, n_vars))
        S_W = np.zeros((n_vars, n_vars))
        
        for i in range(n_groups):
            # 组间散度
            diff = group_means[i] - grand_mean
            S_B += n_samples[i] * np.outer(diff, diff)
            
            # 组内散度
            group_mean = group_means[i]
            for j in range(n_samples[i]):
                diff = selected_groups_arrays[i][j] - group_mean
                S_W += np.outer(diff, diff)
        
        # 检查矩阵是否可逆
        try:
            inv_S_W = np.linalg.inv(S_W)
        except np.linalg.LinAlgError:
            return 0
        
        # 计算Wilks' Lambda
        try:
            wilks_lambda = np.linalg.det(S_W) / np.linalg.det(S_B + S_W)
            # 转换为F统计量
            if n_vars > 0 and n_groups > 1:
                # 近似F统计量
                f_stat = ((n_total - n_groups - n_vars + 1) / (n_vars * (n_groups - 1))) * ((1 - wilks_lambda) / wilks_lambda)
                return float(f_stat)
            else:
                return 0
        except:
            return 0
    except:
        return 0


def _get_interpretation(explained_variance_ratio: np.ndarray, n_discriminants: int, selected_vars: List[int], steps: List[Dict]) -> str:
    """
    根据解释方差比例提供结果解释
    """
    interpretation = f"逐步判别分析结果：\n"
    interpretation += f"共 {len(steps)} 个步骤，最终选择了 {len(selected_vars)} 个变量。\n"
    interpretation += f"共计算出 {n_discriminants} 个判别函数。\n"
    
    for i in range(min(3, n_discriminants)):  # 最多显示前3个
        interpretation += f"判别函数 {i+1} 解释了 {explained_variance_ratio[i]*100:.1f}% 的方差。\n"
    
    if n_discriminants > 0 and explained_variance_ratio[0] > 0.7:
        interpretation += "第一个判别函数能够很好地分离不同组别。\n"
    elif n_discriminants > 0 and explained_variance_ratio[0] > 0.5:
        interpretation += "第一个判别函数对组别分离有一定效果。\n"
    else:
        interpretation += "判别函数对组别分离效果较弱。\n"
    
    interpretation += "\n该工具使用逐步判别法，通过逐步选择变量来构建最优的判别函数。"
    interpretation += "综合考虑了变量的判别能力和统计显著性，避免了冗余变量的干扰。"
    
    return interpretation