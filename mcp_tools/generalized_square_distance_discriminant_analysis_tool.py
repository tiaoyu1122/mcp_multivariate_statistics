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


def perform_generalized_square_distance_discriminant_analysis(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    test_data: List[float] = Field(None, description="待判别样本数据(可选)，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
) -> dict:
    """
    执行广义平方距离判别分析，考虑各组协方差矩阵不等的情况
    
    参数:
    - groups_data: 多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)
    - group_names: 组名称列表
    - var_names: 变量名称列表
    - test_data: 待判别样本数据(可选)，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)
    
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
        
        # 如果提供了测试数据，验证其格式并重组
        test_samples = []
        if test_data is not None:
            if len(test_data) % n_vars != 0:
                raise ValueError("测试数据长度与变量数量不匹配")
            
            n_test = len(test_data) // n_vars
            
            # 重新组织测试数据为嵌套列表: test_data_nested[var][sample]
            test_data_nested = []
            for v in range(n_vars):
                var_data = []
                for s in range(n_test):
                    var_data.append(test_data[v * n_test + s])
                test_data_nested.append(var_data)
            
            test_samples = list(zip(*test_data_nested))  # 转置为样本列表
        else:
            n_test = 0
        
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
        
        # 计算各组协方差矩阵
        group_covs = []
        for i, group_array in enumerate(groups_arrays):
            cov = np.cov(group_array, rowvar=False)
            group_covs.append(cov)
        
        # 计算组协方差矩阵的逆和行列式
        group_inv_covs = []
        group_log_dets = []
        for i, cov in enumerate(group_covs):
            try:
                inv_cov = np.linalg.inv(cov)
                log_det = np.log(np.linalg.det(cov))
                group_inv_covs.append(inv_cov)
                group_log_dets.append(log_det)
            except np.linalg.LinAlgError:
                raise ValueError(f"第{i+1}组协方差矩阵奇异，无法计算广义平方距离")
        
        # 对测试样本进行广义平方距离判别（如果提供了测试数据）
        test_classifications = []
        if test_data is not None:
            for i, test_sample in enumerate(test_samples):
                test_sample = np.array(test_sample)
                distances = []
                
                # 计算每个组的广义平方距离
                for j in range(n_groups):
                    # 计算马氏距离的平方
                    diff = test_sample - group_means[j]
                    mahalanobis_sq = diff.T @ group_inv_covs[j] @ diff
                    
                    # 广义平方距离 = 马氏距离的平方 + ln|Σ_j|
                    generalized_sq_dist = mahalanobis_sq + group_log_dets[j]
                    distances.append((group_names[j], float(generalized_sq_dist), float(mahalanobis_sq)))
                
                # 选择广义平方距离最小的组作为分类结果
                closest_group = min(distances, key=lambda x: x[1])
                test_classifications.append({
                    "sample_index": i+1,
                    "classified_group": closest_group[0],
                    "generalized_square_distances": dict([(name, dist) for name, dist, _ in distances]),
                    "mahalanobis_square_distances": dict([(name, dist) for name, _, dist in distances])
                })
        
        # 计算组间马氏距离（使用组平均协方差矩阵）
        pooled_cov = np.zeros((n_vars, n_vars))
        for i in range(n_groups):
            pooled_cov += (n_samples[i] - 1) * group_covs[i]
        pooled_cov /= (n_total - n_groups)
        
        try:
            inv_pooled_cov = np.linalg.inv(pooled_cov)
        except np.linalg.LinAlgError:
            inv_pooled_cov = np.linalg.pinv(pooled_cov)  # 使用伪逆
            
        mahalanobis_distances = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                diff = group_means[i] - group_means[j]
                mahalanobis_dist = np.sqrt(diff.T @ inv_pooled_cov @ diff)
                mahalanobis_distances.append({
                    "group1": group_names[i],
                    "group2": group_names[j],
                    "mahalanobis_distance": float(mahalanobis_dist)
                })
        
        # 生成可视化图
        plt.figure(figsize=(12, 5))
        
        # 绘制广义平方距离对比（如果有测试数据）
        if test_data is not None and len(test_classifications) > 0:
            plt.subplot(1, 2, 1)
            # 绘制第一个测试样本的广义平方距离
            first_sample = test_classifications[0]
            gen_sq_distances = [first_sample["generalized_square_distances"][name] for name in group_names]
            
            plt.bar(range(n_groups), gen_sq_distances)
            plt.xlabel('组别')
            plt.ylabel('广义平方距离')
            plt.title('广义平方距离（样本1）')
            plt.xticks(range(n_groups), group_names, rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(1, 1, 1)
        
        # 绘制组均值的马氏距离热力图（如果有至少2个变量）
        if n_vars >= 2:
            # 创建距离矩阵
            dist_matrix = np.zeros((n_groups, n_groups))
            group_idx_map = {name: idx for idx, name in enumerate(group_names)}
            
            for dist_info in mahalanobis_distances:
                i = group_idx_map[dist_info["group1"]]
                j = group_idx_map[dist_info["group2"]]
                dist_matrix[i, j] = dist_info["mahalanobis_distance"]
                dist_matrix[j, i] = dist_info["mahalanobis_distance"]
            
            # 对角线设为0
            np.fill_diagonal(dist_matrix, 0)
            
            im = plt.imshow(dist_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(n_groups), group_names, rotation=45)
            plt.yticks(range(n_groups), group_names)
            plt.title('组间马氏距离矩阵')
            
            # 在每个单元格中添加数值
            for i in range(n_groups):
                for j in range(n_groups):
                    if i != j:
                        plt.text(j, i, f'{dist_matrix[i, j]:.3f}', 
                                ha="center", va="center", color="white")
                    else:
                        plt.text(j, i, '0', 
                                ha="center", va="center", color="white")
        else:
            # 只有一个变量的情况
            group_indices = range(n_groups)
            means = [group_means[i][0] for i in group_indices]
            plt.bar(group_indices, means)
            plt.xlabel('组别')
            plt.ylabel('均值')
            plt.xticks(group_indices, group_names, rotation=45)
            plt.title('各组均值')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存广义平方距离判别分析图
        plot_filename = f"generalized_square_distance_discriminant_analysis_{uuid.uuid4().hex}.png"
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
            "group_means": {group_names[i]: group_means[i].tolist() for i in range(n_groups)},
            "mahalanobis_distances": mahalanobis_distances,
            "generalized_square_distance_plot_url": plot_url,
            "interpretation": _get_interpretation(mahalanobis_distances)
        }
        
        # 如果有测试数据，添加分类结果
        if test_data is not None:
            result["test_classifications"] = test_classifications
            result["number_of_test_samples"] = n_test
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"广义平方距离判别分析计算失败: {str(e)}") from e


def _get_interpretation(mahalanobis_distances: List[Dict]) -> str:
    """
    根据马氏距离提供结果解释
    """
    if not mahalanobis_distances:
        return "未计算出有效的马氏距离。"
    
    # 找到最大的组间距离
    max_distance = max(mahalanobis_distances, key=lambda x: x["mahalanobis_distance"])
    
    interpretation = f"广义平方距离判别分析结果：\n"
    interpretation += f"最远的组对是 {max_distance['group1']} 和 {max_distance['group2']}，"
    interpretation += f"马氏距离为 {max_distance['mahalanobis_distance']:.3f}。\n"
    
    if max_distance['mahalanobis_distance'] > 3:
        interpretation += "组间差异较大，判别效果可能较好。\n"
    elif max_distance['mahalanobis_distance'] > 1:
        interpretation += "组间有一定差异，判别效果中等。\n"
    else:
        interpretation += "组间差异较小，判别效果可能较差。\n"
    
    interpretation += "\n该工具使用广义平方距离判别法，考虑了各组协方差矩阵不等的情况。"
    interpretation += "通过计算广义平方距离（马氏距离的平方加上协方差矩阵行列式的对数）来进行分类。"
    
    return interpretation