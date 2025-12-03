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

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()

import hdbscan


def perform_hdbscan_clustering(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    min_cluster_size: int = Field(5, description="形成簇所需的最小样本数"),
    min_samples: Optional[int] = Field(None, description="核心点邻域中的最小样本数，如果为None则默认等于min_cluster_size"),
    cluster_selection_method: str = Field("eom", description="簇选择方法: 'eom'(超额质量算法), 'leaf'(叶簇选择)"),
    allow_single_cluster: bool = Field(False, description="是否允许将所有点归为一个簇"),
    alpha: float = Field(1.0, description="用于计算不稳定性的参数"),
    metric: str = Field("euclidean", description="距离度量方法"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行HDBSCAN聚类分析
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - min_cluster_size: 形成簇所需的最小样本数，默认为5
    - min_samples: 核心点邻域中的最小样本数，如果为None则默认等于min_cluster_size
    - cluster_selection_method: 簇选择方法: 'eom'(超额质量算法), 'leaf'(叶簇选择)，默认为'eom'
    - allow_single_cluster: 是否允许将所有点归为一个簇，默认为False
    - alpha: 用于计算不稳定性的参数，默认为1.0
    - metric: 距离度量方法，默认为"euclidean"
    - standardize: 是否标准化数据，默认为True
    
    返回:
    - 包含聚类结果的字典
    """
    try:
        # 输入验证
        if not data:
            raise ValueError("数据不能为空")
        
        n_vars = len(var_names)
        if n_vars == 0:
            raise ValueError("变量数量不能为0")
            
        if len(data) % n_vars != 0:
            raise ValueError("数据长度与变量数量不匹配")
            
        n_samples = len(data) // n_vars
        if n_samples == 0:
            raise ValueError("样本数量不能为0")
        
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size 必须大于等于2")
        
        if min_samples is not None and min_samples < 1:
            raise ValueError("min_samples 必须大于等于1")
        
        if cluster_selection_method not in ["eom", "leaf"]:
            raise ValueError("cluster_selection_method 必须是 'eom' 或 'leaf'")
        
        # 将一维数组重构为嵌套列表
        data_nested = []
        for i in range(n_vars):
            var_data = data[i * n_samples : (i + 1) * n_samples]
            data_nested.append(var_data)
        
        # 转换为numpy数组并转置为 (n_samples, n_vars)
        data_array = np.array(data_nested).T  # shape: (n_samples, n_vars)
        
        # 数据标准化
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_array)
        else:
            scaled_data = data_array
        
        # 执行HDBSCAN聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            alpha=alpha,
            metric=metric
        )
        
        cluster_labels = clusterer.fit_predict(scaled_data)
        
        # 获取簇的概率和异常值分数
        cluster_probabilities = clusterer.probabilities_
        outlier_scores = clusterer.outlier_scores_
        
        # 计算簇的数量（不包括噪声点-1）
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = np.sum(cluster_labels == -1)
        
        # 计算评估指标（如果有至少2个簇）
        silhouette_avg = None
        ch_score = None
        if n_clusters >= 2:
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                # 只使用非噪声点计算评估指标
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = float(silhouette_score(scaled_data[non_noise_mask], cluster_labels[non_noise_mask]))
                    ch_score = float(calinski_harabasz_score(scaled_data[non_noise_mask], cluster_labels[non_noise_mask]))
            except:
                pass
        
        # 计算每个簇的统计信息
        cluster_stats = []
        for label in unique_labels:
            if label == -1:
                # 噪声点统计
                cluster_data = scaled_data[cluster_labels == label]
                cluster_size = cluster_data.shape[0]
                
                cluster_stats.append({
                    "cluster_id": int(label),
                    "size": int(cluster_size),
                    "is_noise": True
                })
            else:
                # 正常簇统计
                cluster_data = scaled_data[cluster_labels == label]
                cluster_size = cluster_data.shape[0]
                
                # 计算每个簇的均值和标准差
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                # 计算簇内平均概率和异常值分数
                cluster_probs = cluster_probabilities[cluster_labels == label]
                cluster_outliers = outlier_scores[cluster_labels == label] if outlier_scores is not None else None
                
                stat_entry = {
                    "cluster_id": int(label),
                    "size": int(cluster_size),
                    "mean": cluster_mean.tolist(),
                    "std": cluster_std.tolist(),
                    "mean_probability": float(np.mean(cluster_probs)) if cluster_probs is not None else None,
                }
                
                if cluster_outliers is not None:
                    stat_entry["mean_outlier_score"] = float(np.mean(cluster_outliers))
                
                cluster_stats.append(stat_entry)
        
        # 生成聚类结果可视化图
        # 如果变量数大于2，使用PCA降维到2D进行可视化
        if n_vars > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(scaled_data)
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
        else:
            data_2d = scaled_data[:, :2] if n_vars >= 2 else np.hstack([scaled_data, np.zeros((n_samples, 1))])
            xlabel = var_names[0] if n_vars >= 1 else "Variable 1"
            ylabel = var_names[1] if n_vars >= 2 else "Variable 2 (filled with zeros)"
        
        plt.figure(figsize=(12, 10))
        colors = plt.cm.get_cmap('tab10', n_clusters if n_clusters > 0 else 1)
        
        # 绘制数据点
        for i, label in enumerate(unique_labels):
            if label == -1:
                # 噪声点用黑色绘制
                noise_points = data_2d[cluster_labels == label]
                plt.scatter(noise_points[:, 0], noise_points[:, 1], 
                           c='black', marker='x', label='噪声点', alpha=0.7, s=50)
            else:
                cluster_points = data_2d[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=[colors(i)], label=f'簇 {label}', alpha=0.7, s=50)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('HDBSCAN聚类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存聚类结果图
        plot_filename = f"hdbscan_clustering_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 生成簇层次结构图（如果可用）
        condensed_tree_plot_url = None
        try:
            if hasattr(clusterer, 'condensed_tree_'):
                plt.figure(figsize=(10, 6))
                clusterer.condensed_tree_.plot()
                plt.title('HDBSCAN簇层次结构')
                plt.tight_layout()
                
                condensed_tree_plot_filename = f"hdbscan_condensed_tree_{uuid.uuid4().hex}.png"
                condensed_tree_plot_filepath = os.path.join(OUTPUT_DIR, condensed_tree_plot_filename)
                plt.savefig(condensed_tree_plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                condensed_tree_plot_url = f"{PUBLIC_FILE_BASE_URL}/{condensed_tree_plot_filename}"
        except:
            pass
        
        # 生成参数说明图
        param_plot_url = None
        plt.figure(figsize=(10, 6))
        
        # 创建参数说明表格
        param_data = [
            ['参数', '值', '说明'],
            ['最小簇大小', str(min_cluster_size), '形成簇所需的最小样本数'],
            ['最小样本数', str(min_samples) if min_samples is not None else '默认', '核心点邻域中的最小样本数'],
            ['簇选择方法', cluster_selection_method, '簇选择算法'],
            ['允许单簇', str(allow_single_cluster), '是否允许将所有点归为一个簇'],
            ['Alpha参数', str(alpha), '用于计算不稳定性的参数'],
            ['距离度量', metric, '使用的距离度量方法'],
            ['数据标准化', str(standardize), '是否对数据进行标准化']
        ]
        
        table = plt.table(cellText=param_data[1:], colLabels=param_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(param_data)):
            if i == 0:  # 表头
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 2)].set_facecolor('#4CAF50')
            else:
                table[(i, 0)].set_facecolor('#E8F5E8')
                table[(i, 1)].set_facecolor('#F0F0F0')
                table[(i, 2)].set_facecolor('#F0F0F0')
        
        plt.axis('off')
        plt.title('HDBSCAN参数说明')
        plt.tight_layout()
        
        # 保存参数说明图
        param_plot_filename = f"hdbscan_parameters_{uuid.uuid4().hex}.png"
        param_plot_filepath = os.path.join(OUTPUT_DIR, param_plot_filename)
        plt.savefig(param_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        param_plot_url = f"{PUBLIC_FILE_BASE_URL}/{param_plot_filename}"
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "n_variables": n_vars,
            "variable_names": var_names,
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise_points),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
            "allow_single_cluster": allow_single_cluster,
            "alpha": alpha,
            "metric": metric,
            "standardized": standardize,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_probabilities": cluster_probabilities.tolist() if cluster_probabilities is not None else None,
            "outlier_scores": outlier_scores.tolist() if outlier_scores is not None else None,
            "cluster_stats": cluster_stats,
            "evaluation_metrics": {
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_score": ch_score
            } if silhouette_avg is not None and ch_score is not None else None,
            "cluster_plot_url": plot_url,
            "condensed_tree_plot_url": condensed_tree_plot_url,
            "parameter_plot_url": param_plot_url,
            "interpretation": _get_hdbscan_interpretation(n_clusters, n_noise_points, silhouette_avg)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"HDBSCAN聚类分析失败: {str(e)}") from e


def _get_hdbscan_interpretation(n_clusters: int, n_noise_points: int, silhouette_avg: float) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = f"HDBSCAN聚类分析完成，共识别出{n_clusters}个簇"
    
    if n_noise_points > 0:
        interpretation += f"，{n_noise_points}个噪声点"
    
    interpretation += "。"
    
    if n_clusters > 0:
        if silhouette_avg is not None:
            interpretation += f"轮廓系数为{silhouette_avg:.4f}，"
            if silhouette_avg > 0.7:
                interpretation += "聚类效果很好。"
            elif silhouette_avg > 0.5:
                interpretation += "聚类效果较好。"
            elif silhouette_avg > 0.25:
                interpretation += "聚类效果一般。"
            else:
                interpretation += "聚类效果较差。"
        else:
            interpretation += "数据中存在噪声点，聚类效果可能受到影响。"
    else:
        interpretation += "未识别出有效簇，所有点都被标记为噪声点。"
    
    return interpretation