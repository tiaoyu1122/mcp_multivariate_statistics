import os
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pydantic import Field
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_hierarchical_clustering(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_clusters: int = Field(3, description="聚类数量"),
    linkage_method: str = Field("ward", description="链接方法: 'ward', 'complete', 'average', 'single'"),
) -> dict:
    """
    执行凝聚式层次聚类分析
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - n_clusters: 聚类数量
    - linkage_method: 链接方法 ('ward', 'complete', 'average', 'single')
    
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
        
        # 将一维数组还原为嵌套列表结构
        reshaped_data = []
        for i in range(n_vars):
            var_data = data[i * n_samples:(i + 1) * n_samples]
            reshaped_data.append(var_data)
        
        # 检查所有变量的数据长度是否一致
        for v_idx, var_data in enumerate(reshaped_data):
            if len(var_data) != n_samples:
                raise ValueError(f"变量 {var_names[v_idx]} 的数据长度不一致")
        
        # 转换为numpy数组并转置为 (n_samples, n_vars)
        data_array = np.array(reshaped_data).T  # shape: (n_samples, n_vars)
        
        # 执行层次聚类
        # 计算链接矩阵
        if linkage_method == "ward" and n_vars < n_samples:
            # Ward方法要求使用欧氏距离
            linkage_matrix = linkage(data_array, method=linkage_method)
        else:
            # 其他方法可以使用预计算距离矩阵
            linkage_matrix = linkage(data_array, method=linkage_method)
        
        # 根据指定的簇数获取聚类标签
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        # 调整标签从0开始
        cluster_labels = cluster_labels - 1
        
        # 计算评估指标
        try:
            silhouette_avg = silhouette_score(data_array, cluster_labels)
        except:
            silhouette_avg = None
        
        try:
            ch_score = calinski_harabasz_score(data_array, cluster_labels)
        except:
            ch_score = None
        
        # 计算每个簇的统计信息
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = data_array[cluster_labels == i]
            cluster_size = cluster_data.shape[0]
            
            # 计算每个簇的均值和标准差
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            cluster_stats.append({
                "cluster_id": i,
                "size": int(cluster_size),
                "mean": cluster_mean.tolist(),
                "std": cluster_std.tolist()
            })
        
        # 生成聚类结果可视化图
        # 如果变量数大于2，使用PCA降维到2D进行可视化
        if n_vars > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data_array)
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
        else:
            data_2d = data_array[:, :2] if n_vars >= 2 else np.hstack([data_array, np.zeros((n_samples, 1))])
            xlabel = var_names[0] if n_vars >= 1 else "Variable 1"
            ylabel = var_names[1] if n_vars >= 2 else "Variable 2 (filled with zeros)"
        
        # 绘制聚类结果图
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', n_clusters)
        
        # 绘制数据点
        for i in range(n_clusters):
            cluster_points = data_2d[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors(i)], label=f'簇 {i}', alpha=0.7, s=50)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('层次聚类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存聚类结果图
        plot_filename = f"hierarchical_clustering_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url_result = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 生成树状图
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title('层次聚类树状图')
        plt.xlabel('样本索引或簇大小')
        plt.ylabel('距离')
        plt.tight_layout()
        
        # 保存树状图
        dendrogram_filename = f"hierarchical_dendrogram_{uuid.uuid4().hex}.png"
        dendrogram_filepath = os.path.join(OUTPUT_DIR, dendrogram_filename)
        plt.savefig(dendrogram_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        dendrogram_url = f"{PUBLIC_FILE_BASE_URL}/{dendrogram_filename}"
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "n_variables": n_vars,
            "variable_names": var_names,
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_stats": cluster_stats,
            "evaluation_metrics": {
                "silhouette_score": float(silhouette_avg) if silhouette_avg is not None else None,
                "calinski_harabasz_score": float(ch_score) if ch_score is not None else None,
            },
            "parameters": {
                "linkage_method": linkage_method
            },
            "plot_url": plot_url_result,
            "dendrogram_url": dendrogram_url,
            "interpretation": _get_interpretation(silhouette_avg, ch_score)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"层次聚类失败: {str(e)}") from e


def _get_interpretation(silhouette_score: float, ch_score: float) -> str:
    """
    根据评估指标提供结果解释
    """
    interpretation = "聚类分析完成。"
    
    if silhouette_score is not None:
        if silhouette_score > 0.7:
            interpretation += "轮廓系数较高，表明聚类结果良好。"
        elif silhouette_score > 0.5:
            interpretation += "轮廓系数中等，表明聚类结果合理。"
        elif silhouette_score > 0.25:
            interpretation += "轮廓系数较低，表明聚类结果一般。"
        else:
            interpretation += "轮廓系数很低，表明聚类结果较差。"
    
    if ch_score is not None:
        if ch_score > 1000:
            interpretation += " Calinski-Harabasz指数很高，表明簇间分离度很好。"
        elif ch_score > 100:
            interpretation += " Calinski-Harabasz指数较高，表明簇间分离度较好。"
        elif ch_score > 10:
            interpretation += " Calinski-Harabasz指数适中，表明簇间分离度一般。"
        else:
            interpretation += " Calinski-Harabasz指数较低，表明簇间分离度较差。"
    
    return interpretation