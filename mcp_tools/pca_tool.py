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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_pca(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_components: Optional[int] = Field(None, description="主成分数量，如果不指定则保留所有成分"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行主成分分析(PCA)
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - n_components: 主成分数量，如果不指定则保留所有成分
    - standardize: 是否标准化数据
    
    返回:
    - 包含PCA结果的字典
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
        
        # 数据标准化
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
        else:
            data_scaled = data_array
        
        # 确定主成分数量
        if n_components is None:
            n_components = min(n_samples, n_vars)
        else:
            n_components = min(n_components, min(n_samples, n_vars))
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)
        
        # 获取主成分的属性
        components = pca.components_  # 主成分载荷矩阵
        explained_variance = pca.explained_variance_  # 解释的方差
        explained_variance_ratio = pca.explained_variance_ratio_  # 解释的方差比例
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)  # 累积解释方差比例
        
        # 计算每个原始变量对主成分的贡献
        variable_contributions = []
        for i in range(n_vars):
            contributions = []
            for j in range(n_components):
                contribution = (components[j, i] ** 2) * explained_variance[j]
                contributions.append(float(contribution))
            variable_contributions.append({
                "variable": var_names[i],
                "contributions": contributions,
                "total_contribution": float(sum(contributions))
            })
        
        # 计算每个主成分的得分统计信息
        pc_stats = []
        for i in range(n_components):
            pc_scores = pca_result[:, i]
            pc_stats.append({
                "pc": i + 1,
                "explained_variance": float(explained_variance[i]),
                "explained_variance_ratio": float(explained_variance_ratio[i]),
                "cumulative_variance_ratio": float(cumulative_variance_ratio[i]),
                "mean": float(np.mean(pc_scores)),
                "std": float(np.std(pc_scores)),
                "min": float(np.min(pc_scores)),
                "max": float(np.max(pc_scores))
            })
        
        # 生成碎石图 (Scree Plot)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_components + 1), explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('主成分')
        plt.ylabel('解释方差比例')
        plt.title('碎石图 (Scree Plot)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, n_components + 1))
        plt.tight_layout()
        
        # 保存碎石图
        scree_plot_filename = f"pca_scree_plot_{uuid.uuid4().hex}.png"
        scree_plot_filepath = os.path.join(OUTPUT_DIR, scree_plot_filename)
        plt.savefig(scree_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        scree_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scree_plot_filename}"
        
        # 生成累积方差解释比例图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比例')
        plt.title('累积解释方差比例图')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, n_components + 1))
        plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80%阈值')
        plt.axhline(y=0.9, color='b', linestyle='--', alpha=0.7, label='90%阈值')
        plt.legend()
        plt.tight_layout()
        
        # 保存累积方差解释比例图
        cumulative_plot_filename = f"pca_cumulative_variance_{uuid.uuid4().hex}.png"
        cumulative_plot_filepath = os.path.join(OUTPUT_DIR, cumulative_plot_filename)
        plt.savefig(cumulative_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        cumulative_plot_url = f"{PUBLIC_FILE_BASE_URL}/{cumulative_plot_filename}"
        
        # 如果主成分数量 >= 2，生成前两个主成分的散点图
        biplot_url = None
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=50)
            plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} 方差)')
            plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} 方差)')
            plt.title('主成分得分散点图 (PC1 vs PC2)')
            plt.grid(True, alpha=0.3)
            
            # 添加变量载荷箭头（如果变量数 <= 10）
            if n_vars <= 10:
                # 缩放载荷以便更好地显示
                scale_factor = 3  # 调整箭头长度的缩放因子
                for i in range(n_vars):
                    plt.arrow(0, 0, 
                             components[0, i] * scale_factor, 
                             components[1, i] * scale_factor,
                             head_width=0.05, head_length=0.05, fc='red', ec='red')
                    plt.text(components[0, i] * scale_factor * 1.1, 
                            components[1, i] * scale_factor * 1.1,
                            var_names[i], fontsize=10, ha='center', va='center')
            
            plt.tight_layout()
            
            # 保存双标图
            biplot_filename = f"pca_biplot_{uuid.uuid4().hex}.png"
            biplot_filepath = os.path.join(OUTPUT_DIR, biplot_filename)
            plt.savefig(biplot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            biplot_url = f"{PUBLIC_FILE_BASE_URL}/{biplot_filename}"
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "n_variables": n_vars,
            "variable_names": var_names,
            "n_components": n_components,
            "standardized": standardize,
            "principal_components": pca_result.tolist(),  # 主成分得分
            "components": components.tolist(),  # 主成分载荷矩阵
            "explained_variance": explained_variance.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
            "pc_stats": pc_stats,
            "variable_contributions": variable_contributions,
            "plots": {
                "scree_plot_url": scree_plot_url,
                "cumulative_variance_plot_url": cumulative_plot_url,
                "biplot_url": biplot_url
            },
            "interpretation": _get_interpretation(explained_variance_ratio, cumulative_variance_ratio)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"主成分分析失败: {str(e)}") from e


def _get_interpretation(explained_variance_ratio: np.ndarray, cumulative_variance_ratio: np.ndarray) -> str:
    """
    根据解释方差比例提供结果解释
    """
    n_components = len(explained_variance_ratio)
    total_explained = cumulative_variance_ratio[-1]
    
    interpretation = f"主成分分析完成，共提取{n_components}个主成分。"
    interpretation += f"前两个主成分分别解释了{explained_variance_ratio[0]:.2%}和{explained_variance_ratio[1]:.2%}的方差。"
    interpretation += f"所有主成分累计解释了{total_explained:.2%}的总方差。"
    
    # 根据解释方差的比例给出建议
    if total_explained >= 0.9:
        interpretation += "前几个主成分能够很好地概括原始数据的信息。"
    elif total_explained >= 0.8:
        interpretation += "前几个主成分较好地概括了原始数据的主要信息。"
    elif total_explained >= 0.7:
        interpretation += "前几个主成分基本概括了原始数据的信息，但可能需要考虑更多的主成分。"
    else:
        interpretation += "前几个主成分对方差的解释程度较低，建议考虑其他降维方法或检查数据质量。"
    
    return interpretation