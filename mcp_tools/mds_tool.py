import os
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform, pdist
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pydantic import Field
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_mds(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_components: int = Field(2, description="降维后的维度数"),
    metric: bool = Field(True, description="是否使用度量MDS，True表示经典MDS，False表示非度量MDS"),
    n_init: int = Field(4, description="初始化次数，用于寻找最佳解"),
    max_iter: int = Field(300, description="最大迭代次数"),
    dissimilarity: str = Field("euclidean", description="距离度量方法: 'euclidean'(欧氏距离), 'precomputed'(预先计算的距离矩阵)"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行多维尺度分析(MDS)
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - n_components: 降维后的维度数，默认为2
    - metric: 是否使用度量MDS，True表示经典MDS，False表示非度量MDS，默认为True
    - n_init: 初始化次数，用于寻找最佳解，默认为4
    - max_iter: 最大迭代次数，默认为300
    - dissimilarity: 距离度量方法: 'euclidean'(欧氏距离), 'precomputed'(预先计算的距离矩阵)
    - standardize: 是否标准化数据，默认为True
    
    返回:
    - 包含MDS结果的字典
    """
    try:
        # 输入验证
        if not data:
            raise ValueError("数据不能为空")
        
        n_vars = len(var_names)
        if n_vars == 0:
            raise ValueError("变量数量不能为0")
            
        # 根据var_names中元素的个数计算数据长度
        if len(data) % n_vars != 0:
            raise ValueError("数据长度与变量数量不匹配")
            
        n_samples = len(data) // n_vars
        if n_samples == 0:
            raise ValueError("样本数量不能为0")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_data = []
        for i in range(n_vars):
            var_data = data[i * n_samples:(i + 1) * n_samples]
            reshaped_data.append(var_data)
        
        if n_components > n_samples:
            raise ValueError("降维后的维度数不能大于样本数量")
        
        if dissimilarity not in ["euclidean", "precomputed"]:
            raise ValueError("dissimilarity 参数必须是 'euclidean' 或 'precomputed'")
        
        # 创建 DataFrame
        df = pd.DataFrame({var_names[i]: reshaped_data[i] for i in range(n_vars)})
        
        # 数据标准化
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
        else:
            scaled_data = df.values
        
        # 计算距离矩阵
        if dissimilarity == "euclidean":
            dissimilarity_matrix = pairwise_distances(scaled_data, metric='euclidean')
        else:
            # 如果 dissimilarity 是 "precomputed"，则假定输入数据就是距离矩阵
            if n_vars != n_samples:
                raise ValueError("当 dissimilarity='precomputed' 时，输入数据必须是方形距离矩阵")
            dissimilarity_matrix = scaled_data
        
        # 执行MDS
        mds = MDS(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            dissimilarity="precomputed",
            random_state=42
        )
        
        embedding = mds.fit_transform(dissimilarity_matrix)
        
        # 计算应力值 (stress)
        stress = mds.stress_
        
        # 如果是2D，生成散点图
        scatter_plot_url = None
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=60)
            
            # 添加标签（如果样本数不多）
            if n_samples <= 20:
                for i in range(n_samples):
                    plt.annotate(f'P{i+1}', (embedding[i, 0], embedding[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('第一维度')
            plt.ylabel('第二维度')
            plt.title('多维尺度分析(MDS)结果')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存散点图
            scatter_plot_filename = f"mds_scatter_plot_{uuid.uuid4().hex}.png"
            scatter_plot_filepath = os.path.join(OUTPUT_DIR, scatter_plot_filename)
            plt.savefig(scatter_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            scatter_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scatter_plot_filename}"
        
        # 生成应力值图（如果有多个初始化）
        stress_plot_url = None
        if n_init > 1:
            # 重新运行多次MDS以获得不同初始化的应力值
            stress_values = []
            for i in range(min(10, n_init)):  # 最多10次
                mds_temp = MDS(
                    n_components=n_components,
                    metric=metric,
                    n_init=1,
                    max_iter=max_iter,
                    dissimilarity="precomputed",
                    random_state=i
                )
                mds_temp.fit(dissimilarity_matrix)
                stress_values.append(mds_temp.stress_)
            
            if len(stress_values) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(stress_values) + 1), stress_values, 'bo-')
                plt.xlabel('初始化次数')
                plt.ylabel('应力值')
                plt.title('不同初始化的应力值')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # 保存应力值图
                stress_plot_filename = f"mds_stress_plot_{uuid.uuid4().hex}.png"
                stress_plot_filepath = os.path.join(OUTPUT_DIR, stress_plot_filename)
                plt.savefig(stress_plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                stress_plot_url = f"{PUBLIC_FILE_BASE_URL}/{stress_plot_filename}"
        
        # 组织结果
        result = {
            "number_of_samples": n_samples,
            "number_of_variables": n_vars,
            "variable_names": var_names,
            "n_components": n_components,
            "metric_mds": metric,
            "standardized": standardize,
            "dissimilarity_measure": dissimilarity,
            "stress": stress,
            "embedding": embedding.tolist(),
            "scatter_plot_url": scatter_plot_url,
            "stress_plot_url": stress_plot_url,
            "interpretation": _get_mds_interpretation(stress, n_components)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"MDS分析失败: {str(e)}") from e


def _get_mds_interpretation(stress: float, n_components: int) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = f"多维尺度分析(MDS)完成，降维至{n_components}维。"
    
    # 解释应力值
    interpretation += f"模型的应力值为{stress:.4f}。"
    
    if stress < 0.05:
        interpretation += "应力值非常小，表明低维表示能够很好地还原高维数据结构。"
    elif stress < 0.1:
        interpretation += "应力值较小，表明低维表示能够较好地还原高维数据结构。"
    elif stress < 0.2:
        interpretation += "应力值适中，表明低维表示能够大致还原高维数据结构。"
    else:
        interpretation += "应力值较大，表明低维表示可能无法很好地还原高维数据结构，建议增加维度数或检查数据。"
    
    return interpretation