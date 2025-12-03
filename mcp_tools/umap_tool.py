import os
import uuid
from typing import List, Optional
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

import umap

def perform_umap(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_neighbors: int = Field(15, description="邻居数量，控制局部与全局结构的平衡"),
    n_components: int = Field(2, description="降维后的维度数"),
    min_dist: float = Field(0.1, description="最小距离，控制簇的紧密程度"),
    metric: str = Field("euclidean", description="距离度量方法"),
    random_state: Optional[int] = Field(42, description="随机种子，用于结果可重现"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行UMAP非线性降维分析
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - n_neighbors: 邻居数量，较小值关注局部结构，较大值关注全局结构，默认为15
    - n_components: 降维后的维度数，默认为2
    - min_dist: 最小距离，控制簇的紧密程度，较小值产生更密集的簇，默认为0.1
    - metric: 距离度量方法，默认为"euclidean"
    - random_state: 随机种子，用于结果可重现，默认为42
    - standardize: 是否标准化数据，默认为True
    
    返回:
    - 包含UMAP结果的字典
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
        
        if n_neighbors >= n_samples:
            raise ValueError("邻居数量必须小于样本数量")
        
        # 创建 DataFrame
        df = pd.DataFrame({var_names[i]: reshaped_data[i] for i in range(n_vars)})
        
        # 数据标准化
        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
        else:
            scaled_data = df.values
        
        # 执行UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        embedding = reducer.fit_transform(scaled_data)
        
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
            
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('UMAP非线性降维结果')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存散点图
            scatter_plot_filename = f"umap_scatter_plot_{uuid.uuid4().hex}.png"
            scatter_plot_filepath = os.path.join(OUTPUT_DIR, scatter_plot_filename)
            plt.savefig(scatter_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            scatter_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scatter_plot_filename}"
        
        # 如果是3D，生成3D散点图（降维到2D进行可视化）
        scatter_3d_plot_url = None
        if n_components == 3:
            plt.figure(figsize=(10, 8))
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=60)
            
            # 添加标签（如果样本数不多）
            if n_samples <= 20:
                for i in range(n_samples):
                    plt.annotate(f'P{i+1}', (embedding[i, 0], embedding[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('UMAP非线性降维结果 (3D投影到2D)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存散点图
            scatter_3d_plot_filename = f"umap_3d_scatter_plot_{uuid.uuid4().hex}.png"
            scatter_3d_plot_filepath = os.path.join(OUTPUT_DIR, scatter_3d_plot_filename)
            plt.savefig(scatter_3d_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            scatter_3d_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scatter_3d_plot_filename}"
        
        # 生成参数说明图
        param_plot_url = None
        plt.figure(figsize=(10, 6))
        
        # 创建参数说明表格
        param_data = [
            ['参数', '值', '说明'],
            ['邻居数量(n_neighbors)', str(n_neighbors), '较小值关注局部结构，较大值关注全局结构'],
            ['降维维度(n_components)', str(n_components), '降维后的维度数'],
            ['最小距离(min_dist)', str(min_dist), '控制簇的紧密程度'],
            ['距离度量(metric)', metric, '使用的距离度量方法'],
            ['标准化(standardize)', str(standardize), '是否对数据进行标准化']
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
        plt.title('UMAP参数说明')
        plt.tight_layout()
        
        # 保存参数说明图
        param_plot_filename = f"umap_parameters_{uuid.uuid4().hex}.png"
        param_plot_filepath = os.path.join(OUTPUT_DIR, param_plot_filename)
        plt.savefig(param_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        param_plot_url = f"{PUBLIC_FILE_BASE_URL}/{param_plot_filename}"
        
        # 组织结果
        result = {
            "number_of_samples": n_samples,
            "number_of_variables": n_vars,
            "variable_names": var_names,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "standardized": standardize,
            "embedding": embedding.tolist(),
            "scatter_plot_url": scatter_plot_url,
            "scatter_3d_plot_url": scatter_3d_plot_url,
            "parameter_plot_url": param_plot_url,
            "interpretation": _get_umap_interpretation(n_neighbors, min_dist, n_components)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"UMAP分析失败: {str(e)}") from e


def _get_umap_interpretation(n_neighbors: int, min_dist: float, n_components: int) -> str:
    """
    根据分析参数提供解释
    """
    interpretation = f"UMAP非线性降维分析完成，降维至{n_components}维。"
    
    # 解释参数设置
    interpretation += f"使用{n_neighbors}个邻居进行计算，"
    if n_neighbors < 10:
        interpretation += "较小的邻居数量使算法更关注数据的局部结构。"
    elif n_neighbors > 30:
        interpretation += "较大的邻居数量使算法更关注数据的全局结构。"
    else:
        interpretation += "适中的邻居数量使算法在局部和全局结构之间取得平衡。"
    
    interpretation += f"最小距离参数设置为{min_dist}，"
    if min_dist < 0.1:
        interpretation += "较小的值会产生更密集的簇。"
    elif min_dist > 0.5:
        interpretation += "较大的值会产生更分散的簇。"
    else:
        interpretation += "适中的值在簇的紧密程度上取得平衡。"
    
    return interpretation