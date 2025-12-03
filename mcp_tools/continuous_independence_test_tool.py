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


def perform_continuous_independence_test(
    data: List[float] = Field(..., description="连续变量数据，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
) -> dict:
    """
    执行连续变量相关性检验，用于检验多个连续变量之间的线性相关性
    
    参数:
    - data: 连续变量数据，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    
    返回:
    - 包含检验结果的字典
    """
    try:
        # 输入验证
        n_vars = len(var_names)
        
        if n_vars == 0:
            raise ValueError("变量不能为空")
        
        # 检查数据长度是否为变量数的整数倍
        if len(data) % n_vars != 0:
            raise ValueError("数据长度与变量数量不匹配")
        
        n_obs = len(data) // n_vars
        
        # 将一维数组还原为嵌套列表
        data_nested = []
        for i in range(n_vars):
            var_data = []
            for j in range(n_obs):
                var_data.append(data[i * n_obs + j])
            data_nested.append(var_data)
        
        # 转换为numpy数组并转置为 (n_obs, n_vars)
        data_array = np.array(data_nested).T  # shape: (n_obs, n_vars)
        
        # 创建DataFrame便于处理
        df = pd.DataFrame(data_array, columns=var_names)
        
        # 计算相关系数矩阵和p值矩阵
        corr_matrix = np.zeros((n_vars, n_vars))
        p_value_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    # 计算皮尔逊相关系数
                    corr, p_value = stats.pearsonr(df[var_names[i]], df[var_names[j]])
                    corr_matrix[i, j] = corr
                    p_value_matrix[i, j] = p_value
        
        # 计算相关系数显著性标记矩阵
        significant_matrix = p_value_matrix < 0.05
        
        # 生成相关系数热力图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制相关系数热力图
        im1 = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title('皮尔逊相关系数矩阵')
        axes[0].set_xticks(range(n_vars))
        axes[0].set_yticks(range(n_vars))
        axes[0].set_xticklabels(var_names, rotation=45)
        axes[0].set_yticklabels(var_names)
        
        # 在每个单元格中添加数值
        for i in range(n_vars):
            for j in range(n_vars):
                axes[0].text(j, i, f'{corr_matrix[i, j]:.3f}', 
                            ha="center", va="center", 
                            color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
        
        plt.colorbar(im1, ax=axes[0])
        
        # 绘制显著性标记图
        im2 = axes[1].imshow(significant_matrix, cmap='Blues', aspect='auto')
        axes[1].set_title('显著性标记 (p<0.05)')
        axes[1].set_xticks(range(n_vars))
        axes[1].set_yticks(range(n_vars))
        axes[1].set_xticklabels(var_names, rotation=45)
        axes[1].set_yticklabels(var_names)
        
        # 在每个单元格中添加显著性标记
        for i in range(n_vars):
            for j in range(n_vars):
                sig_marker = "*" if significant_matrix[i, j] else ""
                axes[1].text(j, i, sig_marker, 
                            ha="center", va="center", 
                            color="red" if significant_matrix[i, j] else "black", 
                            fontsize=20)
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        # 保存相关系数图
        plot_filename = f"continuous_independence_test_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织结果
        result = {
            "number_of_variables": n_vars,
            "number_of_observations": n_obs,
            "variable_names": var_names,
            "correlation_matrix": corr_matrix.tolist(),
            "p_value_matrix": p_value_matrix.tolist(),
            "significant_matrix": significant_matrix.tolist(),
            "correlation_plot_url": plot_url,
            "interpretation": _get_interpretation(corr_matrix, p_value_matrix, var_names)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"连续变量独立性检验计算失败: {str(e)}") from e


def _get_interpretation(corr_matrix: np.ndarray, p_value_matrix: np.ndarray, var_names: List[str]) -> str:
    """
    根据相关系数和p值提供结果解释
    """
    n_vars = len(var_names)
    interpretation = "变量间相关性分析结果：\n"
    
    # 查找最强的相关性（排除对角线）
    max_corr = 0
    max_pair = None
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            corr = abs(corr_matrix[i, j])
            if corr > max_corr:
                max_corr = corr
                max_pair = (i, j)
    
    if max_pair is not None:
        i, j = max_pair
        corr_value = corr_matrix[i, j]
        p_value = p_value_matrix[i, j]
        
        interpretation += f"变量 {var_names[i]} 和 {var_names[j]} 之间具有最强的相关性 "
        interpretation += f"(r={corr_value:.3f}, p={'<0.05' if p_value < 0.05 else f'={p_value:.3f}'})。"
        
        if p_value < 0.05:
            if abs(corr_value) >= 0.7:
                interpretation += "相关性很强"
            elif abs(corr_value) >= 0.4:
                interpretation += "相关性中等"
            else:
                interpretation += "相关性较弱"
            
            if corr_value > 0:
                interpretation += "，呈正相关关系"
            else:
                interpretation += "，呈负相关关系"
            interpretation += "。\n"
        else:
            interpretation += "但该相关性在统计学上不显著。\n"
    
    # 统计显著相关的变量对数量
    significant_pairs = 0
    total_pairs = 0
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            total_pairs += 1
            if p_value_matrix[i, j] < 0.05:
                significant_pairs += 1
    
    interpretation += f"在所有 {total_pairs} 对变量中，共有 {significant_pairs} 对变量表现出统计学上显著的相关性。"
    
    return interpretation