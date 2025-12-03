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
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_factor_analysis(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_factors: Optional[int] = Field(None, description="因子数量，如果不指定则使用默认方法确定"),
    rotation: str = Field("varimax", description="因子旋转方法: 'varimax', 'promax', None"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行因子分析
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - n_factors: 因子数量，如果不指定则使用默认方法确定
    - rotation: 因子旋转方法 ('varimax', 'promax', None)
    - standardize: 是否标准化数据
    
    返回:
    - 包含因子分析结果的字典
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
        
        # 确定因子数量
        if n_factors is None:
            # 使用特征值大于1的标准确定因子数量
            corr_matrix = np.corrcoef(data_scaled.T)
            eigenvalues, _ = np.linalg.eig(corr_matrix)
            # 只取实部，避免复数
            eigenvalues = np.real(eigenvalues)
            n_factors = np.sum(eigenvalues > 1)
            # 确保因子数量不超过变量数量和样本数量
            n_factors = min(n_factors, min(n_vars - 1, n_samples - 1))
            if n_factors < 1:
                n_factors = 1
        
        # 确保因子数量不超过变量数量和样本数量
        n_factors = min(n_factors, min(n_vars - 1, n_samples - 1))
        
        # 执行因子分析
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        factor_scores = fa.fit_transform(data_scaled)
        
        # 获取因子载荷矩阵
        factor_loadings = fa.components_.T  # shape: (n_vars, n_factors)
        
        # 计算共同度 (communalities)
        communalities = np.sum(factor_loadings**2, axis=1)
        
        # 计算特殊方差 (specific variances)
        specific_variances = 1 - communalities
        
        # 计算因子贡献度
        factor_contributions = np.sum(factor_loadings**2, axis=0)
        total_variance = np.sum(factor_contributions)
        
        # 计算每个因子解释的方差比例
        explained_variance_ratio = factor_contributions / n_vars
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 因子旋转
        rotated_loadings = factor_loadings
        rotation_matrix = None
        if rotation == "varimax":
            rotated_loadings, rotation_matrix = _varimax_rotation(factor_loadings)
        elif rotation == "promax":
            rotated_loadings, rotation_matrix = _promax_rotation(factor_loadings)
        
        # 计算旋转后的共同度
        rotated_communalities = np.sum(rotated_loadings**2, axis=1)
        
        # 计算因子得分系数
        # 因子得分系数 = 因子载荷 * 协方差矩阵的逆
        corr_inv = np.linalg.inv(np.corrcoef(data_scaled.T))
        factor_score_coefficients = corr_inv @ rotated_loadings
        
        # 生成因子载荷图
        plt.figure(figsize=(10, 8))
        # 绘制因子载荷散点图（前两个因子）
        if n_factors >= 2:
            x_loadings = rotated_loadings[:, 0]
            y_loadings = rotated_loadings[:, 1]
            for i in range(n_vars):
                plt.arrow(0, 0, x_loadings[i]*0.8, y_loadings[i]*0.8, 
                         head_width=0.05, head_length=0.05, fc='blue', ec='blue')
                plt.text(x_loadings[i], y_loadings[i], var_names[i], 
                        fontsize=10, ha='center', va='bottom')
            plt.xlabel(f'因子1 ({explained_variance_ratio[0]:.2%} 方差)')
            plt.ylabel(f'因子2 ({explained_variance_ratio[1]:.2%} 方差)')
            plt.title('因子载荷图 (Factor Loadings Plot)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            # 添加单位圆
            circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
            plt.gca().add_patch(circle)
        else:
            # 只有一个因子的情况
            x_loadings = rotated_loadings[:, 0]
            y_zeros = np.zeros(n_vars)
            for i in range(n_vars):
                plt.arrow(0, 0, x_loadings[i], y_zeros[i], 
                         head_width=0.02, head_length=0.02, fc='blue', ec='blue')
                plt.text(x_loadings[i], y_zeros[i], var_names[i], 
                        fontsize=10, ha='center', va='bottom')
            plt.xlabel(f'因子1 ({explained_variance_ratio[0]:.2%} 方差)')
            plt.ylabel('0')
            plt.title('因子载荷图 (Factor Loadings Plot)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
        
        plt.tight_layout()
        
        # 保存因子载荷图
        loadings_plot_filename = f"factor_loadings_plot_{uuid.uuid4().hex}.png"
        loadings_plot_filepath = os.path.join(OUTPUT_DIR, loadings_plot_filename)
        plt.savefig(loadings_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        loadings_plot_url = f"{PUBLIC_FILE_BASE_URL}/{loadings_plot_filename}"
        
        # 生成碎石图 (Scree Plot)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_factors + 1), explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('因子')
        plt.ylabel('解释方差比例')
        plt.title('因子分析碎石图 (Scree Plot)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, n_factors + 1))
        plt.tight_layout()
        
        # 保存碎石图
        scree_plot_filename = f"factor_scree_plot_{uuid.uuid4().hex}.png"
        scree_plot_filepath = os.path.join(OUTPUT_DIR, scree_plot_filename)
        plt.savefig(scree_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        scree_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scree_plot_filename}"
        
        # 如果有因子得分，生成因子得分图
        factor_scores_plot_url = None
        if n_factors >= 2 and factor_scores.shape[0] <= 1000:  # 限制样本数量以免图形过于拥挤
            plt.figure(figsize=(10, 8))
            plt.scatter(factor_scores[:, 0], factor_scores[:, 1], alpha=0.7, s=50)
            plt.xlabel(f'因子1 得分')
            plt.ylabel(f'因子2 得分')
            plt.title('因子得分图 (Factor Scores Plot)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存因子得分图
            scores_plot_filename = f"factor_scores_plot_{uuid.uuid4().hex}.png"
            scores_plot_filepath = os.path.join(OUTPUT_DIR, scores_plot_filename)
            plt.savefig(scores_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            factor_scores_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scores_plot_filename}"
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "n_variables": n_vars,
            "variable_names": var_names,
            "n_factors": n_factors,
            "rotation_method": rotation,
            "standardized": standardize,
            "factor_loadings": rotated_loadings.tolist(),
            "factor_scores": factor_scores.tolist() if factor_scores is not None else None,
            "communalities": communalities.tolist(),
            "rotated_communalities": rotated_communalities.tolist(),
            "specific_variances": specific_variances.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
            "total_variance_explained": float(total_variance),
            "factor_score_coefficients": factor_score_coefficients.tolist() if factor_score_coefficients is not None else None,
            "rotation_matrix": rotation_matrix.tolist() if rotation_matrix is not None else None,
            "plots": {
                "loadings_plot_url": loadings_plot_url,
                "scree_plot_url": scree_plot_url,
                "factor_scores_plot_url": factor_scores_plot_url
            },
            "interpretation": _get_interpretation(explained_variance_ratio, cumulative_variance_ratio, rotated_loadings, var_names)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"因子分析失败: {str(e)}") from e


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> tuple:
    """
    Varimax旋转实现
    
    参数:
    - loadings: 因子载荷矩阵
    - max_iter: 最大迭代次数
    - tol: 收敛容差
    
    返回:
    - rotated_loadings: 旋转后的因子载荷矩阵
    - rotation_matrix: 旋转矩阵
    """
    n_vars, n_factors = loadings.shape
    
    # 初始化旋转矩阵为单位矩阵
    rotation_matrix = np.eye(n_factors)
    
    # 进行Varimax旋转
    for _ in range(max_iter):
        # 计算旋转后的载荷
        rotated_loadings = loadings @ rotation_matrix
        
        # 计算梯度
        scaled_loadings = rotated_loadings * (rotated_loadings**2).sum(axis=0)**0.5
        gradient = loadings.T @ (rotated_loadings**3 - scaled_loadings / n_vars)
        
        # 计算更新
        u, s, vh = np.linalg.svd(gradient)
        update = u @ vh
        
        # 应用更新
        rotation_matrix = rotation_matrix @ update
        
        # 检查收敛
        if np.abs(np.diag(update) - 1).max() < tol:
            break
    
    rotated_loadings = loadings @ rotation_matrix
    return rotated_loadings, rotation_matrix


def _promax_rotation(loadings: np.ndarray, power: int = 4) -> tuple:
    """
    Promax旋转实现（简化的版本）
    
    参数:
    - loadings: 因子载荷矩阵
    - power: Promax幂参数
    
    返回:
    - rotated_loadings: 旋转后的因子载荷矩阵
    - rotation_matrix: 旋转矩阵
    """
    # 先进行Varimax旋转
    varimax_loadings, varimax_rotation = _varimax_rotation(loadings)
    
    # 计算Promax旋转
    n_vars, n_factors = varimax_loadings.shape
    
    # 对Varimax载荷进行幂变换
    abs_loadings = np.abs(varimax_loadings)**power
    normalized_loadings = abs_loadings / np.sqrt(np.sum(abs_loadings**2, axis=0))
    
    # 计算旋转矩阵
    try:
        rotation_matrix = np.linalg.inv(varimax_loadings.T @ varimax_loadings) @ varimax_loadings.T @ normalized_loadings
        # 正交化旋转矩阵
        u, _, vh = np.linalg.svd(rotation_matrix)
        rotation_matrix = u @ vh
        rotated_loadings = loadings @ rotation_matrix
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，返回Varimax结果
        rotated_loadings = varimax_loadings
        rotation_matrix = varimax_rotation
    
    return rotated_loadings, rotation_matrix


def _get_interpretation(explained_variance_ratio: np.ndarray, cumulative_variance_ratio: np.ndarray, 
                       rotated_loadings: np.ndarray, var_names: List[str]) -> str:
    """
    根据因子分析结果提供解释
    """
    n_factors = len(explained_variance_ratio)
    total_explained = cumulative_variance_ratio[-1]
    
    interpretation = f"因子分析完成，共提取{n_factors}个因子。"
    interpretation += f"前两个因子分别解释了{explained_variance_ratio[0]:.2%}和{explained_variance_ratio[1]:.2%}的方差。"
    interpretation += f"所有因子累计解释了{total_explained:.2%}的总方差。"
    
    # 根据解释方差的比例给出建议
    if total_explained >= 0.7:
        interpretation += "因子解能够较好地解释原始变量的信息。"
    elif total_explained >= 0.6:
        interpretation += "因子解基本能够解释原始变量的信息。"
    else:
        interpretation += "因子解对方差的解释程度较低，建议调整因子数量或检查数据质量。"
    
    # 解释因子含义
    interpretation += "\n\n因子含义解释：\n"
    for i in range(n_factors):
        # 找到在该因子上有高载荷的变量
        factor_loadings = rotated_loadings[:, i]
        high_loading_vars = []
        for j, loading in enumerate(factor_loadings):
            if abs(loading) > 0.5:  # 载荷绝对值大于0.5认为是高载荷
                high_loading_vars.append((var_names[j], loading))
        
        if high_loading_vars:
            interpretation += f"因子{i+1}主要与以下变量有关："
            for var_name, loading in high_loading_vars:
                interpretation += f" {var_name}({loading:.3f})"
            interpretation += "\n"
        else:
            interpretation += f"因子{i+1}没有明显的高载荷变量。\n"
    
    return interpretation