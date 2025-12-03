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
from sklearn.preprocessing import StandardScaler

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_canonical_correlation_analysis(
    x_data: List[float] = Field(..., description="第一组变量数据，所有第一组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    y_data: List[float] = Field(..., description="第二组变量数据，所有第二组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    x_var_names: List[str] = Field(..., description="第一组变量名称列表"),
    y_var_names: List[str] = Field(..., description="第二组变量名称列表"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行典型相关分析 (Canonical Correlation Analysis)
    
    参数:
    - x_data: 第一组变量数据，所有第一组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - y_data: 第二组变量数据，所有第二组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - x_var_names: 第一组变量名称列表
    - y_var_names: 第二组变量名称列表
    - standardize: 是否标准化数据
    
    返回:
    - 包含典型相关分析结果的字典
    """
    try:
        # 输入验证
        if not x_data or not y_data:
            raise ValueError("数据不能为空")
        
        n_x_vars = len(x_var_names)
        n_y_vars = len(y_var_names)
        
        if n_x_vars == 0 or n_y_vars == 0:
            raise ValueError("变量数量不能为0")
            
        if len(x_data) % n_x_vars != 0:
            raise ValueError("第一组数据长度与变量数量不匹配")
            
        if len(y_data) % n_y_vars != 0:
            raise ValueError("第二组数据长度与变量数量不匹配")
        
        n_samples = len(x_data) // n_x_vars
        y_samples = len(y_data) // n_y_vars
        
        if n_samples != y_samples:
            raise ValueError("两组数据的样本数量不一致")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_x_data = []
        for i in range(n_x_vars):
            var_data = x_data[i * n_samples:(i + 1) * n_samples]
            reshaped_x_data.append(var_data)
            
        reshaped_y_data = []
        for i in range(n_y_vars):
            var_data = y_data[i * n_samples:(i + 1) * n_samples]
            reshaped_y_data.append(var_data)
        
        # 检查所有变量的数据长度是否一致
        for v_idx, var_data in enumerate(reshaped_x_data):
            if len(var_data) != n_samples:
                raise ValueError(f"第一组变量 {x_var_names[v_idx]} 的数据长度不一致")
                
        for v_idx, var_data in enumerate(reshaped_y_data):
            if len(var_data) != n_samples:
                raise ValueError(f"第二组变量 {y_var_names[v_idx]} 的数据长度不一致")
        
        # 转换为numpy数组并转置为 (n_samples, n_vars)
        x_array = np.array(reshaped_x_data).T  # shape: (n_samples, n_x_vars)
        y_array = np.array(reshaped_y_data).T  # shape: (n_samples, n_y_vars)
        
        # 数据标准化
        if standardize:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_scaled = x_scaler.fit_transform(x_array)
            y_scaled = y_scaler.fit_transform(y_array)
        else:
            x_scaled = x_array
            y_scaled = y_array
        
        # 使用SVD方法计算典型相关分析
        # 计算协方差矩阵
        cov_xx = np.cov(x_scaled.T)  # X变量的协方差矩阵
        cov_yy = np.cov(y_scaled.T)  # Y变量的协方差矩阵
        cov_xy = np.cov(x_scaled.T, y_scaled.T)[:n_x_vars, n_x_vars:]  # X和Y的协方差矩阵
        
        # 检查矩阵是否可逆
        try:
            cov_xx_inv = np.linalg.inv(cov_xx)
            cov_yy_inv = np.linalg.inv(cov_yy)
        except np.linalg.LinAlgError:
            raise ValueError("协方差矩阵奇异，无法求逆")
        
        # 使用SVD方法计算典型相关分析
        # 计算矩阵: cov_xx^(-1/2) * cov_xy * cov_yy^(-1/2)
        # 首先计算协方差矩阵的平方根逆矩阵
        u_x, s_x, vh_x = np.linalg.svd(cov_xx)
        cov_xx_inv_sqrt = u_x @ np.diag(1.0 / np.sqrt(s_x)) @ vh_x
        
        u_y, s_y, vh_y = np.linalg.svd(cov_yy)
        cov_yy_inv_sqrt = u_y @ np.diag(1.0 / np.sqrt(s_y)) @ vh_y
        
        # 构造用于SVD的矩阵
        svd_matrix = cov_xx_inv_sqrt @ cov_xy @ cov_yy_inv_sqrt
        
        # 对构造的矩阵进行SVD分解
        U, singular_values, Vt = np.linalg.svd(svd_matrix, full_matrices=False)
        
        # 典型相关系数就是奇异值
        canonical_correlations = singular_values
        
        # 确定典型变量对数 (最多为 min(n_x_vars, n_y_vars))
        n_pairs = min(n_x_vars, n_y_vars, len(canonical_correlations))
        canonical_correlations = canonical_correlations[:n_pairs]
        
        # 计算典型权重
        # canonical_weights_x = cov_xx^(-1/2) * U
        # canonical_weights_y = cov_yy^(-1/2) * V
        canonical_weights_x = cov_xx_inv_sqrt @ U[:, :n_pairs]
        canonical_weights_y = cov_yy_inv_sqrt @ Vt.T[:, :n_pairs]
        
        # 计算典型变量得分
        canonical_scores_x = x_scaled @ canonical_weights_x
        canonical_scores_y = y_scaled @ canonical_weights_y
        
        # 计算典型载荷 (相关系数)
        # 典型载荷是原始变量与典型变量之间的相关系数
        canonical_loadings_x = np.zeros((n_x_vars, n_pairs))
        canonical_loadings_y = np.zeros((n_y_vars, n_pairs))
        
        for i in range(n_pairs):
            for j in range(n_x_vars):
                canonical_loadings_x[j, i] = np.corrcoef(x_scaled[:, j], canonical_scores_x[:, i])[0, 1]
            for j in range(n_y_vars):
                canonical_loadings_y[j, i] = np.corrcoef(y_scaled[:, j], canonical_scores_y[:, i])[0, 1]
        
        # 计算冗余分析指标
        # 冗余指数表示一个变量组能被另一个变量组的典型变量解释的方差比例
        x_redundancy = np.zeros(n_pairs)
        y_redundancy = np.zeros(n_pairs)
        
        # X变量组的平均方差由Y的典型变量解释的比例
        for i in range(n_pairs):
            # 计算第i个典型变量对X变量组的解释方差
            x_explained_var = np.sum(canonical_loadings_x[:, i]**2) / n_x_vars
            x_redundancy[i] = canonical_correlations[i]**2 * x_explained_var
        
        # Y变量组的平均方差由X的典型变量解释的比例
        for i in range(n_pairs):
            # 计算第i个典型变量对Y变量组的解释方差
            y_explained_var = np.sum(canonical_loadings_y[:, i]**2) / n_y_vars
            y_redundancy[i] = canonical_correlations[i]**2 * y_explained_var
        
        # 计算Wilks' Lambda统计量及其显著性检验
        wilks_lambda_stats = []
        for i in range(n_pairs):
            # 计算从第i个典型变量对开始的所有典型相关系数的Wilks' Lambda
            remaining_corrs = canonical_correlations[i:]
            wilks_lambda = np.prod(1 - remaining_corrs**2)
            
            # 计算自由度
            p = n_x_vars - i  # 剩余的X变量数
            q = n_y_vars - i  # 剩余的Y变量数
            n = n_samples - i - 1  # 有效样本数
            
            # 计算卡方统计量
            df = p * q  # 自由度
            if wilks_lambda > 0:
                chi2_stat = -((n - (p + q + 1) / 2) * np.log(wilks_lambda))
                p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            else:
                chi2_stat = float('inf')
                p_value = 0.0
            
            wilks_lambda_stats.append({
                "pair": i + 1,
                "wilks_lambda": float(wilks_lambda),
                "chi2_statistic": float(chi2_stat),
                "degrees_of_freedom": df,
                "p_value": float(p_value)
            })
        
        # 生成典型相关图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, n_pairs + 1), canonical_correlations, color='skyblue')
        plt.xlabel('典型变量对')
        plt.ylabel('典型相关系数')
        plt.title('典型相关系数')
        plt.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (bar, corr) in enumerate(zip(bars, canonical_correlations)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存典型相关图
        correlation_plot_filename = f"canonical_correlations_plot_{uuid.uuid4().hex}.png"
        correlation_plot_filepath = os.path.join(OUTPUT_DIR, correlation_plot_filename)
        plt.savefig(correlation_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        correlation_plot_url = f"{PUBLIC_FILE_BASE_URL}/{correlation_plot_filename}"
        
        # 生成典型载荷图 (前两个典型变量对)
        if n_pairs >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 第一对典型变量的载荷
            # 为了防止维度不一致，分别绘制散点图
            x_loadings_1 = canonical_loadings_x[:, 0]
            y_loadings_1 = canonical_loadings_y[:, 0]
            
            # 绘制X变量载荷（使用不同标记）
            ax1.scatter(x_loadings_1, [0] * len(x_loadings_1), 
                       s=100, alpha=0.7, c='blue', marker='o', label='X变量')
            # 绘制Y变量载荷（使用不同标记）
            ax1.scatter([0] * len(y_loadings_1), y_loadings_1, 
                       s=100, alpha=0.7, c='red', marker='s', label='Y变量')
            ax1.set_xlabel(f'X变量载荷 (典型变量1)')
            ax1.set_ylabel(f'Y变量载荷 (典型变量1)')
            ax1.set_title('第一对典型变量载荷图')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 添加变量标签
            for i, label in enumerate(x_var_names):
                idx = min(i, len(x_loadings_1) - 1)
                ax1.annotate(label, (x_loadings_1[idx], 0), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')
            
            for i, label in enumerate(y_var_names):
                idx = min(i, len(y_loadings_1) - 1)
                ax1.annotate(label, (0, y_loadings_1[idx]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
            
            # 第二对典型变量的载荷
            x_loadings_2 = canonical_loadings_x[:, 1] if n_pairs > 1 else canonical_loadings_x[:, 0]
            y_loadings_2 = canonical_loadings_y[:, 1] if n_pairs > 1 else canonical_loadings_y[:, 0]
            
            # 绘制X变量载荷
            ax2.scatter(x_loadings_2, [0] * len(x_loadings_2), 
                       s=100, alpha=0.7, c='blue', marker='o', label='X变量')
            # 绘制Y变量载荷
            ax2.scatter([0] * len(y_loadings_2), y_loadings_2, 
                       s=100, alpha=0.7, c='red', marker='s', label='Y变量')
            ax2.set_xlabel(f'X变量载荷 (典型变量2)')
            ax2.set_ylabel(f'Y变量载荷 (典型变量2)')
            ax2.set_title('第二对典型变量载荷图')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 添加变量标签
            for i, label in enumerate(x_var_names):
                idx = min(i, len(x_loadings_2) - 1)
                ax2.annotate(label, (x_loadings_2[idx], 0), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')
            
            for i, label in enumerate(y_var_names):
                idx = min(i, len(y_loadings_2) - 1)
                ax2.annotate(label, (0, y_loadings_2[idx]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
            
            plt.tight_layout()
            
            # 保存典型载荷图
            loadings_plot_filename = f"canonical_loadings_plot_{uuid.uuid4().hex}.png"
            loadings_plot_filepath = os.path.join(OUTPUT_DIR, loadings_plot_filename)
            plt.savefig(loadings_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            loadings_plot_url = f"{PUBLIC_FILE_BASE_URL}/{loadings_plot_filename}"
        else:
            loadings_plot_url = None
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "n_x_variables": n_x_vars,
            "n_y_variables": n_y_vars,
            "x_variable_names": x_var_names,
            "y_variable_names": y_var_names,
            "standardized": standardize,
            "n_pairs": n_pairs,
            "canonical_correlations": canonical_correlations.tolist(),
            "canonical_weights_x": canonical_weights_x.tolist(),
            "canonical_weights_y": canonical_weights_y.tolist(),
            "canonical_loadings_x": canonical_loadings_x.tolist(),
            "canonical_loadings_y": canonical_loadings_y.tolist(),
            "canonical_scores_x": canonical_scores_x.tolist(),
            "canonical_scores_y": canonical_scores_y.tolist(),
            "x_redundancy": x_redundancy.tolist(),
            "y_redundancy": y_redundancy.tolist(),
            "wilks_lambda_test": wilks_lambda_stats,
            "plots": {
                "correlation_plot_url": correlation_plot_url,
                "loadings_plot_url": loadings_plot_url
            },
            "interpretation": _get_interpretation(canonical_correlations, canonical_loadings_x, canonical_loadings_y, 
                                                x_var_names, y_var_names, x_redundancy, y_redundancy)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"典型相关分析失败: {str(e)}") from e


def _get_interpretation(correlations: np.ndarray, loadings_x: np.ndarray, loadings_y: np.ndarray,
                       x_var_names: List[str], y_var_names: List[str],
                       x_redundancy: np.ndarray, y_redundancy: np.ndarray) -> str:
    """
    根据典型相关分析结果提供解释
    """
    n_pairs = len(correlations)
    
    interpretation = f"典型相关分析完成，共得到{n_pairs}对典型变量。"
    
    # 解释每对典型变量的相关性
    for i in range(min(3, n_pairs)):  # 最多解释前3对
        interpretation += f"\n第{i+1}对典型变量的相关系数为{correlations[i]:.3f}。"
        
        # 找到在该对典型变量上有高载荷的变量
        x_high_loadings = []
        y_high_loadings = []
        
        for j, loading in enumerate(loadings_x[:, i]):
            if abs(loading) > 0.5:  # 载荷绝对值大于0.5认为是高载荷
                x_high_loadings.append((x_var_names[j], loading))
        
        for j, loading in enumerate(loadings_y[:, i]):
            if abs(loading) > 0.5:  # 载荷绝对值大于0.5认为是高载荷
                y_high_loadings.append((y_var_names[j], loading))
        
        if x_high_loadings or y_high_loadings:
            interpretation += " 在这一对典型变量中:"
            if x_high_loadings:
                interpretation += " 第一组变量中"
                for var_name, loading in x_high_loadings:
                    interpretation += f" {var_name}({loading:.2f})"
            if y_high_loadings:
                interpretation += " 第二组变量中"
                for var_name, loading in y_high_loadings:
                    interpretation += f" {var_name}({loading:.2f})"
    
    # 解释冗余分析
    if len(x_redundancy) > 0 and len(y_redundancy) > 0:
        total_x_redundancy = np.sum(x_redundancy)
        total_y_redundancy = np.sum(y_redundancy)
        interpretation += f"\n\n冗余分析显示，第一组变量的平均方差中有{total_x_redundancy:.1%}可以被第二组变量的典型变量解释，"
        interpretation += f"第二组变量的平均方差中有{total_y_redundancy:.1%}可以被第一组变量的典型变量解释。"
        
        if total_x_redundancy > 0.3 and total_y_redundancy > 0.3:
            interpretation += "这表明两组变量之间存在较强的相互解释能力。"
        elif total_x_redundancy > 0.1 and total_y_redundancy > 0.1:
            interpretation += "这表明两组变量之间存在一定的相互解释能力。"
        else:
            interpretation += "这表明两组变量之间的相互解释能力较弱。"
    
    return interpretation