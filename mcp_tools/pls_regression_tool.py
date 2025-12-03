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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_pls_regression(
    x_data: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    y_data: List[float] = Field(..., description="因变量数据，所有因变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    x_var_names: List[str] = Field(..., description="自变量名称列表"),
    y_var_names: List[str] = Field(..., description="因变量名称列表"),
    n_components: Optional[int] = Field(None, description="成分数量，如果不指定则自动选择"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行偏最小二乘回归分析 (PLS Regression)
    
    参数:
    - x_data: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - y_data: 因变量数据，所有因变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - x_var_names: 自变量名称列表
    - y_var_names: 因变量名称列表
    - n_components: 成分数量，如果不指定则自动选择
    - standardize: 是否标准化数据
    
    返回:
    - 包含PLS回归分析结果的字典
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
            raise ValueError("自变量数据长度与变量数量不匹配")
            
        if len(y_data) % n_y_vars != 0:
            raise ValueError("因变量数据长度与变量数量不匹配")
        
        n_samples = len(x_data) // n_x_vars
        y_samples = len(y_data) // n_y_vars
        
        if n_samples != y_samples:
            raise ValueError("自变量和因变量的样本数量不一致")
        
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
                raise ValueError(f"自变量 {x_var_names[v_idx]} 的数据长度不一致")
                
        for v_idx, var_data in enumerate(reshaped_y_data):
            if len(var_data) != n_samples:
                raise ValueError(f"因变量 {y_var_names[v_idx]} 的数据长度不一致")
        
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
        
        # 确定成分数量
        if n_components is None:
            # 默认成分数量为 min(n_samples-1, n_x_vars, n_y_vars)
            n_components = min(n_samples - 1, n_x_vars, n_y_vars)
        else:
            n_components = min(n_components, min(n_samples - 1, n_x_vars, n_y_vars))
        
        # 执行PLS回归
        pls = PLSRegression(n_components=n_components)
        pls.fit(x_scaled, y_scaled)
        
        # 获取PLS结果
        # 回归系数
        coefficients = pls.coef_
        
        # X载荷和Y载荷
        x_loadings = pls.x_loadings_
        y_loadings = pls.y_loadings_
        
        # X权重和Y权重
        x_weights = pls.x_weights_
        y_weights = pls.y_weights_
        
        # X得分和Y得分
        x_scores = pls.x_scores_
        y_scores = pls.y_scores_
        
        # 计算预测值和残差
        y_pred = pls.predict(x_scaled)
        residuals = y_scaled - y_pred
        
        # 计算R²和Q²
        ss_res = np.sum((y_scaled - y_pred) ** 2)
        ss_tot = np.sum((y_scaled - np.mean(y_scaled, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 交叉验证计算Q²
        try:
            q2 = cross_val_score(pls, x_scaled, y_scaled, cv=min(10, n_samples//5), scoring='r2').mean()
        except:
            q2 = None
        
        # 计算变量重要性 (VIP)
        vip_scores = _calculate_vip_scores(x_scores, x_weights, y_loadings, n_x_vars)
        
        # 计算X变量对每个成分的贡献
        x_var_contributions = np.zeros((n_x_vars, n_components))
        for i in range(n_components):
            x_var_contributions[:, i] = x_weights[:, i] * np.sqrt(np.sum(x_scores[:, i]**2))
        
        # 计算每个成分解释的方差比例
        x_explained_variance = np.zeros(n_components)
        y_explained_variance = np.zeros(n_components)
        for i in range(n_components):
            x_explained_variance[i] = np.var(x_scores[:, i]) / np.var(x_scaled)
            y_explained_variance[i] = np.var(y_scores[:, i]) / np.var(y_scaled)
        
        # 累积解释方差
        x_cumulative_variance = np.cumsum(x_explained_variance)
        y_cumulative_variance = np.cumsum(y_explained_variance)
        
        # 残差诊断和模型假设检验
        # 由于PLS是多因变量模型，我们对所有因变量的残差进行综合检验
        # 1. 残差正态性检验 (Shapiro-Wilk 和 K-S 检验)
        # 合并所有因变量的残差进行检验
        combined_residuals = residuals.flatten()
        
        # Shapiro-Wilk 检验 (适用于小样本)
        if len(combined_residuals) <= 5000:  # Shapiro-Wilk 检验适用于样本量小于5000的情况
            sw_statistic, sw_p_value = stats.shapiro(combined_residuals)
            sw_test = {
                "test": "Shapiro-Wilk",
                "statistic": float(sw_statistic),
                "p_value": float(sw_p_value),
                "is_normal": sw_p_value > 0.05
            }
        else:
            sw_test = None
        
        # K-S 检验 (适用于大样本)
        ks_statistic, ks_p_value = stats.kstest(combined_residuals, 'norm')
        ks_test = {
            "test": "Kolmogorov-Smirnov",
            "statistic": float(ks_statistic),
            "p_value": float(ks_p_value),
            "is_normal": ks_p_value > 0.05
        }
        
        # 2. 异方差检验 (Breusch-Pagan 检验)
        # 对于多因变量，我们使用所有因变量残差的平方和进行检验
        residual_squares = np.sum(residuals**2, axis=1)
        bp_statistic, bp_p_value = _breusch_pagan_test(x_scaled, residual_squares)
        bp_test = {
            "test": "Breusch-Pagan",
            "statistic": float(bp_statistic),
            "p_value": float(bp_p_value),
            "has_heteroscedasticity": bp_p_value < 0.05
        }
        
        # 3. 自相关检验 (Durbin-Watson 检验)
        # 对于多因变量，我们使用所有因变量残差的平方和进行检验
        dw_statistic = _durbin_watson_test(residual_squares)
        dw_test = {
            "test": "Durbin-Watson",
            "statistic": float(dw_statistic),
            "interpretation": _interpret_dw_statistic(dw_statistic)
        }
        
        # 生成成分解释方差图
        plt.figure(figsize=(18, 12))
        
        # X变量解释方差
        plt.subplot(2, 3, 1)
        plt.bar(range(1, n_components + 1), x_explained_variance, alpha=0.7, color='blue', label='X变量')
        plt.plot(range(1, n_components + 1), x_cumulative_variance, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('成分')
        plt.ylabel('解释方差比例')
        plt.title('X变量解释方差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Y变量解释方差
        plt.subplot(2, 3, 2)
        plt.bar(range(1, n_components + 1), y_explained_variance, alpha=0.7, color='green', label='Y变量')
        plt.plot(range(1, n_components + 1), y_cumulative_variance, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('成分')
        plt.ylabel('解释方差比例')
        plt.title('Y变量解释方差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 生成变量重要性(VIP)图
        plt.subplot(2, 3, 3)
        sorted_indices = np.argsort(vip_scores)[::-1]
        top_indices = sorted_indices[:min(15, n_x_vars)]  # 最多显示前15个变量
        
        bars = plt.bar(range(len(top_indices)), vip_scores[top_indices], color='skyblue')
        plt.xlabel('变量')
        plt.ylabel('VIP得分')
        plt.title('变量重要性(VIP)得分 (前15个最重要变量)')
        plt.xticks(range(len(top_indices)), [x_var_names[i] for i in top_indices], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 添加阈值线 (VIP > 1 被认为是重要的)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='重要性阈值(VIP=1)')
        plt.legend()
        
        # 生成前两个成分的得分图
        if n_components >= 2:
            plt.subplot(2, 3, 4)
            plt.scatter(x_scores[:, 0], x_scores[:, 1], alpha=0.7, s=50, c='blue', label='X样本得分')
            
            # 添加部分变量向量（避免图形过于拥挤）
            n_vectors = min(10, n_x_vars)
            top_var_indices = np.argsort(vip_scores)[-n_vectors:]
            
            for i in top_var_indices:
                plt.arrow(0, 0, x_weights[i, 0]*3, x_weights[i, 1]*3, 
                         head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
                plt.text(x_weights[i, 0]*3.2, x_weights[i, 1]*3.2, x_var_names[i], 
                        fontsize=9, ha='center', va='center')
            
            plt.xlabel(f'成分1 ({x_explained_variance[0]:.2%} X方差)')
            plt.ylabel(f'成分2 ({x_explained_variance[1]:.2%} X方差)')
            plt.title('PLS得分图与变量权重')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        else:
            plt.subplot(2, 3, 4)
            plt.text(0.5, 0.5, '成分不足，无法生成得分图', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('PLS得分图')
        
        # 残差图 (使用残差平方和)
        plt.subplot(2, 3, 5)
        plt.scatter(y_pred.flatten(), residuals.flatten(), alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        
        # 残差Q-Q图
        plt.subplot(2, 3, 6)
        stats.probplot(combined_residuals, dist="norm", plot=plt)
        plt.title('残差 Q-Q 图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存成分解释方差图
        variance_plot_filename = f"pls_variance_plot_{uuid.uuid4().hex}.png"
        variance_plot_filepath = os.path.join(OUTPUT_DIR, variance_plot_filename)
        plt.savefig(variance_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        variance_plot_url = f"{PUBLIC_FILE_BASE_URL}/{variance_plot_filename}"
        
        # 生成变量重要性(VIP)图
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(vip_scores)[::-1]
        top_indices = sorted_indices[:min(15, n_x_vars)]  # 最多显示前15个变量
        
        bars = plt.bar(range(len(top_indices)), vip_scores[top_indices], color='skyblue')
        plt.xlabel('变量')
        plt.ylabel('VIP得分')
        plt.title('变量重要性(VIP)得分 (前15个最重要变量)')
        plt.xticks(range(len(top_indices)), [x_var_names[i] for i in top_indices], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 添加阈值线 (VIP > 1 被认为是重要的)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='重要性阈值(VIP=1)')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存VIP图
        vip_plot_filename = f"pls_vip_plot_{uuid.uuid4().hex}.png"
        vip_plot_filepath = os.path.join(OUTPUT_DIR, vip_plot_filename)
        plt.savefig(vip_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        vip_plot_url = f"{PUBLIC_FILE_BASE_URL}/{vip_plot_filename}"
        
        # 生成前两个成分的得分图
        scores_plot_url = None
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(x_scores[:, 0], x_scores[:, 1], alpha=0.7, s=50, c='blue', label='X样本得分')
            
            # 添加部分变量向量（避免图形过于拥挤）
            n_vectors = min(10, n_x_vars)
            top_var_indices = np.argsort(vip_scores)[-n_vectors:]
            
            for i in top_var_indices:
                plt.arrow(0, 0, x_weights[i, 0]*3, x_weights[i, 1]*3, 
                         head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
                plt.text(x_weights[i, 0]*3.2, x_weights[i, 1]*3.2, x_var_names[i], 
                        fontsize=9, ha='center', va='center')
            
            plt.xlabel(f'成分1 ({x_explained_variance[0]:.2%} X方差)')
            plt.ylabel(f'成分2 ({x_explained_variance[1]:.2%} X方差)')
            plt.title('PLS得分图与变量权重')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # 保存得分图
            scores_plot_filename = f"pls_scores_plot_{uuid.uuid4().hex}.png"
            scores_plot_filepath = os.path.join(OUTPUT_DIR, scores_plot_filename)
            plt.savefig(scores_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            scores_plot_url = f"{PUBLIC_FILE_BASE_URL}/{scores_plot_filename}"
        
        # 组织结果
        # 直接创建完整的结果字典
        result = {
            "n_samples": n_samples,
            "n_x_variables": n_x_vars,
            "n_y_variables": n_y_vars,
            "x_variable_names": x_var_names,
            "y_variable_names": y_var_names,
            "standardized": standardize,
            "n_components": n_components,
            "coefficients": coefficients.tolist(),
            "x_loadings": x_loadings.tolist(),
            "y_loadings": y_loadings.tolist(),
            "x_weights": x_weights.tolist(),
            "y_weights": y_weights.tolist(),
            "x_scores": x_scores.tolist(),
            "y_scores": y_scores.tolist(),
            "r2": float(r2) if not np.isnan(r2) else None,
            "q2": float(q2) if q2 is not None else None,
            "vip_scores": vip_scores.tolist(),
            "x_var_contributions": x_var_contributions.tolist(),
            "x_explained_variance": x_explained_variance.tolist(),
            "y_explained_variance": y_explained_variance.tolist(),
            "x_cumulative_variance": x_cumulative_variance.tolist(),
            "y_cumulative_variance": y_cumulative_variance.tolist(),
            "plots": {
                "variance_plot_url": variance_plot_url,
                "vip_plot_url": vip_plot_url,
                "scores_plot_url": scores_plot_url
            },
            "interpretation": _get_interpretation(r2, q2, vip_scores, x_var_names, y_var_names, 
                                                x_explained_variance, y_explained_variance),
            "residual_diagnostics": {
                "normality_tests": {
                    "shapiro_wilk": sw_test,
                    "kolmogorov_smirnov": ks_test
                },
                "heteroscedasticity_tests": {
                    "breusch_pagan": bp_test
                },
                "autocorrelation_tests": {
                    "durbin_watson": dw_test
                }
            }
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"偏最小二乘回归分析失败: {str(e)}") from e


def _calculate_vip_scores(x_scores: np.ndarray, x_weights: np.ndarray, 
                         y_loadings: np.ndarray, n_x_vars: int) -> np.ndarray:
    """
    计算变量重要性(VIP)得分
    
    参数:
    - x_scores: X得分矩阵
    - x_weights: X权重矩阵
    - y_loadings: Y载荷矩阵
    - n_x_vars: X变量数量
    
    返回:
    - VIP得分数组
    """
    n_components = x_scores.shape[1]
    
    # 计算每个成分对Y的解释方差
    y_var_explained = np.zeros(n_components)
    for i in range(n_components):
        y_var_explained[i] = np.sum(y_loadings[i, :]**2)
    
    # 计算总解释方差
    total_y_var_explained = np.sum(y_var_explained)
    
    # 计算VIP得分
    vip_scores = np.zeros(n_x_vars)
    for j in range(n_x_vars):
        weighted_sum = 0
        for i in range(n_components):
            weighted_sum += y_var_explained[i] * (x_weights[j, i]**2)
        vip_scores[j] = np.sqrt(n_x_vars * weighted_sum / total_y_var_explained)
    
    return vip_scores


def _get_interpretation(r2: float, q2: float, vip_scores: np.ndarray,
                       x_var_names: List[str], y_var_names: List[str],
                       x_explained_variance: np.ndarray, y_explained_variance: np.ndarray) -> str:
    """
    根据PLS回归结果提供解释
    """
    interpretation = "偏最小二乘回归分析完成。"
    
    if r2 is not None:
        interpretation += f"\n模型的R²值为{r2:.3f}，表示模型能解释因变量{r2:.1%}的方差。"
        
        if r2 > 0.8:
            interpretation += "模型拟合效果很好。"
        elif r2 > 0.6:
            interpretation += "模型拟合效果较好。"
        elif r2 > 0.4:
            interpretation += "模型拟合效果一般。"
        else:
            interpretation += "模型拟合效果较差。"
    
    if q2 is not None:
        interpretation += f"\n交叉验证Q²值为{q2:.3f}，表示模型的预测能力。"
        
        if q2 > 0.5:
            interpretation += "模型具有良好的预测能力。"
        elif q2 > 0:
            interpretation += "模型具有一定的预测能力。"
        else:
            interpretation += "模型预测能力较弱。"
    
    # 解释成分效果
    n_components = len(x_explained_variance)
    total_x_explained = np.sum(x_explained_variance)
    total_y_explained = np.sum(y_explained_variance)
    
    interpretation += f"\n\n使用{n_components}个成分，总共解释了{total_x_explained:.1%}的X变量方差和{total_y_explained:.1%}的Y变量方差。"
    
    # 找出最重要的变量
    top_vars_indices = np.argsort(vip_scores)[-5:]  # 前5个最重要的变量
    interpretation += "\n\n最重要的自变量(按VIP得分)："
    for i in reversed(top_vars_indices):
        interpretation += f" {x_var_names[i]}({vip_scores[i]:.2f})"
    
    return interpretation


def _breusch_pagan_test(X, residuals):
    """
    Breusch-Pagan 异方差检验
    
    参数:
    - X: 自变量矩阵
    - residuals: 残差
    
    返回:
    - LM统计量和p值
    """
    n = X.shape[0]
    k = X.shape[1]
    
    # 计算残差平方
    squared_residuals = residuals ** 2
    
    # 对残差平方对自变量进行回归
    model = LinearRegression()
    model.fit(X, squared_residuals)
    fitted_values = model.predict(X)
    
    # 计算R²
    ss_res = np.sum((squared_residuals - fitted_values) ** 2)
    ss_tot = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # LM统计量 = n * R²
    lm_statistic = n * r_squared
    p_value = 1 - stats.chi2.cdf(lm_statistic, k)
    
    return lm_statistic, p_value


def _durbin_watson_test(residuals):
    """
    Durbin-Watson 自相关检验
    
    参数:
    - residuals: 残差
    
    返回:
    - DW统计量
    """
    diff = np.diff(residuals)
    numerator = np.sum(diff ** 2)
    denominator = np.sum(residuals ** 2)
    
    if denominator == 0:
        return 2.0  # 无自相关的情况
    
    dw_statistic = numerator / denominator
    return dw_statistic


def _interpret_dw_statistic(dw_statistic):
    """
    解释Durbin-Watson统计量
    
    参数:
    - dw_statistic: DW统计量
    
    返回:
    - 解释文本
    """
    if dw_statistic < 1.5:
        return "可能存在正自相关"
    elif dw_statistic > 2.5:
        return "可能存在负自相关"
    else:
        return "无明显自相关"