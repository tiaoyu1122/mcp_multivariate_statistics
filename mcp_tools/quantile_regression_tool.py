import os
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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


def perform_quantile_regression(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    quantiles: List[float] = Field([0.25, 0.5, 0.75], description="分位数列表，每个值应在0到1之间"),
    alpha: float = Field(1.0, description="正则化强度参数"),
    standardize: bool = Field(True, description="是否标准化数据"),
) -> dict:
    """
    执行分位数回归分析
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表，第一个为因变量名，后续为自变量名
    - quantiles: 分位数列表
    - alpha: 正则化强度参数
    - standardize: 是否标准化数据
    """
    # 输入验证
    n_obs = len(dependent_var)
    
    # 计算自变量数量（变量名称数量-1）
    n_vars = len(var_names) - 1
    
    if n_obs < n_vars + 1:
        raise ValueError("观测数量不足，无法进行回归分析")
    
    if len(var_names) < 2:
        raise ValueError("变量名称数量至少为2（因变量和至少一个自变量）")
    
    if len(independent_vars) != n_vars * n_obs:
        raise ValueError("自变量数据长度与变量数量和观测数量不匹配")
    
    # 验证分位数
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError("分位数值必须在0到1之间")
    
    if alpha < 0:
        raise ValueError("alpha 参数必须大于等于0")

    try:
        # 将一维数组重构为嵌套列表
        independent_vars_nested = []
        for i in range(n_vars):
            var_data = independent_vars[i * n_obs : (i + 1) * n_obs]
            independent_vars_nested.append(var_data)
        
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        for i, var_data in enumerate(independent_vars_nested):
            data[var_names[i+1]] = var_data
        
        X = data[var_names[1:]]
        y = data[var_names[0]]
        
        # 数据标准化
        if standardize:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        else:
            X_scaled = X
            y_scaled = y
        
        # 存储不同分位数的结果
        results_by_quantile = {}
        predictions = {}
        
        # 对每个分位数进行回归
        for q in quantiles:
            # 创建分位数回归模型
            model = QuantileRegressor(quantile=q, alpha=alpha, solver='highs')
            
            # 拟合模型
            model.fit(X_scaled, y_scaled)
            y_pred_scaled = model.predict(X_scaled)
            
            # 如果进行了标准化，需要将预测值转换回原始尺度
            if standardize:
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred_scaled
            
            # 计算残差
            residuals = y - y_pred
            
            # 统计量
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # 获取系数和截距
            coefficients = model.coef_.tolist()
            intercept = model.intercept_
            
            # 方程字符串
            equation = f"{var_names[0]} = {intercept:.4f}"
            for i, coef in enumerate(coefficients):
                if abs(coef) > 1e-6:  # 只显示非零系数
                    sign = '+' if coef >= 0 else '-'
                    equation += f" {sign} {abs(coef):.4f}*{var_names[i+1]}"
            
            # 存储预测值用于后续可视化
            predictions[q] = y_pred
            
            # 残差诊断和模型假设检验 (以中位数回归为例)
            residual_diagnostics = None
            if q == 0.5:  # 只对中位数回归进行残差诊断
                # 1. 残差正态性检验 (Shapiro-Wilk 和 K-S 检验)
                # Shapiro-Wilk 检验 (适用于小样本)
                if n_obs <= 5000:  # Shapiro-Wilk 检验适用于样本量小于5000的情况
                    sw_statistic, sw_p_value = stats.shapiro(residuals)
                    sw_test = {
                        "test": "Shapiro-Wilk",
                        "statistic": float(sw_statistic),
                        "p_value": float(sw_p_value),
                        "is_normal": sw_p_value > 0.05
                    }
                else:
                    sw_test = None
                
                # K-S 检验 (适用于大样本)
                ks_statistic, ks_p_value = stats.kstest(residuals, 'norm')
                ks_test = {
                    "test": "Kolmogorov-Smirnov",
                    "statistic": float(ks_statistic),
                    "p_value": float(ks_p_value),
                    "is_normal": ks_p_value > 0.05
                }
                
                # 2. 异方差检验 (Breusch-Pagan 检验)
                # Breusch-Pagan 检验
                bp_statistic, bp_p_value = _breusch_pagan_test(X.values, residuals)
                bp_test = {
                    "test": "Breusch-Pagan",
                    "statistic": float(bp_statistic),
                    "p_value": float(bp_p_value),
                    "has_heteroscedasticity": bp_p_value < 0.05
                }
                
                # 3. 自相关检验 (Durbin-Watson 检验)
                dw_statistic = _durbin_watson_test(residuals)
                dw_test = {
                    "test": "Durbin-Watson",
                    "statistic": float(dw_statistic),
                    "interpretation": _interpret_dw_statistic(dw_statistic)
                }
                
                residual_diagnostics = {
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
            
            # 保存结果
            results_by_quantile[q] = {
                "regression_equation": equation,
                "intercept": intercept,
                "coefficients": dict(zip(var_names[1:], coefficients)),
                "mse": mse,
                "mae": mae,
                "residual_diagnostics": residual_diagnostics
            }
        
        # 生成综合可视化图
        plt.figure(figsize=(18, 12))
        
        # 1. 因变量的实际值vs预测值散点图（以中位数为例）
        median_q = 0.5 if 0.5 in quantiles else quantiles[0]
        y_pred_median = predictions[median_q]
        
        plt.subplot(2, 3, 1)
        plt.scatter(y, y_pred_median, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'实际值 vs 预测值 (分位数={median_q})')
        plt.grid(True, alpha=0.3)
        
        # 2. 残差图（以中位数为例）
        residuals_median = y - y_pred_median
        
        plt.subplot(2, 3, 2)
        plt.scatter(y_pred_median, residuals_median, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title(f'残差图 (分位数={median_q})')
        plt.grid(True, alpha=0.3)
        
        # 3. Q-Q图（以中位数为例）
        plt.subplot(2, 3, 3)
        stats.probplot(residuals_median, dist="norm", plot=plt)
        plt.title(f'残差 Q-Q 图 (分位数={median_q})')
        plt.grid(True, alpha=0.3)
        
        # 4. 系数对比图
        plt.subplot(2, 3, 4)
        coef_data = []
        for q in quantiles:
            coefs = list(results_by_quantile[q]["coefficients"].values())
            coef_data.append(coefs)
        
        coef_data = np.array(coef_data)
        x_pos = np.arange(len(var_names) - 1)
        width = 0.8 / len(quantiles)
        
        for i, q in enumerate(quantiles):
            plt.bar(x_pos + i * width, coef_data[i], width, label=f'τ={q}')
        
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('不同分位数下的回归系数')
        plt.xticks(x_pos + width * (len(quantiles) - 1) / 2, var_names[1:], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 不同分位数的系数路径图（以第一个变量为例）
        if len(var_names) > 1:
            plt.subplot(2, 3, 5)
            first_var_coefs = [results_by_quantile[q]["coefficients"][var_names[1]] for q in quantiles]
            plt.plot(quantiles, first_var_coefs, 'o-', linewidth=2, markersize=6)
            plt.xlabel('分位数')
            plt.ylabel(f'{var_names[1]} 的系数')
            plt.title(f'{var_names[1]} 系数随分位数变化')
            plt.grid(True, alpha=0.3)
        
        # 6. 残差直方图（以中位数为例）
        plt.subplot(2, 3, 6)
        plt.hist(residuals_median, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.title(f'残差分布直方图 (分位数={median_q})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        plot_filename = f"quantile_regression_results_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 组织最终结果
        result = {
            "number_of_observations": n_obs,
            "number_of_independent_vars": n_vars,
            "quantiles": quantiles,
            "standardized": standardize,
            "regularization_parameter": alpha,
            "results_by_quantile": results_by_quantile,
            "plot_url": plot_url,
            "interpretation": _get_interpretation(results_by_quantile, quantiles)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"分位数回归分析失败: {str(e)}") from e


def _get_interpretation(results_by_quantile: dict, quantiles: List[float]) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = f"分位数回归分析完成，共计算了{len(quantiles)}个分位数: {quantiles}。\n"
    
    # 分析系数的变化
    if len(quantiles) > 1:
        first_var_name = list(results_by_quantile[quantiles[0]]["coefficients"].keys())[0]
        first_coef_first_q = results_by_quantile[quantiles[0]]["coefficients"][first_var_name]
        first_coef_last_q = results_by_quantile[quantiles[-1]]["coefficients"][first_var_name]
        
        if abs(first_coef_first_q - first_coef_last_q) > 0.01:  # 系数变化较大
            interpretation += f"变量 {first_var_name} 的系数在不同分位数下有明显变化，说明该变量对因变量的影响在不同分位数下是不同的，这体现了分位数回归的价值。\n"
        else:
            interpretation += f"变量 {first_var_name} 的系数在不同分位数下变化较小，说明该变量对因变量的影响在整个分布中相对稳定。\n"
    
    # 分析拟合效果
    mse_values = [results_by_quantile[q]["mse"] for q in quantiles]
    avg_mse = np.mean(mse_values)
    
    interpretation += f"平均均方误差(MSE)为 {avg_mse:.4f}。"
    
    if avg_mse < 1:
        interpretation += "模型拟合效果很好。"
    elif avg_mse < 5:
        interpretation += "模型拟合效果较好。"
    elif avg_mse < 10:
        interpretation += "模型拟合效果一般。"
    else:
        interpretation += "模型拟合效果较差。"
    
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