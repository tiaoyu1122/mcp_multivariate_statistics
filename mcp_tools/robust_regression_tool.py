import os
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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


def perform_robust_regression(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("ransac", description="稳健回归方法: 'ransac'(RANSAC回归), 'huber'(Huber回归)"),
    standardize: bool = Field(True, description="是否标准化数据"),
    fit_intercept: bool = Field(True, description="是否计算截距"),
    # RANSAC特定参数
    min_samples: Optional[int] = Field(None, description="RANSAC算法中随机样本的最小数量"),
    residual_threshold: Optional[float] = Field(None, description="RANSAC算法中样本被视为内点的最大残差"),
    max_trials: int = Field(100, description="RANSAC算法的最大迭代次数"),
    # Huber特定参数
    epsilon: float = Field(1.35, description="Huber回归的参数，决定对异常值的敏感度"),
    alpha: float = Field(0.0001, description="Huber回归的正则化强度"),
) -> dict:
    """
    执行稳健回归分析（RANSAC回归或Huber回归）
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - method: 稳健回归方法 ('ransac', 'huber')
    - standardize: 是否标准化数据
    - fit_intercept: 是否计算截距
    - min_samples: RANSAC算法中随机样本的最小数量
    - residual_threshold: RANSAC算法中样本被视为内点的最大残差
    - max_trials: RANSAC算法的最大迭代次数
    - epsilon: Huber回归的参数，决定对异常值的敏感度
    - alpha: Huber回归的正则化强度
    
    返回:
    - 包含稳健回归分析结果的字典
    """
    # 输入验证
    n_obs = len(dependent_var)
    
    # 根据var_names中元素的个数-1计算自变量数量
    n_vars = len(var_names) - 1
    
    if n_obs < n_vars + 1:
        raise ValueError("观测数量不足，无法进行回归分析")
    
    if len(independent_vars) != n_obs * n_vars:
        raise ValueError("自变量数据长度与变量数量和观测数量不匹配")
    
    if n_vars <= 0:
        raise ValueError("自变量数量必须大于0")
    
    # 将一维数组还原为嵌套列表结构
    reshaped_independent_vars = []
    for i in range(n_vars):
        var_data = independent_vars[i * n_obs:(i + 1) * n_obs]
        reshaped_independent_vars.append(var_data)
    
    if method not in ["ransac", "huber"]:
        raise ValueError("method 参数必须是 'ransac' 或 'huber'")
    
    try:
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        for i, var_data in enumerate(reshaped_independent_vars):
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
        
        # 根据方法选择模型
        if method == "ransac":
            # 处理参数，确保传递正确的值给RANSACRegressor
            ransac_params = {
                "estimator": LinearRegression(fit_intercept=fit_intercept),
                "max_trials": max_trials,
                "random_state": 42
            }
            
            # 只有当参数不是None时才添加
            if min_samples is not None:
                ransac_params["min_samples"] = min_samples
            if residual_threshold is not None:
                ransac_params["residual_threshold"] = residual_threshold
                
            model = RANSACRegressor(**ransac_params)
        elif method == "huber":
            model = HuberRegressor(
                epsilon=epsilon,
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=1000
            )
        
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
        r2 = r2_score(y, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_vars - 1)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        # 获取系数和截距
        if method == "ransac":
            coefficients = model.estimator_.coef_.tolist()
            intercept = model.estimator_.intercept_
        else:  # huber
            coefficients = model.coef_.tolist()
            intercept = model.intercept_
        
        # 方程字符串
        equation = f"{var_names[0]} = {intercept:.4f}"
        for i, coef in enumerate(coefficients):
            if abs(coef) > 1e-6:  # 只显示非零系数
                sign = '+' if coef >= 0 else '-'
                equation += f" {sign} {abs(coef):.4f}*{var_names[i+1]}"
        
        # 识别异常值（仅适用于RANSAC）
        outliers = None
        inlier_mask = None
        if method == "ransac":
            inlier_mask = model.inlier_mask_ if hasattr(model, 'inlier_mask_') else np.full(n_obs, True)
            outliers = np.where(~inlier_mask)[0].tolist()
        
        # 残差诊断和模型假设检验
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
        
        # 生成残差图
        plt.figure(figsize=(18, 12))
        
        # 残差 vs 拟合值图
        plt.subplot(2, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.7)
        if method == "ransac" and inlier_mask is not None:
            plt.scatter(y_pred[inlier_mask], residuals[inlier_mask], color='blue', alpha=0.7, label='内点')
            plt.scatter(y_pred[outliers], residuals[outliers], color='red', alpha=0.7, label='异常值')
            plt.legend()
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('拟合值')
        plt.ylabel('残差')
        plt.title('残差 vs 拟合值')
        plt.grid(True, alpha=0.3)
        
        # Q-Q图
        plt.subplot(2, 3, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('残差 Q-Q 图')
        plt.grid(True, alpha=0.3)
        
        # 残差直方图
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.title('残差分布直方图')
        plt.grid(True, alpha=0.3)
        
        # 标准化残差图
        standardized_residuals = residuals / np.sqrt(mse)
        plt.subplot(2, 3, 4)
        plt.scatter(y_pred, standardized_residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.axhline(y=2, color='r', linestyle=':', alpha=0.7)
        plt.axhline(y=-2, color='r', linestyle=':', alpha=0.7)
        plt.xlabel('拟合值')
        plt.ylabel('标准化残差')
        plt.title('标准化残差图')
        plt.grid(True, alpha=0.3)
        
        # 拟合值 vs 实际值图
        plt.subplot(2, 3, 5)
        plt.scatter(y, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        if method == "ransac" and inlier_mask is not None:
            plt.scatter(y[inlier_mask], y_pred[inlier_mask], color='blue', alpha=0.7, label='内点')
            plt.scatter(y[outliers], y_pred[outliers], color='red', alpha=0.7, label='异常值')
            plt.legend()
        plt.xlabel('实际值')
        plt.ylabel('拟合值')
        plt.title('实际值 vs 拟合值')
        plt.grid(True, alpha=0.3)
        
        # 系数重要性图
        plt.subplot(2, 3, 6)
        coef_indices = range(len(coefficients))
        bars = plt.bar(coef_indices, coefficients, alpha=0.7)
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('回归系数')
        plt.xticks(coef_indices, var_names[1:], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存残差图
        plot_filename = f"robust_regression_residuals_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 直接创建完整的结果字典
        result = {
            "number_of_observations": n_obs,
            "number_of_independent_vars": n_vars,
            "method": method,
            "standardized": standardize,
            "regression_equation": equation,
            "r_squared": r2,
            "adjusted_r_squared": adjusted_r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "intercept": intercept,
            "coefficients": dict(zip(var_names[1:], coefficients)),
            "residual_plot_url": plot_url,
            "interpretation": _get_interpretation(method, r2, adjusted_r2, outliers),
            "outliers": outliers if method == "ransac" and outliers is not None else None,
            "number_of_outliers": len(outliers) if method == "ransac" and outliers is not None else None,
            "inlier_ratio": (n_obs - len(outliers)) / n_obs if method == "ransac" and outliers is not None else None,
            "huber_parameters": {
                "epsilon": epsilon,
                "alpha": alpha
            } if method == "huber" else None,
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
        raise RuntimeError(f"稳健回归分析失败: {str(e)}") from e


def _get_interpretation(method: str, r2: float, adjusted_r2: float, outliers: List[int]) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = f"稳健回归分析({method.upper()})完成。\n"
    
    interpretation += f"模型的R²值为{r2:.4f}，调整后R²值为{adjusted_r2:.4f}。"
    
    if r2 > 0.8:
        interpretation += "模型拟合效果很好。"
    elif r2 > 0.6:
        interpretation += "模型拟合效果较好。"
    elif r2 > 0.4:
        interpretation += "模型拟合效果一般。"
    else:
        interpretation += "模型拟合效果较差。"
    
    if method == "ransac" and outliers is not None:
        interpretation += f"\nRANSAC算法检测到{len(outliers)}个异常值，"
        if len(outliers) > 0:
            interpretation += "这些数据点对模型的影响已被降低。"
        else:
            interpretation += "数据中未发现明显的异常值。"
    
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