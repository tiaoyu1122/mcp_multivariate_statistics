import os
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
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


def perform_regularized_regression(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("ridge", description="正则化方法: 'ridge'(岭回归), 'lasso'(套索回归), 'elastic_net'(弹性网络)"),
    alpha: float = Field(1.0, description="正则化强度参数"),
    l1_ratio: float = Field(0.5, description="Elastic Net中L1正则化的比例 (仅用于elastic_net方法)"),
    standardize: bool = Field(True, description="是否标准化数据"),
    fit_intercept: bool = Field(True, description="是否计算截距"),
) -> dict:
    """
    执行正则化回归分析（岭回归、套索回归、Elastic Net回归）
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表，第一个为因变量名，后续为自变量名
    - method: 正则化方法 ('ridge', 'lasso', 'elastic_net')
    - alpha: 正则化强度参数
    - l1_ratio: Elastic Net中L1正则化的比例
    - standardize: 是否标准化数据
    - fit_intercept: 是否计算截距
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
    
    # 检查所有自变量的数据长度是否一致
    for i, var_data in enumerate(reshaped_independent_vars):
        if len(var_data) != n_obs:
            raise ValueError(f"自变量 {var_names[i+1]} 的数据长度与因变量不匹配")
    
    if method not in ["ridge", "lasso", "elastic_net"]:
        raise ValueError("method 参数必须是 'ridge', 'lasso', 或 'elastic_net' 之一")
    
    if alpha < 0:
        raise ValueError("alpha 参数必须大于等于0")
    
    if method == "elastic_net" and not (0 <= l1_ratio <= 1):
        raise ValueError("l1_ratio 参数必须在0到1之间")

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
        if method == "ridge":
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
        elif method == "lasso":
            model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=42, max_iter=2000)
        elif method == "elastic_net":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=42, max_iter=2000)
        
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
        
        # 获取系数和截距
        coefficients = model.coef_.tolist()
        intercept = model.intercept_ if fit_intercept else 0.0
        
        # 方程字符串
        equation = f"{var_names[0]} = {intercept:.4f}"
        for i, coef in enumerate(coefficients):
            if abs(coef) > 1e-6:  # 只显示非零系数
                sign = '+' if coef >= 0 else '-'
                equation += f" {sign} {abs(coef):.4f}*{var_names[i+1]}"
        
        # 交叉验证评估
        try:
            cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=min(5, n_obs//5), scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            cv_scores = None
            cv_mean = None
            cv_std = None
        
        # 计算被压缩为零的系数数量（主要用于套索回归）
        zero_coefficients = np.sum(np.abs(coefficients) < 1e-6)
        
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
        
        # 系数路径图（显示各变量系数的大小）
        plt.subplot(2, 3, 4)
        coef_indices = range(len(coefficients))
        bars = plt.bar(coef_indices, coefficients, alpha=0.7)
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('回归系数')
        plt.xticks(coef_indices, var_names[1:], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 为零系数添加颜色标识
        for i, (bar, coef) in enumerate(zip(bars, coefficients)):
            if abs(coef) < 1e-6:
                bar.set_color('red')
        
        # 残差自相关图 (滞后残差图)
        plt.subplot(2, 3, 5)
        lagged_residuals = residuals[:-1]
        current_residuals = residuals[1:]
        plt.scatter(lagged_residuals, current_residuals, alpha=0.7)
        plt.xlabel('滞后残差')
        plt.ylabel('当前残差')
        plt.title('残差自相关图')
        plt.grid(True, alpha=0.3)
        # 添加趋势线
        if len(lagged_residuals) > 1:
            z = np.polyfit(lagged_residuals, current_residuals, 1)
            p = np.poly1d(z)
            plt.plot(lagged_residuals, p(lagged_residuals), "r--", alpha=0.8)
        
        # 残差正态性检验图
        plt.subplot(2, 3, 6)
        # 绘制残差的经验累积分布函数与理论正态分布的比较
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(0, 1, len(sorted_residuals))
        normal_quantiles = np.percentile(np.random.normal(0, 1, len(sorted_residuals)), theoretical_quantiles * 100)
        plt.scatter(normal_quantiles, sorted_residuals, alpha=0.7)
        plt.xlabel('理论正态分位数')
        plt.ylabel('残差分位数')
        plt.title('残差正态性检验图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存残差图
        plot_filename = f"regularized_regression_residuals_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 生成系数重要性图
        plt.figure(figsize=(10, 6))
        coef_abs = np.abs(coefficients)
        sorted_indices = np.argsort(coef_abs)[::-1]
        top_indices = sorted_indices[:min(15, len(coefficients))]  # 最多显示前15个变量
        
        bars = plt.bar(range(len(top_indices)), coef_abs[top_indices], color='skyblue')
        plt.xlabel('变量')
        plt.ylabel('|系数值|')
        plt.title('系数重要性 (按绝对值排序，前15个)')
        plt.xticks(range(len(top_indices)), [var_names[i+1] for i in top_indices], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存系数重要性图
        coef_plot_filename = f"regularized_regression_coefficients_{uuid.uuid4().hex}.png"
        coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
        plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        
        # 组织结果
        result = {
            "number_of_observations": n_obs,
            "number_of_independent_vars": n_vars,
            "method": method,
            "standardized": standardize,
            "regularization_parameters": {
                "alpha": alpha,
                "l1_ratio": l1_ratio if method == "elastic_net" else None
            },
            "regression_equation": equation,
            "r_squared": r2,
            "adjusted_r_squared": adjusted_r2,
            "mse": mse,
            "rmse": rmse,
            "intercept": intercept,
            "coefficients": dict(zip(var_names[1:], coefficients)),
            "zero_coefficients": int(zero_coefficients),
            "cross_validation": {
                "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
                "mean_cv_score": float(cv_mean) if cv_mean is not None else None,
                "std_cv_score": float(cv_std) if cv_std is not None else None
            },
            "residual_plot_url": plot_url,
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_interpretation(method, r2, adjusted_r2, cv_mean, zero_coefficients, n_vars)
        }
        
        # 添加残差诊断和模型假设检验结果
        result["residual_diagnostics"] = {
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
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"正则化回归分析失败: {str(e)}") from e


def _get_interpretation(method: str, r2: float, adjusted_r2: float, cv_mean: float, 
                       zero_coefficients: int, n_vars: int) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = f"正则化回归分析({method})完成。\n"
    
    interpretation += f"模型的R²值为{r2:.4f}，调整后R²值为{adjusted_r2:.4f}。"
    
    if r2 > 0.8:
        interpretation += "模型拟合效果很好。"
    elif r2 > 0.6:
        interpretation += "模型拟合效果较好。"
    elif r2 > 0.4:
        interpretation += "模型拟合效果一般。"
    else:
        interpretation += "模型拟合效果较差。"
    
    if cv_mean is not None:
        interpretation += f"\n交叉验证平均R²值为{cv_mean:.4f}，表明模型具有"
        if cv_mean > 0.7:
            interpretation += "很好的泛化能力。"
        elif cv_mean > 0.5:
            interpretation += "较好的泛化能力。"
        else:
            interpretation += "一般的泛化能力。"
    
    if method == "lasso":
        interpretation += f"\nLasso回归将{zero_coefficients}个变量的系数压缩为零，起到了变量选择的作用。"
        if zero_coefficients > 0:
            interpretation += "这有助于简化模型并提高解释性。"
    
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