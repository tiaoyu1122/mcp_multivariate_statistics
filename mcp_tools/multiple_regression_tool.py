import os
import uuid
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def perform_multiple_regression(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
) -> dict:
    """
    执行多元线性回归分析，包含统计显著性检验和残差分析
    """
    # 输入验证（失败则抛异常）
    n_obs = len(dependent_var)
    n_vars = len(var_names) - 1  # 自变量个数
    
    # 将一维的independent_vars转换为嵌套list
    if len(independent_vars) != n_obs * n_vars:
        raise ValueError("自变量数据长度与因变量数据长度和变量名称数量不匹配")
    
    # 将一维数组还原为嵌套list
    independent_vars_nested = []
    for i in range(n_vars):
        var_data = independent_vars[i * n_obs:(i + 1) * n_obs]
        independent_vars_nested.append(var_data)
    
    if n_obs < n_vars + 1:
        raise ValueError("观测数量不足，无法进行回归分析")
    
    for i, var_data in enumerate(independent_vars_nested):
        if len(var_data) != n_obs:
            raise ValueError(f"自变量 {var_names[i+1]} 的数据长度与因变量不匹配")

    try:
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        for i, var_data in enumerate(independent_vars_nested):
            data[var_names[i+1]] = var_data
        
        X = data[var_names[1:]]
        y = data[var_names[0]]
        
        # 回归
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # 计算残差
        residuals = y - y_pred
        
        # 统计量
        r2 = r2_score(y, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_vars - 1)
        coefficients = model.coef_.tolist()
        intercept = model.intercept_
        
        # 整体显著性检验 (F检验)
        # 计算总平方和(SST)、回归平方和(SSR)和残差平方和(SSE)
        sst = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        ssr = np.sum((y_pred - np.mean(y)) ** 2)  # 回归平方和
        sse = np.sum(residuals ** 2)  # 残差平方和
        
        # F检验统计量
        # F = (SSR/k) / (SSE/(n-k-1))
        # 其中k是自变量个数
        f_statistic = (ssr / n_vars) / (sse / (n_obs - n_vars - 1))
        f_p_value = 1 - stats.f.cdf(f_statistic, n_vars, n_obs - n_vars - 1)
        
        # 多重共线性检验
        # 计算方差膨胀因子(VIF)
        vif_data = []
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # 为VIF计算添加常数项
            X_with_const = np.column_stack([np.ones(n_obs), X.values])
            feature_names = ['const'] + var_names[1:]
            
            for i in range(1, len(feature_names)):  # 跳过常数项
                vif = variance_inflation_factor(X_with_const, i)
                vif_data.append({
                    "variable": feature_names[i],
                    "vif": vif
                })
        except ImportError:
            # 如果没有安装statsmodels，跳过多重共线性检验
            vif_data = None
        
        # 方程字符串
        equation = f"{var_names[0]} = {intercept:.4f}"
        for i, coef in enumerate(coefficients):
            sign = '+' if coef >= 0 else '-'
            equation += f" {sign} {abs(coef):.4f}*{var_names[i+1]}"
        
        # 计算统计显著性指标
        # 添加常数项用于统计检验
        X_with_const = np.column_stack([np.ones(n_obs), X.values])
        
        # 计算MSE
        mse = np.sum(residuals**2) / (n_obs - n_vars - 1)
        
        # 计算系数标准误差
        try:
            cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # 计算t统计量和p值
            t_stats = np.concatenate(([intercept], coefficients)) / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - n_vars - 1))
            
            # 构建系数摘要
            coef_summary = []
            coef_summary.append({
                "variable": "Intercept",
                "coefficient": intercept,
                "std_error": std_errors[0],
                "t_statistic": t_stats[0],
                "p_value": p_values[0]
            })
            
            for i, var_name in enumerate(var_names[1:]):
                coef_summary.append({
                    "variable": var_name,
                    "coefficient": coefficients[i],
                    "std_error": std_errors[i+1],
                    "t_statistic": t_stats[i+1],
                    "p_value": p_values[i+1]
                })
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，则无法计算统计显著性
            coef_summary = None
        
        # 生成残差图
        plt.figure(figsize=(15, 12))
        
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
        
        # 残差自相关图 (Durbin-Watson检验)
        dw_statistic = _durbin_watson_test(residuals)
        plt.subplot(2, 3, 5)
        lag_residuals = residuals[1:]
        plt.scatter(residuals[:-1], lag_residuals, alpha=0.7)
        plt.xlabel('残差(t-1)')
        plt.ylabel('残差(t)')
        plt.title(f'残差自相关图\nDW统计量: {dw_statistic:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 残差正态性检验 (Shapiro-Wilk, K-S)
        normality_tests = _residual_normality_tests(residuals)
        plt.subplot(2, 3, 6)
        plt.axis('off')
        normality_text = f"残差正态性检验:\n\n"
        normality_text += f"Shapiro-Wilk:\n  统计量={normality_tests['shapiro_wilk']['statistic']:.4f}\n  p值={normality_tests['shapiro_wilk']['p_value']:.4f}\n\n"
        normality_text += f"K-S:\n  统计量={normality_tests['kolmogorov_smirnov']['statistic']:.4f}\n  p值={normality_tests['kolmogorov_smirnov']['p_value']:.4f}"
        plt.text(0.1, 0.5, normality_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        plt.title('残差正态性检验')
        
        plt.tight_layout()
        
        # 保存残差图
        plot_filename = f"regression_residuals_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # ✅ 只返回纯结果，无 message/error 字段
        # 直接创建完整的结果字典
        result = {
            "number_of_observations": n_obs,
            "number_of_independent_vars": n_vars,
            "regression_equation": equation,
            "r_squared": r2,
            "adjusted_r_squared": adjusted_r2,
            "intercept": intercept,
            "coefficients": dict(zip(var_names[1:], coefficients)),
            "residual_plot_url": plot_url,
            "overall_significance_test": {
                "f_statistic": float(f_statistic),
                "p_value": float(f_p_value),
                "significant": f_p_value < 0.05,
                "degrees_of_freedom": {
                    "numerator": n_vars,
                    "denominator": n_obs - n_vars - 1
                }
            },
            "coefficient_summary": coef_summary if coef_summary is not None else None,
            "multicollinearity_test": vif_data if vif_data is not None else None,
            "residual_diagnostics": {
                "durbin_watson_test": {
                    "statistic": float(dw_statistic),
                    "interpretation": _interpret_durbin_watson(dw_statistic)
                },
                "normality_tests": normality_tests,
                "heteroscedasticity_tests": _heteroscedasticity_tests(X.values, residuals)
            }
        }
        
        return result

    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"回归计算失败: {str(e)}") from e


def _durbin_watson_test(residuals):
    """
    执行Durbin-Watson检验以检测残差的自相关性
    """
    diff = np.diff(residuals)
    dw_statistic = np.sum(diff**2) / np.sum(residuals**2)
    return dw_statistic


def _interpret_durbin_watson(dw_statistic):
    """
    解释Durbin-Watson统计量
    """
    if dw_statistic < 1.5:
        return "存在正自相关"
    elif dw_statistic > 2.5:
        return "存在负自相关"
    else:
        return "无明显自相关"


def _residual_normality_tests(residuals):
    """
    执行残差正态性检验 (Shapiro-Wilk, Kolmogorov-Smirnov)
    """
    try:
        # Shapiro-Wilk检验 (适用于小样本)
        if len(residuals) <= 5000:  # Shapiro-Wilk适用于n <= 5000
            sw_stat, sw_p = stats.shapiro(residuals)
        else:
            # 对于大样本，使用近似方法
            sw_stat, sw_p = stats.normaltest(residuals)
        
        # Kolmogorov-Smirnov检验
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals, ddof=1)))
        
        return {
            "shapiro_wilk": {
                "statistic": float(sw_stat),
                "p_value": float(sw_p),
                "is_normal": sw_p > 0.05
            },
            "kolmogorov_smirnov": {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "is_normal": ks_p > 0.05
            }
        }
    except Exception as e:
        return {
            "shapiro_wilk": {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "error": str(e)
            },
            "kolmogorov_smirnov": {
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "error": str(e)
            }
        }


def _heteroscedasticity_tests(X, residuals):
    """
    执行异方差检验 (Breusch-Pagan, White)
    """
    try:
        n_obs, n_vars = X.shape
        
        # Breusch-Pagan检验
        # 回归残差平方对解释变量
        residuals_squared = residuals**2
        bp_model = LinearRegression()
        bp_model.fit(X, residuals_squared)
        bp_r2 = bp_model.score(X, residuals_squared)
        bp_statistic = n_obs * bp_r2
        bp_p_value = 1 - stats.chi2.cdf(bp_statistic, n_vars)
        
        # White检验 (使用解释变量、其平方项和交叉项)
        # 构建White检验的变量矩阵
        X_with_squares = np.column_stack([X, X**2])
        # 添加交叉项 (只考虑前5个变量以避免维度爆炸)
        max_vars = min(5, n_vars)
        cross_terms = []
        for i in range(max_vars):
            for j in range(i+1, max_vars):
                cross_terms.append(X[:, i] * X[:, j])
        
        if cross_terms:
            cross_terms = np.column_stack(cross_terms)
            X_white = np.column_stack([np.ones(n_obs), X_with_squares, cross_terms])
        else:
            X_white = np.column_stack([np.ones(n_obs), X_with_squares])
        
        # White检验回归
        white_model = LinearRegression()
        white_model.fit(X_white[:, 1:], residuals_squared)  # 不包括常数项
        white_r2 = white_model.score(X_white[:, 1:], residuals_squared)
        white_statistic = n_obs * white_r2
        white_p_value = 1 - stats.chi2.cdf(white_statistic, X_white.shape[1] - 1)
        
        return {
            "breusch_pagan": {
                "statistic": float(bp_statistic),
                "p_value": float(bp_p_value),
                "has_heteroscedasticity": bp_p_value < 0.05
            },
            "white": {
                "statistic": float(white_statistic),
                "p_value": float(white_p_value),
                "has_heteroscedasticity": white_p_value < 0.05
            }
        }
    except Exception as e:
        return {
            "breusch_pagan": {
                "statistic": None,
                "p_value": None,
                "has_heteroscedasticity": None,
                "error": str(e)
            },
            "white": {
                "statistic": None,
                "p_value": None,
                "has_heteroscedasticity": None,
                "error": str(e)
            }
        }
