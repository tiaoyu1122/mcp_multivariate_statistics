import os
import uuid
from typing import List, Dict, Any, Optional
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

def perform_stepwise_regression(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("both", description="逐步回归方法(可选): 'forward'(向前), 'backward'(向后), 'both'(双向)"),
    significance_level_enter: float = Field(0.05, description="变量进入模型的显著性水平(可选)"),
    significance_level_remove: float = Field(0.10, description="变量移出模型的显著性水平(可选)"),
) -> dict:
    """
    执行逐步回归分析，包含统计显著性检验和残差分析
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - method: 逐步回归方法 ('forward', 'backward', 'both')
    - significance_level_enter: 变量进入模型的显著性水平 (默认0.05)
    - significance_level_remove: 变量移出模型的显著性水平 (默认0.10)
    """
    # 输入验证
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
    
    if method not in ["forward", "backward", "both"]:
        raise ValueError("method 参数必须是 'forward', 'backward', 或 'both' 之一")

    try:
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        for i, var_data in enumerate(independent_vars_nested):
            data[var_names[i+1]] = var_data
        
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 执行逐步回归
        selected_vars, steps = stepwise_selection(
            X, y, method, significance_level_enter, significance_level_remove
        )
        
        # 对最终模型进行回归分析
        if len(selected_vars) == 0:
            raise ValueError("逐步回归未选择任何变量")
        
        X_final = X[selected_vars]
        
        # 回归
        model = LinearRegression()
        model.fit(X_final, y)
        y_pred = model.predict(X_final)
        
        # 计算残差
        residuals = y - y_pred
        
        # 统计量
        r2 = r2_score(y, y_pred)
        n_vars_final = len(selected_vars)
        adjusted_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_vars_final - 1)
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
        f_statistic = (ssr / n_vars_final) / (sse / (n_obs - n_vars_final - 1))
        f_p_value = 1 - stats.f.cdf(f_statistic, n_vars_final, n_obs - n_vars_final - 1)
        
        # 方程字符串
        equation = f"{var_names[0]} = {intercept:.4f}"
        for i, coef in enumerate(coefficients):
            sign = '+' if coef >= 0 else '-'
            equation += f" {sign} {abs(coef):.4f}*{selected_vars[i]}"
        
        # 计算统计显著性指标
        # 添加常数项用于统计检验
        X_with_const = np.column_stack([np.ones(n_obs), X_final.values])
        
        # 计算MSE
        mse = np.sum(residuals**2) / (n_obs - n_vars_final - 1)
        
        # 计算系数标准误差
        try:
            cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # 计算t统计量和p值
            t_stats = np.concatenate(([intercept], coefficients)) / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - n_vars_final - 1))
            
            # 构建系数摘要
            coef_summary = []
            coef_summary.append({
                "variable": "Intercept",
                "coefficient": intercept,
                "std_error": std_errors[0],
                "t_statistic": t_stats[0],
                "p_value": p_values[0]
            })
            
            for i, var_name in enumerate(selected_vars):
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
        
        # 2. 异方差检验 (Breusch-Pagan 和 White 检验)
        # Breusch-Pagan 检验
        bp_statistic, bp_p_value = _breusch_pagan_test(X_final.values, residuals)
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
        plot_filename = f"stepwise_regression_residuals_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # ✅ 只返回纯结果，无 message/error 字段
        result = {
            "number_of_observations": n_obs,
            "number_of_selected_vars": n_vars_final,
            "selected_variables": selected_vars,
            "regression_equation": equation,
            "r_squared": r2,
            "adjusted_r_squared": adjusted_r2,
            "intercept": intercept,
            "coefficients": dict(zip(selected_vars, coefficients)),
            "residual_plot_url": plot_url,
            "steps": steps
        }
        
        # 添加整体显著性检验结果 (F检验)
        result["overall_significance_test"] = {
            "f_statistic": float(f_statistic),
            "p_value": float(f_p_value),
            "significant": f_p_value < 0.05,
            "degrees_of_freedom": {
                "numerator": n_vars_final,
                "denominator": n_obs - n_vars_final - 1
            }
        }
        
        # 如果成功计算了统计显著性，则添加到结果中
        if coef_summary is not None:
            result["coefficient_summary"] = coef_summary
        
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
        raise RuntimeError(f"逐步回归计算失败: {str(e)}") from e


def stepwise_selection(X, y, method="both", sl_enter=0.05, sl_remove=0.10):
    """
    执行逐步回归选择
    
    参数:
    - X: 自变量数据框
    - y: 因变量数据
    - method: 选择方法 ('forward', 'backward', 'both')
    - sl_enter: 进入显著性水平
    - sl_remove: 移除显著性水平
    
    返回:
    - selected_vars: 选中的变量列表
    - steps: 步骤记录
    """
    initial_list = []
    included = list(initial_list)
    steps = []
    
    candidates = set(X.columns)
    
    while True:
        changed = False
        
        if method in ['forward', 'backward', 'both']:
            # 前向选择
            excluded = list(candidates - set(included))
            best_pval = sl_enter
            new_var = None
            
            for new_column in excluded:
                model = LinearRegression()
                temp_included = included + [new_column]
                model.fit(X[temp_included], y)
                y_pred = model.predict(X[temp_included])
                residuals = y - y_pred
                
                # 计算新增变量的显著性
                X_with_const = np.column_stack([np.ones(len(y)), X[temp_included].values])
                mse = np.sum(residuals**2) / (len(y) - len(temp_included) - 1)
                
                try:
                    cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    t_stats = model.coef_[-1] / std_errors[-1]  # 最后一个变量的t统计量
                    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(temp_included) - 1))
                    
                    if p_val < best_pval:
                        best_pval = p_val
                        new_var = new_column
                except np.linalg.LinAlgError:
                    continue
            
            if new_var is not None:
                included.append(new_var)
                changed = True
                steps.append({
                    "step": len(steps) + 1,
                    "action": "added",
                    "variable": new_var,
                    "p_value": best_pval
                })
        
        if method in ['backward', 'both'] and len(included) > 0:
            # 后向消除
            pvalues = []
            model = LinearRegression()
            model.fit(X[included], y)
            y_pred = model.predict(X[included])
            residuals = y - y_pred
            
            # 计算所有变量的显著性
            X_with_const = np.column_stack([np.ones(len(y)), X[included].values])
            mse = np.sum(residuals**2) / (len(y) - len(included) - 1)
            
            try:
                cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                std_errors = np.sqrt(np.diag(cov_matrix))
                
                for i, var in enumerate(included):
                    t_stats = model.coef_[i] / std_errors[i+1]  # 跳过截距
                    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(included) - 1))
                    pvalues.append((var, p_val))
                
                pvalues.sort(key=lambda x: x[1], reverse=True)
                worst_var, worst_pval = pvalues[0]
                
                if worst_pval > sl_remove:
                    included.remove(worst_var)
                    changed = True
                    steps.append({
                        "step": len(steps) + 1,
                        "action": "removed",
                        "variable": worst_var,
                        "p_value": worst_pval
                    })
            except np.linalg.LinAlgError:
                pass
        
        if not changed:
            break
    
    return included, steps


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