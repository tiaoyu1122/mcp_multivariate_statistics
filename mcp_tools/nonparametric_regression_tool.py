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

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline


def perform_nonparametric_regression(
    x_data: List[float] = Field(..., description="自变量数据"),
    y_data: List[float] = Field(..., description="因变量数据"),
    method: str = Field("loess", description="非参数回归方法: 'loess'(局部加权回归), 'spline'(样条回归)"),
    loess_frac: float = Field(0.3, description="LOESS方法中用于局部回归的窗口大小比例，值越大越平滑"),
    loess_it: int = Field(3, description="LOESS方法的迭代次数"),
    spline_degree: int = Field(3, description="样条回归的多项式阶数"),
    spline_smooth_factor: Optional[float] = Field(None, description="样条回归的平滑因子，值越大越平滑，None表示自动选择"),
    confidence_level: float = Field(0.95, description="置信区间水平，如0.95表示95%置信区间"),
) -> dict:
    """
    执行非参数/半参数回归分析（LOESS/LOWESS、样条回归）
    
    参数:
    - x_data: 自变量数据
    - y_data: 因变量数据
    - method: 非参数回归方法 ('loess', 'spline')
    - loess_frac: LOESS方法中用于局部回归的窗口大小比例
    - loess_it: LOESS方法的迭代次数
    - spline_degree: 样条回归的多项式阶数
    - spline_smooth_factor: 样条回归的平滑因子
    - confidence_level: 置信区间水平
    
    返回:
    - 包含非参数回归分析结果的字典
    """    
    try:
        # 输入验证
        n_obs = len(x_data)
        
        if n_obs == 0:
            raise ValueError("数据不能为空")
        
        if len(y_data) != n_obs:
            raise ValueError("因变量数据长度必须与自变量数据长度相同")
        
        if method not in ["loess", "spline"]:
            raise ValueError("method 参数必须是 'loess' 或 'spline'")
        
        if not 0 < loess_frac <= 1:
            raise ValueError("loess_frac 参数必须在(0, 1]范围内")
        
        if loess_it < 0:
            raise ValueError("loess_it 参数必须大于等于0")
        
        if spline_degree < 1:
            raise ValueError("spline_degree 参数必须大于等于1")
        
        if spline_smooth_factor is not None and spline_smooth_factor < 0:
            raise ValueError("spline_smooth_factor 参数必须大于等于0或为None")
        
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level 参数必须在(0, 1)范围内")
        
        # 创建DataFrame并排序
        df = pd.DataFrame({'x': x_data, 'y': y_data})
        df = df.sort_values('x').reset_index(drop=True)
        x_sorted = df['x'].values
        y_sorted = df['y'].values
        
        # 根据方法执行非参数回归
        if method == "loess":
            # LOESS/LOWESS回归
            result = _perform_loess_regression(
                x_sorted, y_sorted, loess_frac, loess_it, confidence_level
            )
            method_name = "局部加权回归 (LOESS/LOWESS)"
            
        elif method == "spline":
            # 样条回归
            result = _perform_spline_regression(
                x_sorted, y_sorted, spline_degree, spline_smooth_factor, confidence_level
            )
            method_name = "样条回归 (Spline Regression)"
        
        # 生成可视化图
        plt.figure(figsize=(15, 10))
        
        # 1. 原始数据和拟合曲线
        plt.subplot(2, 3, 1)
        plt.scatter(x_sorted, y_sorted, alpha=0.6, color='lightblue', label='原始数据')
        
        if method == "loess":
            plt.plot(result['fitted_values']['x'], result['fitted_values']['y'], 
                    color='red', linewidth=2, label='LOESS拟合')
            # 绘制置信区间
            if 'confidence_interval' in result:
                ci = result['confidence_interval']
                plt.fill_between(ci['x'], ci['lower'], ci['upper'], 
                               color='red', alpha=0.2, label=f'{confidence_level*100}%置信区间')
        elif method == "spline":
            plt.plot(result['fitted_values']['x'], result['fitted_values']['y'], 
                    color='red', linewidth=2, label='样条拟合')
            # 绘制置信区间
            if 'confidence_interval' in result:
                ci = result['confidence_interval']
                plt.fill_between(ci['x'], ci['lower'], ci['upper'], 
                               color='red', alpha=0.2, label=f'{confidence_level*100}%置信区间')
        
        plt.xlabel('自变量 (X)')
        plt.ylabel('因变量 (Y)')
        plt.title('非参数回归拟合结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 残差图
        plt.subplot(2, 3, 2)
        residuals = y_sorted - result['fitted_values']['y_original']
        plt.scatter(result['fitted_values']['y_original'], residuals, alpha=0.6, color='lightblue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('拟合值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        
        # 3. 残差Q-Q图
        plt.subplot(2, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('残差Q-Q图')
        plt.grid(True, alpha=0.3)
        
        # 4. 残差直方图
        plt.subplot(2, 3, 4)
        plt.hist(residuals, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.title('残差分布直方图')
        plt.grid(True, alpha=0.3)
        
        # 5. 方法说明
        plt.subplot(2, 3, 5)
        plt.axis('off')
        method_explanations = {
            "loess": "局部加权回归 (LOESS/LOWESS):\n- 对每个点的邻域进行加权回归\n- 权重随距离增加而减小\n- 适用于探索性数据分析",
            "spline": "样条回归 (Spline Regression):\n- 使用分段多项式拟合数据\n- 在连接点处保持平滑性\n- 适用于平滑曲线拟合"
        }
        explanation = method_explanations.get(method, "未知方法")
        plt.text(0.1, 0.5, explanation, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title('方法说明')
        
        # 6. 拟合优度指标
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # 计算R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_sorted - np.mean(y_sorted)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        metrics_text = f"拟合优度指标:\n\n"
        metrics_text += f"R² = {r_squared:.4f}\n"
        metrics_text += f"均方误差 (MSE) = {np.mean(residuals**2):.4f}\n"
        metrics_text += f"均方根误差 (RMSE) = {np.sqrt(np.mean(residuals**2)):.4f}\n"
        metrics_text += f"平均绝对误差 (MAE) = {np.mean(np.abs(residuals)):.4f}\n"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        plt.title('模型评估指标')
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"nonparametric_regression_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 直接创建完整的结果字典
        result = {
            "success": True,
            "method": method_name,
            "number_of_observations": n_obs,
            "chart_url": f"{PUBLIC_FILE_BASE_URL}/{chart_filename}",
            "model_evaluation": {
                "r_squared": float(r_squared),
                "mse": float(np.mean(residuals**2)),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mae": float(np.mean(np.abs(residuals)))
            },
            "interpretation": _get_nonparametric_interpretation(
                method, r_squared, n_obs
            ),
            "fitted_values": result["fitted_values"],
            "residuals": result["residuals"],
            "parameters": result["parameters"],
            "confidence_interval": result["confidence_interval"]
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"非参数回归分析失败: {str(e)}") from e


def _perform_loess_regression(x, y, frac, it, confidence_level):
    """
    执行LOESS/LOWESS回归
    """
    try:
        # 执行LOWESS
        lowess_result = lowess(y, x, frac=frac, it=it, return_sorted=False)
        
        # 计算残差
        residuals = y - lowess_result
        
        # 计算置信区间（简化方法）
        # 使用残差的标准误差来估计置信区间
        residual_std = np.std(residuals)
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(x) - 2)
        margin_error = t_value * residual_std
        
        # 构建结果
        result = {
            "fitted_values": {
                "x": x.tolist(),
                "y": lowess_result.tolist(),
                "y_original": y.tolist()
            },
            "residuals": residuals.tolist(),
            "parameters": {
                "fraction": frac,
                "iterations": it
            }
        }
        
        # 添加置信区间
        result["confidence_interval"] = {
            "x": x.tolist(),
            "lower": (lowess_result - margin_error).tolist(),
            "upper": (lowess_result + margin_error).tolist()
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"LOESS回归执行失败: {str(e)}")


def _perform_spline_regression(x, y, degree, smooth_factor, confidence_level):
    """
    执行样条回归
    """
    try:
        # 创建样条插值
        if smooth_factor is None:
            # 自动选择平滑因子
            spline = UnivariateSpline(x, y, k=degree)
        else:
            spline = UnivariateSpline(x, y, k=degree, s=smooth_factor)
        
        # 生成更密集的点用于绘图
        x_dense = np.linspace(x.min(), x.max(), 300)
        y_dense = spline(x_dense)
        
        # 在原始x点处的拟合值
        y_fitted = spline(x)
        
        # 计算残差
        residuals = y - y_fitted
        
        # 计算置信区间（简化方法）
        residual_std = np.std(residuals)
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(x) - degree - 1)
        margin_error = t_value * residual_std
        
        # 在密集点上计算置信区间
        y_dense_lower = y_dense - margin_error
        y_dense_upper = y_dense + margin_error
        
        # 构建结果
        result = {
            "fitted_values": {
                "x": x_dense.tolist(),
                "y": y_dense.tolist(),
                "y_original": y.tolist()
            },
            "residuals": residuals.tolist(),
            "parameters": {
                "degree": degree,
                "smooth_factor": smooth_factor
            }
        }
        
        # 添加置信区间
        result["confidence_interval"] = {
            "x": x_dense.tolist(),
            "lower": y_dense_lower.tolist(),
            "upper": y_dense_upper.tolist()
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"样条回归执行失败: {str(e)}")


def _get_nonparametric_interpretation(method, r_squared, n_obs):
    """
    生成非参数回归结果解释
    """
    method_names = {
        "loess": "局部加权回归(LOESS)",
        "spline": "样条回归"
    }
    
    interpretation = [
        f"非参数回归分析完成，使用{method_names.get(method, method)}方法。",
        f"共分析{n_obs}个观测值。",
        f"模型R²值为{r_squared:.4f}，表示模型解释了{r_squared*100:.2f}%的数据变异。"
    ]
    
    if r_squared > 0.7:
        interpretation.append("模型拟合效果较好。")
    elif r_squared > 0.5:
        interpretation.append("模型拟合效果一般。")
    else:
        interpretation.append("模型拟合效果较差，可能需要调整参数或尝试其他方法。")
    
    return " ".join(interpretation)