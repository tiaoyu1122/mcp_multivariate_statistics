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

# 时间序列分析库
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def perform_time_series_preprocessing_tests(
    time_series: List[float] = Field(..., description="时间序列数据"),
    time_labels: List[str] = Field(..., description="时间标签列表"),
    seasonal_period: Optional[int] = Field(None, description="季节性周期（如月度数据为12，季度数据为4）"),
    test_types: List[str] = Field(["adf", "kpss", "normality", "autocorrelation"], description="要执行的检验类型列表: 'adf'(ADF平稳性检验), 'kpss'(KPSS平稳性检验), 'normality'(正态性检验), 'autocorrelation'(自相关检验)"),
) -> dict:
    """
    执行时间序列预处理检验，用于判断时间序列是否适合建模
    
    参数:
    - time_series: 时间序列数据
    - time_labels: 时间标签列表
    - seasonal_period: 季节性周期
    - test_types: 要执行的检验类型列表
    
    返回:
    - 包含各种预处理检验结果的字典
    """
    try:
        # 输入验证
        if not time_series:
            raise ValueError("时间序列数据不能为空")
        
        n_samples = len(time_series)
        if n_samples < 10:
            raise ValueError("时间序列数据至少需要10个观测值")
        
        # 处理时间标签
        if time_labels is not None:
            if len(time_labels) != n_samples:
                raise ValueError("时间标签数量与数据不匹配")
        else:
            # 如果没有提供时间标签，则生成默认标签
            time_labels = [f"t{i+1}" for i in range(n_samples)]
        
        # 转换为numpy数组
        ts_array = np.array(time_series)
        
        # 转换为pandas Series
        ts_series = pd.Series(time_series, index=pd.to_datetime(time_labels) if _is_date_string(time_labels[0]) else time_labels)
        
        # 初始化结果字典
        results = {
            "n_samples": n_samples,
            "time_labels": time_labels,
            "test_types": test_types
        }
        
        # 执行ADF平稳性检验
        if "adf" in test_types:
            results["adf_test"] = _perform_adf_test(ts_array)
        
        # 执行KPSS平稳性检验
        if "kpss" in test_types:
            results["kpss_test"] = _perform_kpss_test(ts_array)
        
        # 执行正态性检验
        if "normality" in test_types:
            results["normality_test"] = _perform_normality_test(ts_array)
        
        # 执行自相关检验
        if "autocorrelation" in test_types:
            acf_result, pacf_result, plot_url = _perform_autocorrelation_test(ts_array, n_samples)
            results["autocorrelation_test"] = {
                "acf": acf_result,
                "pacf": pacf_result,
                "acf_pacf_plot_url": plot_url
            }
        
        # 季节性检验（如果有季节性周期）
        if seasonal_period is not None and seasonal_period > 1 and "seasonality" in test_types:
            results["seasonality_test"] = _perform_seasonality_test(ts_series, seasonal_period)
        
        # 添加解释
        results["interpretation"] = _get_interpretation(results)
        
        return results
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"时间序列预处理检验失败: {str(e)}") from e


def _perform_adf_test(time_series: np.ndarray) -> dict:
    """
    执行ADF平稳性检验
    """
    try:
        # ADF检验
        adf_result = adfuller(time_series)
        
        # 解释结果
        is_stationary = adf_result[1] < 0.05  # p值小于0.05认为是平稳的
        
        return {
            "test_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "is_stationary": is_stationary,
            "used_lags": int(adf_result[2]),
            "observations": int(adf_result[3]),
            "description": "ADF检验用于检验时间序列的平稳性。原假设：序列存在单位根（非平稳）；备择假设：序列平稳。"
        }
    except Exception as e:
        return {"error": f"ADF检验失败: {str(e)}"}


def _perform_kpss_test(time_series: np.ndarray) -> dict:
    """
    执行KPSS平稳性检验
    """
    try:
        # KPSS检验
        kpss_result = kpss(time_series, nlags="auto")
        
        # 解释结果
        is_stationary = kpss_result[1] > 0.05  # p值大于0.05认为是平稳的
        
        return {
            "test_statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
            "is_stationary": is_stationary,
            "used_lags": int(kpss_result[2]),
            "description": "KPSS检验用于检验时间序列的平稳性。原假设：序列平稳；备择假设：序列存在单位根（非平稳）。"
        }
    except Exception as e:
        return {"error": f"KPSS检验失败: {str(e)}"}


def _perform_normality_test(time_series: np.ndarray) -> dict:
    """
    执行正态性检验
    """
    try:
        # Shapiro-Wilk正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(time_series)
        
        # Jarque-Bera正态性检验
        jb_stat, jb_p = stats.jarque_bera(time_series)
        
        # 偏度和峰度
        skewness = stats.skew(time_series)
        kurtosis = stats.kurtosis(time_series)
        
        # 解释结果
        is_normal = shapiro_p > 0.05  # p值大于0.05认为是正态分布
        
        return {
            "shapiro_wilk": {
                "test_statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "is_normal": is_normal
            },
            "jarque_bera": {
                "test_statistic": float(jb_stat),
                "p_value": float(jb_p),
                "is_normal": jb_p > 0.05
            },
            "skewness": float(skewness),  # 偏度：0为对称，正为右偏，负为左偏
            "kurtosis": float(kurtosis),  # 峰度：0为正态峰度，正为尖峰，负为平峰
            "description": "正态性检验用于检验时间序列是否服从正态分布。对于某些时间序列模型，残差需要满足正态性假设。"
        }
    except Exception as e:
        return {"error": f"正态性检验失败: {str(e)}"}


def _perform_autocorrelation_test(time_series: np.ndarray, n_samples: int) -> tuple:
    """
    执行自相关检验
    """
    try:
        # 计算ACF和PACF
        # 由于statsmodels的ACF计算需要pandas Series，我们直接使用numpy计算
        
        # 生成ACF/PACF图
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plot_acf(time_series, ax=axes[0], lags=min(20, n_samples//4))
        plot_pacf(time_series, ax=axes[1], lags=min(20, n_samples//4))
        axes[0].set_title('自相关函数 (ACF)')
        axes[1].set_title('偏自相关函数 (PACF)')
        plt.tight_layout()
        
        # 保存ACF/PACF图
        acf_pacf_plot_filename = f"preprocessing_acf_pacf_{uuid.uuid4().hex}.png"
        acf_pacf_plot_filepath = os.path.join(OUTPUT_DIR, acf_pacf_plot_filename)
        plt.savefig(acf_pacf_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        acf_pacf_plot_url = f"{PUBLIC_FILE_BASE_URL}/{acf_pacf_plot_filename}"
        
        return {"description": "自相关函数(ACF)和偏自相关函数(PACF)用于识别时间序列的自相关结构，帮助选择ARIMA模型的参数。"}, \
               {"description": "偏自相关函数(PACF)用于识别时间序列的偏自相关结构，帮助选择ARIMA模型的参数。"}, \
               acf_pacf_plot_url
               
    except Exception as e:
        return {"error": f"自相关检验失败: {str(e)}"}, \
               {"error": f"偏自相关检验失败: {str(e)}"}, \
               None


def _perform_seasonality_test(time_series: pd.Series, seasonal_period: int) -> dict:
    """
    执行季节性检验
    """
    try:
        # 季节性分解
        decomposition = seasonal_decompose(time_series, model='additive', period=seasonal_period)
        
        # 计算季节性强度
        seasonal_var = np.var(decomposition.seasonal.dropna())
        total_var = np.var(time_series)
        seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
        
        # 生成季节性图
        plt.figure(figsize=(12, 8))
        
        # 绘制原始序列和季节性成分
        plt.subplot(2, 1, 1)
        plt.plot(time_series.index, time_series.values, label='原始序列')
        plt.title('原始时间序列')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(decomposition.seasonal.index, decomposition.seasonal.values, label='季节性成分', color='orange')
        plt.title(f'季节性成分 (周期={seasonal_period})')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存季节性图
        seasonality_plot_filename = f"preprocessing_seasonality_{uuid.uuid4().hex}.png"
        seasonality_plot_filepath = os.path.join(OUTPUT_DIR, seasonality_plot_filename)
        plt.savefig(seasonality_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        seasonality_plot_url = f"{PUBLIC_FILE_BASE_URL}/{seasonality_plot_filename}"
        
        return {
            "seasonal_strength": float(seasonal_strength),
            "seasonal_period": seasonal_period,
            "has_seasonality": seasonal_strength > 0.1,  # 简单的阈值判断
            "seasonal_plot_url": seasonality_plot_url,
            "description": "季节性检验用于识别时间序列中是否存在季节性模式。季节性强度大于0.1通常认为存在明显的季节性。"
        }
    except Exception as e:
        return {"error": f"季节性检验失败: {str(e)}"}


def _is_date_string(s: str) -> bool:
    """
    判断字符串是否为日期格式
    """
    try:
        pd.to_datetime(s)
        return True
    except:
        return False


def _get_interpretation(results: dict) -> str:
    """
    根据检验结果提供解释
    """
    interpretation = "时间序列预处理检验完成。\n\n"
    
    # 平稳性检验解释
    if "adf_test" in results and "kpss_test" in results:
        adf_stationary = results["adf_test"].get("is_stationary", False)
        kpss_stationary = results["kpss_test"].get("is_stationary", False)
        
        if adf_stationary and kpss_stationary:
            interpretation += "ADF和KPSS检验均表明序列是平稳的，可以直接用于建模。\n"
        elif not adf_stationary and not kpss_stationary:
            interpretation += "ADF和KPSS检验均表明序列是非平稳的，需要进行差分处理。\n"
        elif adf_stationary and not kpss_stationary:
            interpretation += "ADF检验认为序列平稳，但KPSS检验认为序列非平稳，建议进一步检查或进行差分处理。\n"
        else:
            interpretation += "ADF检验认为序列非平稳，但KPSS检验认为序列平稳，建议进一步检查或进行差分处理。\n"
    
    # 正态性检验解释
    if "normality_test" in results:
        is_normal = results["normality_test"]["shapiro_wilk"]["is_normal"]
        if is_normal:
            interpretation += "序列服从正态分布，满足某些时间序列模型的假设。\n"
        else:
            interpretation += "序列不服从正态分布，可能需要进行变换处理。\n"
    
    # 季节性检验解释
    if "seasonality_test" in results:
        has_seasonality = results["seasonality_test"].get("has_seasonality", False)
        seasonal_strength = results["seasonality_test"].get("seasonal_strength", 0)
        if has_seasonality:
            interpretation += f"序列存在明显的季节性(pattern)，季节性强度为{seasonal_strength:.2f}，建议使用季节性模型。\n"
        else:
            interpretation += f"序列季节性较弱，季节性强度为{seasonal_strength:.2f}。\n"
    
    return interpretation