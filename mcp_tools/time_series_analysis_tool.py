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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()

# 时间序列分析库
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def perform_time_series_analysis(
    time_series: List[float] = Field(..., description="时间序列数据"),
    time_labels: List[str] = Field(..., description="时间标签列表"),
    model_type: str = Field("auto_arima", description="模型类型: 'auto_arima', 'sarima', 'exponential_smoothing', 'manual_arima'"),
    forecast_steps: int = Field(10, description="预测步数"),
    seasonal_period: Optional[int] = Field(None, description="季节性周期（如月度数据为12，季度数据为4）"),
    order: Optional[List[int]] = Field(None, description="ARIMA模型的(p,d,q)参数，格式为[p,d,q]"),
    seasonal_order: Optional[List[int]] = Field(None, description="季节性ARIMA模型的(P,D,Q,s)参数，格式为[P,D,Q,s]"),
) -> dict:
    """
    执行时间序列分析
    
    参数:
    - time_series: 时间序列数据
    - time_labels: 时间标签列表
    - model_type: 模型类型 ('auto_arima', 'sarima', 'exponential_smoothing', 'manual_arima')
    - forecast_steps: 预测步数
    - seasonal_period: 季节性周期
    - order: ARIMA模型的(p,d,q)参数
    - seasonal_order: 季节性ARIMA模型的(P,D,Q,s)参数
    
    返回:
    - 包含时间序列分析结果的字典
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
        
        # 转换为pandas Series
        ts_series = pd.Series(time_series, index=pd.to_datetime(time_labels) if _is_date_string(time_labels[0]) else time_labels)
        
        # 时间序列分解（如果有季节性周期）
        decomposition_result = None
        decomposition_plot_url = None
        if seasonal_period is not None and seasonal_period > 1:
            try:
                decomposition = seasonal_decompose(ts_series, model='additive', period=seasonal_period)
                
                # 生成分解图
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                decomposition.observed.plot(ax=axes[0], title='原始序列')
                decomposition.trend.plot(ax=axes[1], title='趋势')
                decomposition.seasonal.plot(ax=axes[2], title='季节性')
                decomposition.resid.plot(ax=axes[3], title='残差')
                plt.tight_layout()
                
                # 保存分解图
                decomposition_plot_filename = f"ts_decomposition_{uuid.uuid4().hex}.png"
                decomposition_plot_filepath = os.path.join(OUTPUT_DIR, decomposition_plot_filename)
                plt.savefig(decomposition_plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                decomposition_plot_url = f"{PUBLIC_FILE_BASE_URL}/{decomposition_plot_filename}"
                
                decomposition_result = {
                    "trend": decomposition.trend.dropna().tolist(),
                    "seasonal": decomposition.seasonal.tolist(),
                    "residual": decomposition.resid.dropna().tolist()
                }
            except Exception as e:
                print(f"时间序列分解失败: {e}")
        
        # 平稳性检验（ADF检验）
        adf_result = adfuller(time_series)
        is_stationary = adf_result[1] < 0.05  # p值小于0.05认为是平稳的
        
        # ACF和PACF图
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            plot_acf(time_series, ax=axes[0], lags=min(20, n_samples//4))
            plot_pacf(time_series, ax=axes[1], lags=min(20, n_samples//4))
            axes[0].set_title('自相关函数 (ACF)')
            axes[1].set_title('偏自相关函数 (PACF)')
            plt.tight_layout()
            
            # 保存ACF/PACF图
            acf_pacf_plot_filename = f"acf_pacf_{uuid.uuid4().hex}.png"
            acf_pacf_plot_filepath = os.path.join(OUTPUT_DIR, acf_pacf_plot_filename)
            plt.savefig(acf_pacf_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            acf_pacf_plot_url = f"{PUBLIC_FILE_BASE_URL}/{acf_pacf_plot_filename}"
        except Exception as e:
            print(f"ACF/PACF图生成失败: {e}")
            acf_pacf_plot_url = None
        
        # 模型拟合和预测
        model_result = None
        forecast_result = None
        model_diagnostics_plot_url = None
        
        if model_type == "auto_arima":
            # 自动选择ARIMA模型
            model_result, forecast_result = _fit_auto_arima_model(ts_series, forecast_steps, seasonal_period)
        elif model_type == "sarima":
            # 季节性ARIMA模型
            model_result, forecast_result = _fit_sarima_model(ts_series, forecast_steps, seasonal_period, seasonal_order)
        elif model_type == "exponential_smoothing":
            # 指数平滑模型
            model_result, forecast_result = _fit_exponential_smoothing_model(ts_series, forecast_steps, seasonal_period)
        elif model_type == "manual_arima":
            # 手动指定参数的ARIMA模型
            if order is None:
                raise ValueError("使用manual_arima模型类型时必须指定order参数")
            model_result, forecast_result = _fit_manual_arima_model(ts_series, forecast_steps, order)
        
        # 生成模型诊断图
        if model_result is not None:
            try:
                # 残差诊断图
                if hasattr(model_result, 'resid') and model_result.resid is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # 残差时间序列图
                    axes[0, 0].plot(model_result.resid)
                    axes[0, 0].set_title('残差时间序列')
                    axes[0, 0].grid(True)
                    
                    # 残差直方图
                    axes[0, 1].hist(model_result.resid, bins=20, alpha=0.7)
                    axes[0, 1].set_title('残差直方图')
                    axes[0, 1].grid(True)
                    
                    # 残差Q-Q图
                    stats.probplot(model_result.resid, dist="norm", plot=axes[1, 0])
                    axes[1, 0].set_title('残差Q-Q图')
                    axes[1, 0].grid(True)
                    
                    # 残差ACF图
                    plot_acf(model_result.resid, ax=axes[1, 1], lags=min(20, len(model_result.resid)//4))
                    axes[1, 1].set_title('残差ACF')
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    
                    # 保存模型诊断图
                    diagnostics_plot_filename = f"ts_diagnostics_{uuid.uuid4().hex}.png"
                    diagnostics_plot_filepath = os.path.join(OUTPUT_DIR, diagnostics_plot_filename)
                    plt.savefig(diagnostics_plot_filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    model_diagnostics_plot_url = f"{PUBLIC_FILE_BASE_URL}/{diagnostics_plot_filename}"
            except Exception as e:
                print(f"模型诊断图生成失败: {e}")
        
        # 生成预测图
        forecast_plot_url = None
        if forecast_result is not None:
            try:
                plt.figure(figsize=(12, 6))
                
                # 绘制历史数据
                plt.plot(ts_series.index, ts_series.values, label='历史数据', color='blue')
                
                # 绘制预测值
                if hasattr(forecast_result, 'predicted_mean'):
                    # ARIMA预测
                    forecast_mean = forecast_result.predicted_mean
                    forecast_index = forecast_mean.index
                    plt.plot(forecast_index, forecast_mean, label='预测值', color='red', linestyle='--')
                    
                    # 绘制置信区间
                    if hasattr(forecast_result, 'conf_int'):
                        conf_int = forecast_result.conf_int()
                        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
                else:
                    # 指数平滑预测
                    forecast_index = range(len(ts_series), len(ts_series) + len(forecast_result))
                    plt.plot(forecast_index, forecast_result, label='预测值', color='red', linestyle='--')
                
                plt.xlabel('时间')
                plt.ylabel('值')
                plt.title('时间序列预测')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # 保存预测图
                forecast_plot_filename = f"ts_forecast_{uuid.uuid4().hex}.png"
                forecast_plot_filepath = os.path.join(OUTPUT_DIR, forecast_plot_filename)
                plt.savefig(forecast_plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                forecast_plot_url = f"{PUBLIC_FILE_BASE_URL}/{forecast_plot_filename}"
            except Exception as e:
                print(f"预测图生成失败: {e}")
        
        # 组织结果
        result = {
            "n_samples": n_samples,
            "time_labels": time_labels,
            "is_stationary": is_stationary,
            "adf_test": {
                "test_statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "critical_values": adf_result[4]
            },
            "decomposition": decomposition_result,
            "model_type": model_type,
            "model_summary": str(model_result.summary()) if model_result is not None and hasattr(model_result, 'summary') else None,
            "forecast": {
                "steps": forecast_steps,
                "predicted_values": forecast_result.tolist() if isinstance(forecast_result, (np.ndarray, pd.Series)) else None,
                "mean_forecast": forecast_result.predicted_mean.tolist() if hasattr(forecast_result, 'predicted_mean') else None,
                "confidence_intervals": forecast_result.conf_int().tolist() if hasattr(forecast_result, 'conf_int') else None
            } if forecast_result is not None else None,
            "plots": {
                "decomposition_plot_url": decomposition_plot_url,
                "acf_pacf_plot_url": acf_pacf_plot_url,
                "model_diagnostics_plot_url": model_diagnostics_plot_url,
                "forecast_plot_url": forecast_plot_url
            },
            "interpretation": _get_interpretation(is_stationary, adf_result, model_result, forecast_result)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"时间序列分析失败: {str(e)}") from e


def _fit_auto_arima_model(ts_series: pd.Series, forecast_steps: int, seasonal_period: Optional[int]):
    """
    自动选择ARIMA模型
    """
    try:
        # 简单的自动模型选择（实际应用中可以更复杂）
        # 这里我们使用(1,1,1)作为默认参数
        order = (1, 1, 1)
        if seasonal_period is not None and seasonal_period > 1:
            # 如果有季节性，使用季节性ARIMA
            seasonal_order = (1, 1, 1, seasonal_period)
            model = ARIMA(ts_series, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(ts_series, order=order)
        
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        return model_fit, forecast
    except Exception as e:
        print(f"自动ARIMA模型拟合失败: {e}")
        return None, None


def _fit_sarima_model(ts_series: pd.Series, forecast_steps: int, seasonal_period: Optional[int], seasonal_order: Optional[List[int]]):
    """
    拟合季节性ARIMA模型
    """
    try:
        order = (1, 1, 1)  # 默认非季节性参数
        if seasonal_order is None:
            if seasonal_period is not None and seasonal_period > 1:
                seasonal_order = (1, 1, 1, seasonal_period)
            else:
                seasonal_order = (0, 0, 0, 0)
        
        model = ARIMA(ts_series, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        return model_fit, forecast
    except Exception as e:
        print(f"季节性ARIMA模型拟合失败: {e}")
        return None, None


def _fit_exponential_smoothing_model(ts_series: pd.Series, forecast_steps: int, seasonal_period: Optional[int]):
    """
    拟合指数平滑模型
    """
    try:
        if seasonal_period is not None and seasonal_period > 1:
            model = ExponentialSmoothing(ts_series, trend='add', seasonal='add', seasonal_periods=seasonal_period)
        else:
            model = ExponentialSmoothing(ts_series, trend='add')
        
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        return model_fit, forecast
    except Exception as e:
        print(f"指数平滑模型拟合失败: {e}")
        return None, None


def _fit_manual_arima_model(ts_series: pd.Series, forecast_steps: int, order: List[int]):
    """
    拟合手动指定参数的ARIMA模型
    """
    try:
        model = ARIMA(ts_series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        return model_fit, forecast
    except Exception as e:
        print(f"手动ARIMA模型拟合失败: {e}")
        return None, None


def _is_date_string(s: str) -> bool:
    """
    判断字符串是否为日期格式
    """
    try:
        pd.to_datetime(s)
        return True
    except:
        return False


def _get_interpretation(is_stationary: bool, adf_result: tuple, model_result: Any, forecast_result: Any) -> str:
    """
    根据时间序列分析结果提供解释
    """
    interpretation = "时间序列分析完成。"
    
    # 平稳性解释
    if is_stationary:
        interpretation += "\n序列是平稳的（ADF检验p值 < 0.05）。"
    else:
        interpretation += "\n序列是非平稳的（ADF检验p值 >= 0.05），可能需要差分处理。"
    
    # 模型拟合解释
    if model_result is not None:
        interpretation += f"\n模型已成功拟合。"
        
        # 预测解释
        if forecast_result is not None:
            interpretation += f"\n已生成未来{len(forecast_result) if isinstance(forecast_result, (np.ndarray, pd.Series)) else 10}期的预测。"
    
    return interpretation