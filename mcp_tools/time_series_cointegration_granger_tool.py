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

from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnonp
    


def perform_time_series_cointegration_granger_tests(
    time_series_list: List[float] = Field(..., description="时间序列数据，所有时间序列的值按时间序列拼接成一维数组(列优先，先放完时间序列1、再放时间序列2 ...)"),
    series_names: List[str] = Field(..., description="时间序列名称列表"),
    max_lag: int = Field(5, description="Granger因果检验的最大滞后阶数"),
    significance_level: float = Field(0.05, description="显著性水平"),
) -> dict:
    """
    执行时间序列协整检验和Granger因果检验
    
    参数:
    - time_series_list: 时间序列数据，所有时间序列的值按时间序列拼接成一维数组(列优先，先放完时间序列1、再放时间序列2 ...)
    - series_names: 时间序列名称列表
    - max_lag: Granger因果检验的最大滞后阶数，默认为5
    - significance_level: 显著性水平，默认为0.05
    
    返回:
    - 包含协整检验和Granger因果检验结果的字典
    """
    try:
        # 输入验证
        if not time_series_list:
            raise ValueError("时间序列数据不能为空")
        
        n_series = len(series_names)
        if n_series < 2:
            raise ValueError("至少需要两个时间序列才能进行协整检验和Granger因果检验")
        
        # 根据series_names中元素的个数计算数据长度
        if len(time_series_list) % n_series != 0:
            raise ValueError("时间序列数据长度与序列数量不匹配")
        
        n_samples = len(time_series_list) // n_series
        if n_samples == 0:
            raise ValueError("时间序列数据不能为空")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_time_series_list = []
        for i in range(n_series):
            ts_data = time_series_list[i * n_samples:(i + 1) * n_samples]
            reshaped_time_series_list.append(ts_data)
        
        if max_lag < 1:
            raise ValueError("最大滞后阶数必须大于0")
        
        if not (0 < significance_level < 1):
            raise ValueError("显著性水平必须在0到1之间")
        
        # 转换为numpy数组
        ts_array = np.array(reshaped_time_series_list)  # shape: (n_series, n_samples)
        
        # 转换为pandas DataFrame
        df = pd.DataFrame(ts_array.T, columns=series_names)
        
        # 执行协整检验
        cointegration_results = _perform_cointegration_tests(ts_array, series_names, significance_level)
        
        # 执行Granger因果检验
        granger_results = _perform_granger_causality_tests(df, series_names, max_lag, significance_level)
        
        # 生成参数说明图
        param_plot_url = None
        plt.figure(figsize=(10, 6))
        
        # 创建参数说明表格
        param_data = [
            ['参数', '值', '说明'],
            ['最大滞后阶数', str(max_lag), 'Granger因果检验中考虑的最大滞后阶数'],
            ['显著性水平', str(significance_level), '用于判断统计显著性的阈值'],
            ['时间序列数', str(n_series), '分析的时间序列数量'],
            ['样本数量', str(n_samples), '每个时间序列的样本数量']
        ]
        
        table = plt.table(cellText=param_data[1:], colLabels=param_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(param_data)):
            if i == 0:  # 表头
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 2)].set_facecolor('#4CAF50')
            else:
                table[(i, 0)].set_facecolor('#E8F5E8')
                table[(i, 1)].set_facecolor('#F0F0F0')
                table[(i, 2)].set_facecolor('#F0F0F0')
        
        plt.axis('off')
        plt.title('协整检验和Granger因果检验参数说明')
        plt.tight_layout()
        
        # 保存参数说明图
        param_plot_filename = f"cointegration_granger_parameters_{uuid.uuid4().hex}.png"
        param_plot_filepath = os.path.join(OUTPUT_DIR, param_plot_filename)
        plt.savefig(param_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        param_plot_url = f"{PUBLIC_FILE_BASE_URL}/{param_plot_filename}"
        
        # 组织结果
        result = {
            "n_series": n_series,
            "n_samples": n_samples,
            "series_names": series_names,
            "max_lag": max_lag,
            "significance_level": significance_level,
            "cointegration_tests": cointegration_results,
            "granger_causality_tests": granger_results,
            "parameter_plot_url": param_plot_url,
            "interpretation": _get_interpretation(cointegration_results, granger_results, significance_level)
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"协整检验和Granger因果检验失败: {str(e)}") from e


def _perform_cointegration_tests(ts_array: np.ndarray, series_names: List[str], significance_level: float) -> List[Dict]:
    """
    执行协整检验
    
    参数:
    - ts_array: 时间序列数据数组，shape: (n_series, n_samples)
    - series_names: 时间序列名称列表
    - significance_level: 显著性水平
    
    返回:
    - 协整检验结果列表
    """
    n_series = len(series_names)
    results = []
    
    # 对每一对时间序列执行协整检验
    for i in range(n_series):
        for j in range(i + 1, n_series):
            try:
                # 执行Engle-Granger协整检验
                # coint函数返回 (test_statistic, p_value, critical_values)
                coint_t, p_value, crit_value = coint(ts_array[i], ts_array[j])
                
                # 判断是否协整
                is_cointegrated = p_value < significance_level
                
                # 获取1%、5%、10%的临界值
                crit_1 = crit_value['1%']
                crit_5 = crit_value['5%']
                crit_10 = crit_value['10%']
                
                result = {
                    "series_pair": f"{series_names[i]} & {series_names[j]}",
                    "series_1": series_names[i],
                    "series_2": series_names[j],
                    "test_statistic": float(coint_t),
                    "p_value": float(p_value),
                    "critical_values": {
                        "1%": float(crit_1),
                        "5%": float(crit_5),
                        "10%": float(crit_10)
                    },
                    "is_cointegrated": is_cointegrated,
                    "significance_level": significance_level
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "series_pair": f"{series_names[i]} & {series_names[j]}",
                    "series_1": series_names[i],
                    "series_2": series_names[j],
                    "error": f"协整检验失败: {str(e)}"
                })
    
    return results


def _perform_granger_causality_tests(df: pd.DataFrame, series_names: List[str], max_lag: int, significance_level: float) -> List[Dict]:
    """
    执行Granger因果检验
    
    参数:
    - df: 包含时间序列数据的DataFrame
    - series_names: 时间序列名称列表
    - max_lag: 最大滞后阶数
    - significance_level: 显著性水平
    
    返回:
    - Granger因果检验结果列表
    """
    n_series = len(series_names)
    results = []
    
    # 对每一对时间序列执行Granger因果检验
    for i in range(n_series):
        for j in range(n_series):
            if i != j:  # 不对自身进行检验
                try:
                    # 执行Granger因果检验
                    # 原假设: series_j 不是 series_i 的Granger原因
                    granger_result = grangercausalitytests(df[[series_names[i], series_names[j]]], max_lag, verbose=False)
                    
                    # 提取各滞后阶数的检验结果
                    lag_results = []
                    for lag in range(1, max_lag + 1):
                        if lag in granger_result:
                            # 获取F检验的p值
                            f_test = granger_result[lag][0]['ssr_ftest']
                            f_statistic, f_p_value, f_df1, f_df2 = f_test
                            
                            # 判断是否拒绝原假设（即存在Granger因果关系）
                            is_granger_cause = f_p_value < significance_level
                            
                            lag_results.append({
                                "lag": lag,
                                "f_statistic": float(f_statistic),
                                "p_value": float(f_p_value),
                                "degrees_of_freedom": (int(f_df1), int(f_df2)),
                                "is_granger_cause": is_granger_cause
                            })
                    
                    # 找到最显著的滞后阶数
                    best_lag = None
                    if lag_results:
                        best_lag_idx = np.argmin([r['p_value'] for r in lag_results])
                        best_lag = lag_results[best_lag_idx]
                    
                    result = {
                        "causal_pair": f"{series_names[j]} -> {series_names[i]}",
                        "cause_variable": series_names[j],
                        "effect_variable": series_names[i],
                        "max_lag": max_lag,
                        "lag_results": lag_results,
                        "best_lag_result": best_lag,
                        "is_granger_cause": best_lag['is_granger_cause'] if best_lag else False,
                        "significance_level": significance_level
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "causal_pair": f"{series_names[j]} -> {series_names[i]}",
                        "cause_variable": series_names[j],
                        "effect_variable": series_names[i],
                        "error": f"Granger因果检验失败: {str(e)}"
                    })
    
    return results


def _get_interpretation(cointegration_results: List[Dict], granger_results: List[Dict], significance_level: float) -> str:
    """
    根据检验结果提供解释
    
    参数:
    - cointegration_results: 协整检验结果
    - granger_results: Granger因果检验结果
    - significance_level: 显著性水平
    
    返回:
    - 解释文本
    """
    interpretation = f"时间序列协整检验和Granger因果检验完成，显著性水平为{significance_level}。\n"
    
    # 解释协整检验结果
    cointegrated_pairs = [r for r in cointegration_results if 'is_cointegrated' in r and r['is_cointegrated']]
    if cointegrated_pairs:
        interpretation += f"发现{len(cointegrated_pairs)}对协整时间序列，表明这些序列间存在长期均衡关系："
        for pair in cointegrated_pairs:
            interpretation += f"\n- {pair['series_pair']} (p值={pair['p_value']:.4f})"
    else:
        interpretation += "未发现协整的时间序列对，表明序列间可能不存在长期均衡关系。"
    
    # 解释Granger因果检验结果
    granger_causes = [r for r in granger_results if 'is_granger_cause' in r and r['is_granger_cause']]
    if granger_causes:
        interpretation += f"\n\n发现{len(granger_causes)}个Granger因果关系："
        for cause in granger_causes:
            best_lag = cause['best_lag_result']
            if best_lag:
                interpretation += f"\n- {cause['causal_pair']} (最优滞后阶数={best_lag['lag']}, p值={best_lag['p_value']:.4f})"
    else:
        interpretation += "\n\n未发现显著的Granger因果关系。"
    
    return interpretation