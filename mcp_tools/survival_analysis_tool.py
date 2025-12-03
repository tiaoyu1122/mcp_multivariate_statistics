import os
import uuid
from typing import List, Optional, Dict, Any
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

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def perform_survival_analysis(
    time_var: List[float] = Field(..., description="时间变量"),
    event_var: List[int] = Field(..., description="事件指示变量（1表示事件发生，0表示删失）"),
    group_var: Optional[List[int]] = Field(None, description="分组变量（用于Kaplan-Meier曲线和Log-rank检验）"),
    covariates: Optional[List[float]] = Field(None, description="协变量数据，所有协变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    covariate_names: Optional[List[str]] = Field(None, description="协变量名称列表"),
    confidence_level: float = Field(0.95, description="置信区间水平，如0.95表示95%置信区间"),
) -> dict:
    """
    执行生存分析，包括Kaplan-Meier生存曲线估计、Log-rank检验和Cox比例风险模型
    
    参数:
    - time_var: 时间变量
    - event_var: 事件指示变量（1表示事件发生，0表示删失）
    - group_var: 分组变量（用于Kaplan-Meier曲线和Log-rank检验）
    - covariates: 协变量数据，所有协变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - covariate_names: 协变量名称列表
    - confidence_level: 置信区间水平，如0.95表示95%置信区间
    
    返回:
    - 包含生存分析结果的字典
    """
    try:
        # 输入验证
        n_obs = len(time_var)
        
        if n_obs == 0:
            raise ValueError("时间变量数据不能为空")
        
        if len(event_var) != n_obs:
            raise ValueError("事件指示变量的长度必须与时间变量相同")
        
        # 检查事件变量是否为0/1值
        unique_events = set(event_var)
        if not unique_events.issubset({0, 1}):
            raise ValueError("事件指示变量只能包含0（删失）和1（事件发生）")
        
        if group_var is not None and len(group_var) != n_obs:
            raise ValueError("分组变量的长度必须与时间变量相同")
        
        # 检查协变量
        n_covariates = 0
        covariates_list = None
        if covariates is not None and covariate_names is not None:
            n_covariates = len(covariate_names)
            if n_covariates > 0:
                # 将一维列表重构为嵌套列表
                covariates_list = []
                for i in range(n_covariates):
                    covariate = covariates[i * n_obs : (i + 1) * n_obs]
                    covariates_list.append(covariate)
                
                for i, covariate in enumerate(covariates_list):
                    if len(covariate) != n_obs:
                        raise ValueError(f"协变量 {covariate_names[i]} 的长度必须与时间变量相同")
        
        # 创建数据框
        data = pd.DataFrame({
            'time': time_var,
            'event': event_var
        })
        
        if group_var is not None:
            data['group'] = group_var
        
        # 添加协变量
        if covariates_list is not None:
            for i, covariate in enumerate(covariates_list):
                data[covariate_names[i]] = covariate
        
        # 统计信息
        n_events = sum(event_var)
        n_censored = n_obs - n_events
        survival_rate = n_censored / n_obs if n_obs > 0 else 0
        
        # 结果字典
        result = {
            "success": True,
            "number_of_observations": n_obs,
            "number_of_events": n_events,
            "number_of_censored": n_censored,
            "censoring_rate": survival_rate,
            "confidence_level": confidence_level
        }
        
        # Kaplan-Meier 生存曲线估计
        km_results = {}
        if group_var is not None:
            # 按组进行Kaplan-Meier分析
            unique_groups = sorted(data['group'].unique())
            km_fitters = {}
            
            plt.figure(figsize=(12, 8))
            
            # 存储各组的结果用于后续Log-rank检验
            group_data = {}
            
            for group in unique_groups:
                group_mask = (data['group'] == group)
                group_data[group] = {
                    'time': data.loc[group_mask, 'time'],
                    'event': data.loc[group_mask, 'event']
                }
                
                kmf = KaplanMeierFitter()
                kmf.fit(group_data[group]['time'], group_data[group]['event'], 
                       label=f'组 {group} (n={group_mask.sum()})')
                km_fitters[group] = kmf
                
                # 添加到KM结果
                km_results[f'group_{group}'] = {
                    'number_of_samples': int(group_mask.sum()),
                    'number_of_events': int(group_data[group]['event'].sum()),
                    'median_survival_time': float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None,
                    'survival_function': {
                        'time': kmf.survival_function_.index.tolist(),
                        'survival_probability': kmf.survival_function_['KM_estimate'].tolist(),
                        'confidence_interval': {
                            'lower': kmf.confidence_interval_['KM_estimate_lower_0.95'].tolist(),
                            'upper': kmf.confidence_interval_['KM_estimate_upper_0.95'].tolist()
                        } if 'KM_estimate_lower_0.95' in kmf.confidence_interval_.columns else None
                    }
                }
                
                # 绘制生存曲线
                kmf.plot_survival_function()
            
            plt.title('Kaplan-Meier 生存曲线')
            plt.xlabel('时间')
            plt.ylabel('生存概率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存Kaplan-Meier图
            km_plot_filename = f"kaplan_meier_survival_{uuid.uuid4().hex}.png"
            km_plot_filepath = os.path.join(OUTPUT_DIR, km_plot_filename)
            plt.savefig(km_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            km_results["plot_url"] = f"{PUBLIC_FILE_BASE_URL}/{km_plot_filename}"
            
            # Log-rank检验
            if len(unique_groups) >= 2:
                # 只比较前两个组
                group1 = unique_groups[0]
                group2 = unique_groups[1]
                
                logrank_result = logrank_test(
                    group_data[group1]['time'],
                    group_data[group2]['time'],
                    group_data[group1]['event'],
                    group_data[group2]['event']
                )
                
                logrank_test_result = {
                    "group_1": int(group1),
                    "group_2": int(group2),
                    "test_statistic": float(logrank_result.test_statistic),
                    "p_value": float(logrank_result.p_value),
                    "is_significant": logrank_result.p_value < 0.05,
                    "degrees_of_freedom": int(logrank_result.degrees_of_freedom)
                }
                
                result["logrank_test"] = logrank_test_result
        else:
            # 整体Kaplan-Meier分析
            kmf = KaplanMeierFitter()
            kmf.fit(data['time'], data['event'])
            
            km_results = {
                'number_of_samples': n_obs,
                'number_of_events': n_events,
                'median_survival_time': float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None,
                'survival_function': {
                    'time': kmf.survival_function_.index.tolist(),
                    'survival_probability': kmf.survival_function_['KM_estimate'].tolist(),
                    'confidence_interval': {
                        'lower': kmf.confidence_interval_['KM_estimate_lower_0.95'].tolist(),
                        'upper': kmf.confidence_interval_['KM_estimate_upper_0.95'].tolist()
                    } if 'KM_estimate_lower_0.95' in kmf.confidence_interval_.columns else None
                }
            }
            
            # 绘制生存曲线
            plt.figure(figsize=(10, 6))
            kmf.plot_survival_function()
            plt.title('Kaplan-Meier 生存曲线')
            plt.xlabel('时间')
            plt.ylabel('生存概率')
            plt.grid(True, alpha=0.3)
            
            # 保存Kaplan-Meier图
            km_plot_filename = f"kaplan_meier_survival_{uuid.uuid4().hex}.png"
            km_plot_filepath = os.path.join(OUTPUT_DIR, km_plot_filename)
            plt.savefig(km_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            km_results["plot_url"] = f"{PUBLIC_FILE_BASE_URL}/{km_plot_filename}"
        
        result["kaplan_meier"] = km_results
        
        # Cox比例风险模型
        if covariates_list is not None and n_covariates > 0:
            try:
                # 准备Cox模型数据
                cox_data = data[['time', 'event'] + covariate_names].copy()
                
                # 拟合Cox模型
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col='time', event_col='event')
                
                # 获取模型摘要
                summary = cph.summary
                
                # 提取系数、p值和风险比
                cox_results = {
                    "number_of_covariates": n_covariates,
                    "covariate_names": covariate_names,
                    "coefficients": summary['coef'].to_dict(),
                    "p_values": summary['p'].to_dict(),
                    "hazard_ratios": summary['exp(coef)'].to_dict(),
                    "confidence_intervals": {
                        var: {
                            "lower": summary.loc[var, 'exp(coef) lower 95%'],
                            "upper": summary.loc[var, 'exp(coef) upper 95%']
                        } for var in covariate_names
                    },
                    "concordance_index": float(cph.concordance_index_),
                    "partial_log_likelihood": float(cph.log_likelihood_),
                    "interpretation": _get_cox_interpretation(summary, cph.concordance_index_)
                }
                
                # 绘制森林图
                plt.figure(figsize=(10, 6))
                hr_values = [summary.loc[var, 'exp(coef)'] for var in covariate_names]
                hr_lower = [summary.loc[var, 'exp(coef) lower 95%'] for var in covariate_names]
                hr_upper = [summary.loc[var, 'exp(coef) upper 95%'] for var in covariate_names]
                
                y_pos = np.arange(len(covariate_names))
                plt.scatter(hr_values, y_pos, color='blue', s=100, label='风险比')
                plt.hlines(y_pos, hr_lower, hr_upper, color='blue', alpha=0.7)
                plt.vlines(1, -0.5, len(covariate_names)-0.5, color='red', linestyle='--', alpha=0.7, label='无效应线(HR=1)')
                
                plt.yticks(y_pos, covariate_names)
                plt.xlabel('风险比 (Hazard Ratio)')
                plt.ylabel('协变量')
                plt.title('Cox模型森林图')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存森林图
                forest_plot_filename = f"cox_forest_plot_{uuid.uuid4().hex}.png"
                forest_plot_filepath = os.path.join(OUTPUT_DIR, forest_plot_filename)
                plt.savefig(forest_plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                cox_results["forest_plot_url"] = f"{PUBLIC_FILE_BASE_URL}/{forest_plot_filename}"
                
                result["cox_proportional_hazards_model"] = cox_results
                
            except Exception as e:
                result["cox_proportional_hazards_model"] = {
                    "error": str(e)
                }
        
        # 添加总体解释
        result["interpretation"] = _get_survival_analysis_interpretation(
            n_obs, n_events, n_censored, survival_rate
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _get_cox_interpretation(summary, concordance_index: float) -> str:
    """
    生成Cox回归结果解释
    """
    interpretation = f"Cox比例风险模型分析完成，一致性指数为{concordance_index:.4f}。"
    
    if concordance_index > 0.7:
        interpretation += "模型预测能力较好。"
    elif concordance_index > 0.6:
        interpretation += "模型预测能力一般。"
    else:
        interpretation += "模型预测能力较弱。"
    
    return interpretation


def _get_survival_analysis_interpretation(
    n_obs: int, 
    n_events: int, 
    n_censored: int, 
    censoring_rate: float
) -> str:
    """
    生成生存分析结果解释
    """
    interpretation = [
        f"生存分析完成，共分析{n_obs}个观测值。",
        f"其中{n_events}个事件发生，{n_censored}个删失。",
        f"删失率为{censoring_rate:.2%}。"
    ]
    
    return " ".join(interpretation)