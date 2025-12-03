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

from semopy import Model, Optimizer
from semopy import semplot

def perform_sem(
    data: List[float] = Field(..., description="观测数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    model_description: str = Field(..., description="模型描述字符串，使用semopy语法定义测量模型和结构模型"),
    group_var: Optional[str] = Field(None, description="分组变量名称，用于多组分析"),
) -> dict:
    """
    执行结构方程模型(SEM)分析
    
    参数:
    - data: 观测数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - model_description: 模型描述字符串，使用semopy语法定义测量模型和结构模型
    - group_var: 分组变量名称，用于多组分析
    
    示例model_description:
    '''
    # 测量模型
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    
    # 结构模型
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
    
    # 方差和协方差
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y8 ~~ y4 + y6
    '''
    """    
    # 输入验证
    n_vars = len(var_names)
    if n_vars == 0:
        raise ValueError("变量名称不能为空")
    
    if len(data) % n_vars != 0:
        raise ValueError("数据长度与变量数量不匹配")
    
    n_obs = len(data) // n_vars
    if n_obs == 0:
        raise ValueError("数据不能为空")
    
    try:
        # 将一维数组重构为嵌套列表
        data_nested = []
        for i in range(n_vars):
            var_data = data[i * n_obs : (i + 1) * n_obs]
            data_nested.append(var_data)
        
        # 创建 DataFrame
        df = pd.DataFrame({var_names[i]: data_nested[i] for i in range(n_vars)})
        
        # 创建SEM模型
        model = Model(model_description)
        
        # 拟合模型
        result = model.fit(df)
        
        # 获取参数估计结果
        params = model.inspect()
        
        # # 获取模型摘要
        # optimizer = Optimizer()
        # opt_result = optimizer.optimize(model)
        # summary = opt_result.summarize()
        
        # 获取模型拟合指标
        try:
            fit_measures = model.fit_measures()
        except:
            fit_measures = None
        
        # 生成路径图
        plt.figure(figsize=(12, 10))
        try:
            # 生成SEM路径图
            semplot(model, filename=None)  # 不保存到文件，只用于绘图
        except Exception as e:
            # 如果无法生成路径图，绘制简单的变量关系图
            _draw_simple_sem_plot(model_description, var_names)
        
        plt.title('结构方程模型路径图')
        plt.tight_layout()
        
        # 保存路径图
        plot_filename = f"sem_path_plot_{uuid.uuid4().hex}.png"
        plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        
        # 生成参数估计表格图
        plt.figure(figsize=(12, 8))
        param_table_data = params.head(20)  # 只显示前20个参数
        
        # 创建表格
        table = plt.table(cellText=param_table_data.values,
                         colLabels=param_table_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.axis('off')
        plt.title('参数估计结果')
        plt.tight_layout()
        
        # 保存参数表格图
        param_plot_filename = f"sem_param_table_{uuid.uuid4().hex}.png"
        param_plot_filepath = os.path.join(OUTPUT_DIR, param_plot_filename)
        plt.savefig(param_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        param_plot_url = f"{PUBLIC_FILE_BASE_URL}/{param_plot_filename}"
        
        # 组织结果
        result_dict = {
            "model_type": "结构方程模型(SEM)",
            "number_of_observations": n_obs,
            "number_of_variables": n_vars,
            "variable_names": var_names,
            "model_description": model_description,
            "fit_successful": result is not None,
            "parameters": params.to_dict('records') if params is not None else [],
            "fit_measures": fit_measures.to_dict() if fit_measures is not None else {},
            "path_plot_url": plot_url,
            "parameter_table_url": param_plot_url,
            "interpretation": _get_sem_interpretation(params, fit_measures)
        }
        
        return result_dict
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"结构方程模型分析失败: {str(e)}") from e


def _draw_simple_sem_plot(model_description: str, var_names: List[str]):
    """
    绘制简化的SEM路径图
    """
    # 解析模型描述，提取变量关系
    lines = model_description.strip().split('\n')
    relations = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if '=~' in line:  # 测量模型
            parts = line.split('=~')
            latent = parts[0].strip()
            indicators = [ind.strip() for ind in parts[1].split('+')]
            for indicator in indicators:
                relations.append((latent, indicator, 'measurement'))
        elif '~' in line:  # 结构模型
            parts = line.split('~')
            dependent = parts[0].strip()
            independents = [ind.strip() for ind in parts[1].split('+')]
            for independent in independents:
                relations.append((independent, dependent, 'structural'))
    
    # 绘制节点和边
    plt.figure(figsize=(10, 8))
    
    # 简单布局 - 潜变量在上方，观测变量在下方
    latent_vars = list(set([rel[0] for rel in relations if rel[2] == 'measurement'] + 
                          [rel[1] for rel in relations if rel[2] == 'measurement']))
    observed_vars = list(set(var_names) - set(latent_vars))
    
    # 节点位置
    positions = {}
    n_latent = len(latent_vars)
    n_observed = len(observed_vars)
    
    # 潜变量位置
    for i, lv in enumerate(latent_vars):
        x = i * (10 / max(n_latent, 1)) if n_latent > 1 else 5
        positions[lv] = (x, 6)
    
    # 观测变量位置
    for i, ov in enumerate(observed_vars):
        x = i * (10 / max(n_observed, 1)) if n_observed > 1 else 5
        positions[ov] = (x, 2)
    
    # 绘制节点
    for var, (x, y) in positions.items():
        if var in latent_vars:
            plt.scatter(x, y, s=1000, facecolor='lightblue', edgecolor='black', linewidth=2)
            plt.text(x, y, var, ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            plt.scatter(x, y, s=800, facecolor='lightgreen', edgecolor='black', linewidth=2)
            plt.text(x, y, var, ha='center', va='center', fontsize=9)
    
    # 绘制边
    for source, target, rel_type in relations:
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            
            # 添加一些弯曲避免线条重叠
            if rel_type == 'measurement':
                plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
            else:  # structural
                plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.xlim(-1, 11)
    plt.ylim(0, 8)
    plt.axis('off')


def _get_sem_interpretation(params: pd.DataFrame, fit_measures: pd.DataFrame) -> str:
    """
    根据分析结果提供解释
    """
    interpretation = "结构方程模型分析完成。\n"
    
    if params is not None and not params.empty:
        # 确保 'p-value' 和 'Type' 列存在且没有缺失值
        if 'p-value' in params.columns and 'Type' in params.columns:
            # 过滤掉p-value为NaN或非数值的行
            valid_params = params.dropna(subset=['p-value'])
            # 确保p-value列为数值类型
            valid_params = valid_params[pd.to_numeric(valid_params['p-value'], errors='coerce').notna()]
            
            if not valid_params.empty:
                significant_params = valid_params[
                    (pd.to_numeric(valid_params['p-value']) < 0.05) & 
                    (valid_params['Type'] == 'REG')
                ]
                interpretation += f"模型中共有 {len(significant_params)} 个显著的路径系数。\n"
    
    if fit_measures is not None and not fit_measures.empty:
        # 常见的拟合指标
        if 'chisqr' in fit_measures:
            chisqr_val = fit_measures['chisqr']
            # 确保值是数值类型
            if pd.api.types.is_number(chisqr_val):
                interpretation += f"卡方值为 {chisqr_val:.4f}。"
        if 'cfi' in fit_measures:
            cfi = fit_measures['cfi']
            # 确保值是数值类型
            if pd.api.types.is_number(cfi):
                interpretation += f"CFI为 {cfi:.4f}，"
                if cfi > 0.95:
                    interpretation += "模型拟合非常好。"
                elif cfi > 0.90:
                    interpretation += "模型拟合良好。"
                else:
                    interpretation += "模型拟合需要改进。"
        if 'rmsea' in fit_measures:
            rmsea = fit_measures['rmsea']
            # 确保值是数值类型
            if pd.api.types.is_number(rmsea):
                interpretation += f"RMSEA为 {rmsea:.4f}，"
                if rmsea < 0.05:
                    interpretation += "模型拟合非常好。"
                elif rmsea < 0.08:
                    interpretation += "模型拟合良好。"
                else:
                    interpretation += "模型拟合较差。"
    
    return interpretation