import os
import uuid
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from pydantic import Field
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 加载配置
from config import OUTPUT_DIR, PUBLIC_FILE_BASE_URL

# 设置中文字体
from set_font import set_chinese_font
plt = set_chinese_font()


def perform_data_standardization(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("zscore", description="标准化方法: 'zscore'(Z-score标准化), 'minmax'(Min-Max标准化)"),
    feature_range: List[float] = Field([0, 1], description="Min-Max标准化的目标范围，格式为[min, max]"),
) -> dict:
    """
    执行数据标准化（Z-score标准化或Min-Max标准化）
    
    参数:
    - data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表
    - method: 标准化方法 ('zscore', 'minmax')
    - feature_range: Min-Max标准化的目标范围，格式为[min, max]
    
    返回:
    - 包含标准化结果的字典
    """
    try:
        # 输入验证
        if not data:
            raise ValueError("数据不能为空")
        
        n_vars = len(var_names)
        if n_vars == 0:
            raise ValueError("变量数量不能为0")
            
        # 根据var_names中元素的个数计算数据长度
        if len(data) % n_vars != 0:
            raise ValueError("数据长度与变量数量不匹配")
            
        n_samples = len(data) // n_vars
        if n_samples == 0:
            raise ValueError("样本数量不能为0")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_data = []
        for i in range(n_vars):
            var_data = data[i * n_samples:(i + 1) * n_samples]
            reshaped_data.append(var_data)
        
        if method not in ["zscore", "minmax"]:
            raise ValueError("method 参数必须是 'zscore' 或 'minmax'")
        
        if len(feature_range) != 2:
            raise ValueError("feature_range 参数必须包含两个值 [min, max]")
        
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range 中的最小值必须小于最大值")
        
        # 创建 DataFrame
        df = pd.DataFrame({var_names[i]: reshaped_data[i] for i in range(n_vars)})
        
        # 执行标准化
        if method == "zscore":
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(df)
            method_name = "Z-score标准化"
        elif method == "minmax":
            scaler = MinMaxScaler(feature_range=(feature_range[0], feature_range[1]))
            standardized_data = scaler.fit_transform(df)
            method_name = f"Min-Max标准化 (范围: [{feature_range[0]}, {feature_range[1]}])"
        
        # 转换回列表格式
        standardized_data_list = standardized_data.T.tolist()
        
        # 生成结果摘要
        result_summary = []
        for i, var_name in enumerate(var_names):
            original_mean = np.mean(df[var_name])
            original_std = np.std(df[var_name])
            standardized_mean = np.mean(standardized_data[:, i])
            standardized_std = np.std(standardized_data[:, i])
            
            result_summary.append({
                "variable": var_name,
                "original_mean": float(original_mean),
                "original_std": float(original_std),
                "standardized_mean": float(standardized_mean),
                "standardized_std": float(standardized_std)
            })
        
        # 生成可视化图表
        fig, axes = plt.subplots(2, n_vars, figsize=(5*n_vars, 10))
        if n_vars == 1:
            axes = axes.reshape(-1, 1)
            
        for i, var_name in enumerate(var_names):
            # 原始数据分布
            axes[0, i].hist(df[var_name], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].set_title(f'{var_name} (原始数据)')
            axes[0, i].set_xlabel('值')
            axes[0, i].set_ylabel('频率')
            axes[0, i].grid(True, alpha=0.3)
            
            # 标准化后数据分布
            axes[1, i].hist(standardized_data[:, i], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, i].set_title(f'{var_name} ({method_name})')
            axes[1, i].set_xlabel('值')
            axes[1, i].set_ylabel('频率')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"data_standardization_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成结果字典
        result = {
            "success": True,
            "method": method_name,
            "standardized_data": standardized_data_list,
            "variables": var_names,
            "samples_count": n_samples,
            "result_summary": result_summary,
            "chart_url": f"{PUBLIC_FILE_BASE_URL}/{chart_filename}"
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }