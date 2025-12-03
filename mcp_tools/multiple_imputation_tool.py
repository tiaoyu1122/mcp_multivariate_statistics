import os
import uuid
from typing import List, Dict, Any, Optional, Union
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

from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge


def perform_multiple_imputation(
    data: List[Union[float, None]] = Field(..., description="数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)，缺失值用null表示"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("mice", description="插补方法: 'mice'(多重插补), 'mean'(均值插补), 'median'(中位数插补), 'mode'(众数插补), 'knn'(K近邻插补)"),
    n_imputations: int = Field(5, description="插补次数，仅对'mice'方法有效"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
) -> dict:
    """
    执行缺失值多重插补分析
    
    参数:
    - data: 数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)，缺失值用null表示
    - var_names: 变量名称列表
    - method: 插补方法 ('mice', 'mean', 'median', 'mode', 'knn')
    - n_imputations: 插补次数，仅对'mice'方法有效
    - random_state: 随机种子，用于结果可重现
    
    返回:
    - 包含缺失值插补结果的字典
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
        
        if method not in ["mice", "mean", "median", "mode", "knn"]:
            raise ValueError("method 参数必须是 'mice', 'mean', 'median', 'mode', 'knn' 之一")
        
        if n_imputations < 1:
            raise ValueError("n_imputations 必须大于等于1")
        
        # 创建 DataFrame 并处理缺失值
        # 将 None 转换为 np.nan
        processed_data = []
        for var_data in reshaped_data:
            processed_var = [np.nan if val is None else val for val in var_data]
            processed_data.append(processed_var)
        
        df = pd.DataFrame({var_names[i]: processed_data[i] for i in range(n_vars)})
        
        # 统计缺失值信息
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / n_samples) * 100
        
        missing_info = []
        total_missing = 0
        for var_name in var_names:
            count = missing_counts[var_name]
            percentage = missing_percentages[var_name]
            missing_info.append({
                "variable": var_name,
                "missing_count": int(count),
                "missing_percentage": float(percentage)
            })
            total_missing += count
        
        # 如果没有缺失值，直接返回原数据
        if total_missing == 0:
            result = {
                "success": True,
                "message": "数据中没有缺失值，无需插补",
                "original_data": processed_data,
                "imputed_data": processed_data,
                "variables": var_names,
                "samples_count": n_samples,
                "missing_info": missing_info,
                "method": method,
                "imputation_details": "无缺失值，未执行插补"
            }
            return result
        
        # 执行插补
        imputed_datasets = []
        
        if method == "mice":
            # 多重插补（MICE - Multiple Imputation by Chained Equations）
            imputers = []
            for i in range(n_imputations):
                # 使用不同的随机种子
                seed = random_state + i if random_state is not None else i
                imputer = IterativeImputer(
                    random_state=seed,
                    sample_posterior=True,  # 启用后验采样以获得不确定性
                    max_iter=10
                )
                imputed_data = imputer.fit_transform(df)
                imputed_datasets.append(imputed_data)
                imputers.append(imputer)
                
        elif method == "mean":
            # 均值插补
            imputer = SimpleImputer(strategy="mean")
            imputed_data = imputer.fit_transform(df)
            imputed_datasets.append(imputed_data)
            
        elif method == "median":
            # 中位数插补
            imputer = SimpleImputer(strategy="median")
            imputed_data = imputer.fit_transform(df)
            imputed_datasets.append(imputed_data)
            
        elif method == "mode":
            # 众数插补
            imputer = SimpleImputer(strategy="most_frequent")
            imputed_data = imputer.fit_transform(df)
            imputed_datasets.append(imputed_data)
            
        elif method == "knn":
            # K近邻插补
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df)
            imputed_datasets.append(imputed_data)
        
        # 转换回列表格式
        imputed_data_lists = []
        for imputed_data in imputed_datasets:
            imputed_data_list = imputed_data.T.tolist()
            imputed_data_lists.append(imputed_data_list)
        
        # 如果是MICE方法，计算汇总统计量
        pooled_results = None
        if method == "mice" and len(imputed_datasets) > 1:
            # 计算各次插补的均值和方差
            imputed_arrays = [np.array(dataset) for dataset in imputed_datasets]
            
            # 计算均值的平均值
            mean_of_means = np.mean([np.mean(arr, axis=1) for arr in imputed_arrays], axis=0)
            
            # 计算总方差（包含插补间方差和插补内方差）
            # 插补内方差：各次插补数据方差的平均值
            within_imputation_variances = [np.var(arr, axis=1) for arr in imputed_arrays]
            average_within_variance = np.mean(within_imputation_variances, axis=0)
            
            # 插补间方差：各变量均值的方差乘以(1+1/m)
            between_imputation_variances = np.var([np.mean(arr, axis=1) for arr in imputed_arrays], axis=0)
            total_variances = average_within_variance + (1 + 1/n_imputations) * between_imputation_variances
            
            pooled_results = {
                "mean_of_means": mean_of_means.tolist(),
                "average_within_variance": average_within_variance.tolist(),
                "between_imputation_variance": between_imputation_variances.tolist(),
                "total_variances": total_variances.tolist()
            }
        
        # 生成缺失值模式图
        plt.figure(figsize=(12, 8))
        
        # 缺失值模式热力图
        plt.subplot(2, 2, 1)
        missing_matrix = df.isnull().astype(int).values
        plt.imshow(missing_matrix, cmap='Blues', aspect='auto')
        plt.xlabel('样本')
        plt.ylabel('变量')
        plt.title('缺失值模式（蓝色表示存在值，白色表示缺失值）')
        plt.colorbar(shrink=0.8)
        
        # 缺失值百分比柱状图
        plt.subplot(2, 2, 2)
        missing_percentages_values = [info["missing_percentage"] for info in missing_info]
        bars = plt.bar(range(n_vars), missing_percentages_values, color='skyblue')
        plt.xlabel('变量')
        plt.ylabel('缺失值百分比 (%)')
        plt.title('各变量缺失值比例')
        plt.xticks(range(n_vars), var_names, rotation=45, ha='right')
        # 在柱状图上添加数值标签
        for i, (bar, pct) in enumerate(zip(bars, missing_percentages_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 插补前后数据分布对比（以第一个插补数据集为例）
        if imputed_datasets:
            plt.subplot(2, 2, 3)
            # 选择第一个变量进行对比（如果没有缺失值则跳过）
            comparison_var_index = None
            for i, info in enumerate(missing_info):
                if info["missing_count"] > 0:
                    comparison_var_index = i
                    break
            
            if comparison_var_index is not None:
                original_var_data = df[var_names[comparison_var_index]].dropna()
                imputed_var_data = imputed_datasets[0][:, comparison_var_index]
                
                plt.hist(original_var_data, bins=30, alpha=0.7, label='原始数据（非缺失值）', 
                         color='skyblue', density=True)
                plt.hist(imputed_var_data, bins=30, alpha=0.7, label='插补后数据', 
                         color='lightcoral', density=True)
                plt.xlabel(var_names[comparison_var_index])
                plt.ylabel('密度')
                plt.title(f'插补前后数据分布对比\n（{var_names[comparison_var_index]}）')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, '所有变量均无缺失值', ha='center', va='center')
                plt.title('插补前后数据分布对比')
        
        # 插补方法说明
        plt.subplot(2, 2, 4)
        plt.axis('off')
        method_explanation = {
            "mice": "MICE (多重插补):\n使用链式方程进行多重插补\n通过回归模型迭代插补缺失值\n考虑插补的不确定性",
            "mean": "均值插补:\n用各变量的均值填充缺失值\n简单易用但会降低数据变异性",
            "median": "中位数插补:\n用各变量的中位数填充缺失值\n对异常值更鲁棒",
            "mode": "众数插补:\n用各变量的众数填充缺失值\n适用于分类变量",
            "knn": "K近邻插补:\n基于K个最近邻样本的均值插补\n考虑变量间的关系"
        }
        explanation = method_explanation.get(method, "未知方法")
        plt.text(0.1, 0.5, explanation, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title('插补方法说明')
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"multiple_imputation_analysis_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成结果字典
        result = {
            "success": True,
            "original_data": processed_data,
            "imputed_data": imputed_data_lists[0] if imputed_data_lists else processed_data,
            "all_imputed_datasets": imputed_data_lists if len(imputed_data_lists) > 1 else None,
            "variables": var_names,
            "samples_count": n_samples,
            "missing_info": missing_info,
            "total_missing_values": int(total_missing),
            "method": method,
            "n_imputations": n_imputations if method == "mice" else 1,
            "pooled_results": pooled_results,
            "chart_url": f"{PUBLIC_FILE_BASE_URL}/{chart_filename}",
            "interpretation": [
                f"总共检测到 {total_missing} 个缺失值，占数据的 {(total_missing/(n_samples*n_vars)*100):.2f}%。",
                f"使用 {method.upper()} 方法进行插补。",
                "建议在后续分析中考虑数据的不确定性，特别是在使用MICE方法时。"
            ]
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }