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


def perform_multicollinearity_diagnosis(
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按自变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="自变量名称列表"),
    vif_threshold: float = Field(5.0, description="VIF阈值，用于判断是否存在多重共线性"),
) -> dict:
    """
    执行多重共线性诊断（方差膨胀因子VIF分析）
    
    参数:
    - independent_vars: 自变量数据，所有自变量的值按自变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 自变量名称列表
    - vif_threshold: VIF阈值，用于判断是否存在多重共线性，默认为5.0
    
    返回:
    - 包含多重共线性诊断结果的字典
    """
    try:
        # 输入验证
        if not independent_vars:
            raise ValueError("自变量数据不能为空")
        
        n_vars = len(var_names)
        if n_vars == 0:
            raise ValueError("自变量数量不能为0")
            
        # 根据var_names中元素的个数计算数据长度
        if len(independent_vars) % n_vars != 0:
            raise ValueError("自变量数据长度与变量数量不匹配")
            
        n_obs = len(independent_vars) // n_vars
        if n_obs == 0:
            raise ValueError("样本数量不能为0")
        
        # 将一维数组还原为嵌套列表结构
        reshaped_independent_vars = []
        for i in range(n_vars):
            var_data = independent_vars[i * n_obs:(i + 1) * n_obs]
            reshaped_independent_vars.append(var_data)
        
        if n_vars >= n_obs:
            raise ValueError("自变量数量不能大于等于样本数量")
        
        if vif_threshold <= 1:
            raise ValueError("VIF阈值必须大于1")
        
        # 创建 DataFrame
        df = pd.DataFrame({var_names[i]: reshaped_independent_vars[i] for i in range(n_vars)})
        
        # 计算方差膨胀因子(VIF)
        vif_data = []
        problematic_vars = []
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # 为VIF计算添加常数项
            X_with_const = np.column_stack([np.ones(n_obs), df.values])
            feature_names = ['const'] + var_names
            
            for i in range(1, len(feature_names)):  # 跳过常数项
                vif = variance_inflation_factor(X_with_const, i)
                is_problematic = vif > vif_threshold
                
                vif_entry = {
                    "variable": feature_names[i],
                    "vif": vif,
                    "is_problematic": is_problematic
                }
                
                vif_data.append(vif_entry)
                
                if is_problematic:
                    problematic_vars.append(vif_entry)
                    
        except ImportError:
            return {
                "success": False,
                "error": "未安装statsmodels库，无法计算方差膨胀因子(VIF)"
            }
        
        # 计算相关系数矩阵
        correlation_matrix = df.corr().values.tolist()
        
        # 生成相关系数热力图
        plt.figure(figsize=(10, 8))
        im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, shrink=0.8)
        
        # 设置坐标轴标签
        plt.xticks(range(n_vars), var_names, rotation=45, ha='right')
        plt.yticks(range(n_vars), var_names)
        
        # 在每个格子中添加相关系数值
        for i in range(n_vars):
            for j in range(n_vars):
                plt.text(j, i, f'{correlation_matrix[i][j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if abs(correlation_matrix[i][j]) > 0.5 else 'black')
        
        plt.title('变量相关系数矩阵热力图')
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"multicollinearity_correlation_matrix_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成结果摘要
        max_vif = max([entry["vif"] for entry in vif_data])
        min_vif = min([entry["vif"] for entry in vif_data])
        mean_vif = np.mean([entry["vif"] for entry in vif_data])
        
        # 判断整体是否存在多重共线性问题
        overall_multicollinearity = len(problematic_vars) > 0
        
        # 生成解释性文本
        interpretation = []
        if overall_multicollinearity:
            interpretation.append(f"检测到{len(problematic_vars)}个变量存在多重共线性问题(VIF > {vif_threshold})。")
            interpretation.append("建议考虑以下处理方法：")
            interpretation.append("1. 移除VIF值最高的变量")
            interpretation.append("2. 使用主成分分析(PCA)进行降维")
            interpretation.append("3. 使用岭回归等正则化方法")
            interpretation.append("4. 合并相关变量")
        else:
            interpretation.append("未检测到明显的多重共线性问题。")
        
        interpretation.append("\nVIF解释：")
        interpretation.append("- VIF = 1: 无多重共线性")
        interpretation.append("- 1 < VIF < 5: 存在轻微多重共线性")
        interpretation.append("- VIF ≥ 5: 存在严重多重共线性")
        
        # 生成结果字典
        result = {
            "success": True,
            "number_of_variables": n_vars,
            "number_of_observations": n_obs,
            "vif_threshold": vif_threshold,
            "vif_results": vif_data,
            "problematic_variables": problematic_vars,
            "correlation_matrix": correlation_matrix,
            "correlation_matrix_variables": var_names,
            "correlation_heatmap_url": f"{PUBLIC_FILE_BASE_URL}/{chart_filename}",
            "statistics": {
                "max_vif": float(max_vif),
                "min_vif": float(min_vif),
                "mean_vif": float(mean_vif)
            },
            "overall_multicollinearity_detected": overall_multicollinearity,
            "interpretation": interpretation
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }