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

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


def perform_mixed_effects_model(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="固定效应自变量数据，所有固定效应自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    random_effects_vars: List[float] = Field(..., description="随机效应变量数据，所有随机效应变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    grouping_vars: List[float] = Field(..., description="分组变量数据，所有分组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为固定效应自变量名"),
    random_effects_names: List[str] = Field(..., description="随机效应变量名称列表"),
    grouping_names: List[str] = Field(..., description="分组变量名称列表"),
    fit_method: str = Field("ml", description="拟合方法: 'ml'(最大似然), 'reml'(受限最大似然)"),
) -> dict:
    """
    执行混合效应模型分析（Mixed Effects Models / Hierarchical Models）
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 固定效应自变量数据，所有固定效应自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - random_effects_vars: 随机效应变量数据，所有随机效应变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - grouping_vars: 分组变量数据，所有分组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表，第一个为因变量名，后续为固定效应自变量名
    - random_effects_names: 随机效应变量名称列表
    - grouping_names: 分组变量名称列表
    - fit_method: 拟合方法 ('ml', 'reml')
    
    返回:
    - 包含混合效应模型分析结果的字典
    """    
    try:
        # 输入验证
        n_obs = len(dependent_var)
        
        # 根据var_names中元素的个数-1计算固定效应变量数量
        n_fixed_vars = len(var_names) - 1
        
        # 根据random_effects_names中元素的个数计算随机效应变量数量
        n_random_vars = len(random_effects_names)
        
        # 根据grouping_names中元素的个数计算分组变量数量
        n_grouping_vars = len(grouping_names)
        
        if n_obs == 0:
            raise ValueError("因变量数据不能为空")
        
        # 检查固定效应变量数据长度是否匹配
        if len(independent_vars) != n_obs * n_fixed_vars:
            raise ValueError("固定效应自变量数据长度与变量数量和观测数量不匹配")
        
        # 检查随机效应变量数据长度是否匹配
        if len(random_effects_vars) != n_obs * n_random_vars:
            raise ValueError("随机效应变量数据长度与变量数量和观测数量不匹配")
        
        # 检查分组变量数据长度是否匹配
        if len(grouping_vars) != n_obs * n_grouping_vars:
            raise ValueError("分组变量数据长度与变量数量和观测数量不匹配")
        
        # 将一维数组还原为嵌套列表结构
        # 重构固定效应变量数据
        reshaped_independent_vars = []
        for i in range(n_fixed_vars):
            var_data = independent_vars[i * n_obs:(i + 1) * n_obs]
            reshaped_independent_vars.append(var_data)
        
        # 重构随机效应变量数据
        reshaped_random_effects_vars = []
        for i in range(n_random_vars):
            var_data = random_effects_vars[i * n_obs:(i + 1) * n_obs]
            reshaped_random_effects_vars.append(var_data)
        
        # 重构分组变量数据
        reshaped_grouping_vars = []
        for i in range(n_grouping_vars):
            var_data = grouping_vars[i * n_obs:(i + 1) * n_obs]
            reshaped_grouping_vars.append(var_data)
        
        # 检查所有变量的数据长度是否一致
        for i, var_data in enumerate(reshaped_independent_vars):
            if len(var_data) != n_obs:
                raise ValueError(f"固定效应自变量 {var_names[i+1]} 的数据长度与因变量不匹配")
        
        for i, var_data in enumerate(reshaped_random_effects_vars):
            if len(var_data) != n_obs:
                raise ValueError(f"随机效应变量 {random_effects_names[i]} 的数据长度与因变量不匹配")
        
        for i, var_data in enumerate(reshaped_grouping_vars):
            if len(var_data) != n_obs:
                raise ValueError(f"分组变量 {grouping_names[i]} 的数据长度与因变量不匹配")
        
        if fit_method not in ["ml", "reml"]:
            raise ValueError("fit_method 参数必须是 'ml' 或 'reml'")
        
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        
        # 添加固定效应变量
        for i, var_data in enumerate(reshaped_independent_vars):
            data[var_names[i+1]] = var_data
        
        # 添加随机效应变量
        for i, var_data in enumerate(reshaped_random_effects_vars):
            data[random_effects_names[i]] = var_data
        
        # 添加分组变量（转换为类别型）
        for i, var_data in enumerate(reshaped_grouping_vars):
            data[grouping_names[i]] = pd.Categorical(var_data)
        
        # 准备模型公式
        # 固定效应部分
        fixed_effects_formula = " + ".join(var_names[1:]) if n_fixed_vars > 0 else "1"
        
        # 随机效应部分
        random_effects_formula = " + ".join(random_effects_names) if n_random_vars > 0 else "1"
        
        # 构建完整公式
        formula = f"{var_names[0]} ~ {fixed_effects_formula}"
        
        # 构建随机效应公式
        if n_grouping_vars > 0:
            # 如果有多个分组变量，我们使用第一个作为主要分组变量
            grouping_var = grouping_names[0]
            re_formula = f"{random_effects_formula}" if n_random_vars > 0 else "1"
        else:
            raise ValueError("至少需要一个分组变量")
        
        # 拟合混合效应模型
        try:
            model = MixedLM.from_formula(
                formula=formula,
                data=data,
                re_formula=re_formula,
                groups=data[grouping_var]
            )
            
            # 拟合模型
            if fit_method == "reml":
                fitted_model = model.fit(reml=True)
            else:
                fitted_model = model.fit(reml=False)
        except Exception as e:
            raise RuntimeError(f"模型拟合失败: {str(e)}")
        
        # 提取模型结果
        # 固定效应结果
        fixed_effects_summary = []
        fe_params = fitted_model.fe_params
        fe_stderr = fitted_model.bse_fe
        fe_tvalues = fitted_model.tvalues
        fe_pvalues = fitted_model.pvalues
        
        for i, param_name in enumerate(fe_params.index):
            fixed_effects_summary.append({
                "variable": param_name,
                "coefficient": float(fe_params.iloc[i]),
                "std_error": float(fe_stderr.iloc[i]),
                "t_value": float(fe_tvalues.iloc[i]),
                "p_value": float(fe_pvalues.iloc[i]),
                "significant": float(fe_pvalues.iloc[i]) < 0.05
            })
        
        # 随机效应结果
        random_effects_summary = []
        try:
            # 获取随机效应方差分量
            if hasattr(fitted_model, 'cov_re') and fitted_model.cov_re is not None:
                # 随机效应方差
                re_var = fitted_model.cov_re
                if re_var.ndim == 0:  # 标量情况
                    random_effects_summary.append({
                        "component": "随机效应方差",
                        "variance": float(re_var),
                        "std_deviation": float(np.sqrt(re_var))
                    })
                else:  # 矩阵情况
                    for i in range(re_var.shape[0]):
                        random_effects_summary.append({
                            "component": f"随机效应方差 (组件 {i+1})",
                            "variance": float(re_var[i, i]),
                            "std_deviation": float(np.sqrt(re_var[i, i]))
                        })
            
            # 残差方差
            random_effects_summary.append({
                "component": "残差方差",
                "variance": float(fitted_model.scale),
                "std_deviation": float(np.sqrt(fitted_model.scale))
            })
        except Exception:
            # 如果无法提取随机效应信息，则留空
            pass
        
        # 模型拟合统计量
        try:
            llf = fitted_model.llf  # 对数似然值
            aic = fitted_model.aic   # AIC
            bic = fitted_model.bic   # BIC
        except Exception as e:
            llf = None
            aic = None
            bic = None
        
        # 计算边际R²和条件R²（如果可能）
        r_squared_marginal = None
        r_squared_conditional = None
        try:
            # 边际R²：仅固定效应解释的方差比例
            # 条件R²：固定效应和随机效应共同解释的方差比例
            if hasattr(fitted_model, 'predict') and len(fixed_effects_summary) > 0:
                # 获取固定效应预测值
                fixed_effects_pred = fitted_model.predict()
                
                # 计算总方差
                y = data[var_names[0]]
                total_var = np.var(y)
                
                if total_var > 0:
                    # 边际R²
                    fixed_var = np.var(fixed_effects_pred)
                    r_squared_marginal = fixed_var / total_var
                    
                    # 条件R²（简化估算）
                    # 这里我们使用一个简化的近似方法
                    if len(random_effects_summary) > 0:
                        random_var = sum(item['variance'] for item in random_effects_summary if 'variance' in item)
                        r_squared_conditional = (fixed_var + random_var) / (total_var + random_var)
        except Exception:
            pass
        
        # 生成残差图
        try:
            # 获取残差
            residuals = fitted_model.resid
            
            # 绘制残差图
            plt.figure(figsize=(15, 10))
            
            # 残差 vs 拟合值图
            plt.subplot(2, 3, 1)
            fitted_values = fitted_model.fittedvalues
            plt.scatter(fitted_values, residuals, alpha=0.7)
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
            try:
                standardized_residuals = residuals / np.sqrt(fitted_model.scale)
                plt.subplot(2, 3, 4)
                plt.scatter(fitted_values, standardized_residuals, alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.axhline(y=2, color='r', linestyle=':', alpha=0.7)
                plt.axhline(y=-2, color='r', linestyle=':', alpha=0.7)
                plt.xlabel('拟合值')
                plt.ylabel('标准化残差')
                plt.title('标准化残差图')
                plt.grid(True, alpha=0.3)
            except Exception:
                plt.subplot(2, 3, 4)
                plt.text(0.5, 0.5, '无法生成标准化残差图', ha='center', va='center')
                plt.title('标准化残差图')
            
            # 随机效应图（如果有）
            try:
                if hasattr(fitted_model, 'random_effects') and fitted_model.random_effects:
                    re_values = []
                    group_labels = []
                    for group, re_dict in fitted_model.random_effects.items():
                        # 提取随机效应值
                        if isinstance(re_dict, dict):
                            re_values.extend(list(re_dict.values()))
                            group_labels.extend([group] * len(re_dict))
                        else:
                            re_values.append(re_dict)
                            group_labels.append(group)
                    
                    if re_values:
                        plt.subplot(2, 3, 5)
                        plt.hist(re_values, bins=20, edgecolor='black', alpha=0.7)
                        plt.xlabel('随机效应值')
                        plt.ylabel('频率')
                        plt.title('随机效应分布')
                        plt.grid(True, alpha=0.3)
                    else:
                        plt.subplot(2, 3, 5)
                        plt.text(0.5, 0.5, '无随机效应数据', ha='center', va='center')
                        plt.title('随机效应分布')
                else:
                    plt.subplot(2, 3, 5)
                    plt.text(0.5, 0.5, '无随机效应数据', ha='center', va='center')
                    plt.title('随机效应分布')
            except Exception:
                plt.subplot(2, 3, 5)
                plt.text(0.5, 0.5, '无法生成随机效应图', ha='center', va='center')
                plt.title('随机效应分布')
            
            # 拟合值 vs 实际值
            plt.subplot(2, 3, 6)
            plt.scatter(y, fitted_values, alpha=0.7)
            min_val = min(min(y), min(fitted_values))
            max_val = max(max(y), max(fitted_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('拟合值')
            plt.title('实际值 vs 拟合值')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存残差图
            plot_filename = f"mixed_effects_model_residuals_{uuid.uuid4().hex}.png"
            plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_url = f"{PUBLIC_FILE_BASE_URL}/{plot_filename}"
        except Exception as e:
            plot_url = None
        
        # 生成结果字典
        # 直接创建完整的结果字典
        result = {
            "success": True,
            "number_of_observations": n_obs,
            "number_of_fixed_effects": n_fixed_vars,
            "number_of_random_effects": n_random_vars,
            "number_of_groups": n_grouping_vars,
            "fit_method": fit_method.upper(),
            "fixed_effects": fixed_effects_summary,
            "random_effects": random_effects_summary,
            "model_fit_statistics": {
                "log_likelihood": float(llf) if llf is not None else None,
                "aic": float(aic) if aic is not None else None,
                "bic": float(bic) if bic is not None else None,
                "r_squared_marginal": float(r_squared_marginal) if r_squared_marginal is not None else None,
                "r_squared_conditional": float(r_squared_conditional) if r_squared_conditional is not None else None
            },
            "residual_plot_url": plot_url,
            "converged": fitted_model.converged,
            "warnings": fitted_model.warnings if hasattr(fitted_model, 'warnings') else None,
            "model_summary": str(fitted_model.summary()) if hasattr(fitted_model, 'summary') else None
        }
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"混合效应模型分析失败: {str(e)}") from e
