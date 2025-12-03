import os
import uuid
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
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

from statsmodels.discrete.discrete_model import Logit, MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.count_model import Poisson
import statsmodels.api as sm


def perform_generalized_linear_model(
    dependent_var: List = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    model_type: str = Field("logistic", description="模型类型: 'logistic'(二分类逻辑回归), 'multinomial'(多项逻辑回归), 'ordinal'(有序逻辑回归), 'poisson'(泊松回归), 'negative_binomial'(负二项回归)"),
) -> dict:
    """
    执行广义线性模型分析
    
    参数:
    - dependent_var: 因变量数据
    - independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
    - var_names: 变量名称列表，第一个为因变量名，后续为自变量名
    - model_type: 模型类型
    """    
    # 输入验证
    n_obs = len(dependent_var)
    
    # 根据var_names中元素的个数-1计算自变量数量
    n_vars = len(var_names) - 1
    
    if n_obs < n_vars + 1:
        raise ValueError("观测数量不足，无法进行回归分析")
    
    if len(independent_vars) != n_obs * n_vars:
        raise ValueError("自变量数据长度与变量数量和观测数量不匹配")
    
    if n_vars <= 0:
        raise ValueError("自变量数量必须大于0")
    
    # 将一维数组还原为嵌套列表结构
    reshaped_independent_vars = []
    for i in range(n_vars):
        var_data = independent_vars[i * n_obs:(i + 1) * n_obs]
        reshaped_independent_vars.append(var_data)
    
    # 检查所有自变量的数据长度是否一致
    for i, var_data in enumerate(reshaped_independent_vars):
        if len(var_data) != n_obs:
            raise ValueError(f"自变量 {var_names[i+1]} 的数据长度与因变量不匹配")
    
    if model_type not in ["logistic", "multinomial", "ordinal", "poisson", "negative_binomial"]:
        raise ValueError("model_type 参数必须是 'logistic', 'multinomial', 'ordinal', 'poisson', 'negative_binomial' 之一")
    
    try:
        # 创建 DataFrame
        data = pd.DataFrame({var_names[0]: dependent_var})
        for i, var_data in enumerate(reshaped_independent_vars):
            data[var_names[i+1]] = var_data
        
        # 根据模型类型处理数据和执行分析
        if model_type == "logistic":
            result = _perform_logistic_regression(data, var_names)
        elif model_type == "multinomial":
            result = _perform_multinomial_logistic_regression(data, var_names)
        elif model_type == "ordinal":
            result = _perform_ordinal_logistic_regression(data, var_names)
        elif model_type == "poisson":
            result = _perform_poisson_regression(data, var_names)
        elif model_type == "negative_binomial":
            result = _perform_negative_binomial_regression(data, var_names)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return result
        
    except Exception as e:
        # 让异常向上抛，FastMCP 会处理
        raise RuntimeError(f"广义线性模型分析失败: {str(e)}") from e


def _perform_logistic_regression(data: pd.DataFrame, var_names: List[str]) -> dict:
    """
    执行二分类逻辑回归
    """
    try:
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 添加常数项
        X_with_const = pd.concat([pd.Series(np.ones(len(X)), name='const'), X], axis=1)
        
        # 检查因变量是否为二分类
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            raise ValueError("逻辑回归要求因变量为二分类数据")
        
        # 编码因变量为0和1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 拟合逻辑回归模型
        model = Logit(y_encoded, X_with_const)
        result = model.fit(disp=0)
        
        # 获取预测概率和分类结果
        y_pred_prob = result.predict()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算准确率
        accuracy = accuracy_score(y_encoded, y_pred)
        
        # 生成系数图
        coef_names = ['常数项'] + var_names[1:]
        coef_values = result.params.tolist()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(coef_values)), coef_values, color='skyblue')
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('逻辑回归系数')
        plt.xticks(range(len(coef_values)), coef_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存系数图
        coef_plot_filename = f"logistic_regression_coefficients_{uuid.uuid4().hex}.png"
        coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
        plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        
        # 组织结果
        result_dict = {
            "model_type": "二分类逻辑回归",
            "number_of_observations": len(y),
            "number_of_variables": len(var_names) - 1,
            "variable_names": var_names[1:],
            "coefficients": dict(zip(coef_names, coef_values)),
            "p_values": dict(zip(coef_names, result.pvalues.tolist())),
            "odds_ratios": dict(zip(coef_names, np.exp(coef_values).tolist())),
            "accuracy": float(accuracy),
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_logistic_interpretation(result, accuracy)
        }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"逻辑回归分析失败: {str(e)}") from e


def _perform_multinomial_logistic_regression(data: pd.DataFrame, var_names: List[str]) -> dict:
    """
    执行多项逻辑回归
    """
    try:
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 添加常数项
        X_with_const = pd.concat([pd.Series(np.ones(len(X)), name='const'), X], axis=1)
        
        # 检查因变量类别数
        unique_vals = y.unique()
        if len(unique_vals) < 3:
            raise ValueError("多项逻辑回归要求因变量至少有3个类别")
        
        # 编码因变量
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 拟合多项逻辑回归模型
        model = MNLogit(y_encoded, X_with_const)
        result = model.fit(disp=0)
        
        # 获取系数名称
        coef_names = ['常数项'] + var_names[1:]
        category_names = [str(cat) for cat in le.classes_]
        
        # 组织系数结果
        coefficients = {}
        p_values = {}
        for i, category in enumerate(category_names[1:], 1):  # 跳过基准类别
            coefficients[category] = dict(zip(coef_names, result.params.iloc[:, i-1].tolist()))
            p_values[category] = dict(zip(coef_names, result.pvalues.iloc[:, i-1].tolist()))
        
        # 生成系数图（以第二个类别为例）
        if len(category_names) > 1:
            coef_values = result.params.iloc[:, 0].tolist()  # 第二个类别的系数
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(coef_values)), coef_values, color='lightcoral')
            plt.xlabel('变量')
            plt.ylabel('系数值')
            plt.title(f'多项逻辑回归系数 (类别: {category_names[1]} vs {category_names[0]})')
            plt.xticks(range(len(coef_values)), coef_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存系数图
            coef_plot_filename = f"multinomial_regression_coefficients_{uuid.uuid4().hex}.png"
            coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
            plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        else:
            coef_plot_url = None
        
        # 组织结果
        result_dict = {
            "model_type": "多项逻辑回归",
            "number_of_observations": len(y),
            "number_of_variables": len(var_names) - 1,
            "number_of_categories": len(unique_vals),
            "category_names": category_names,
            "variable_names": var_names[1:],
            "coefficients": coefficients,
            "p_values": p_values,
            "odds_ratios": {cat: {var: np.exp(coef) for var, coef in coefs.items()} 
                           for cat, coefs in coefficients.items()},
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_multinomial_interpretation(result, category_names)
        }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"多项逻辑回归分析失败: {str(e)}") from e


def _perform_ordinal_logistic_regression(data: pd.DataFrame, var_names: List[str]) -> dict:
    """
    执行有序逻辑回归
    """
    try:
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 检查因变量是否为有序分类
        unique_vals = sorted(y.unique())
        if len(unique_vals) < 3:
            raise ValueError("有序逻辑回归要求因变量至少有3个有序类别")
        
        # 拟合有序逻辑回归模型
        model = OrderedModel(y, X, distr='logit')
        result = model.fit(disp=0)
        
        # 获取系数名称
        coef_names = var_names[1:]
        coef_values = result.params[:len(coef_names)].tolist()
        
        # 获取阈值
        threshold_names = [f"阈值_{i}_{i+1}" for i in range(len(unique_vals)-1)]
        threshold_values = result.params[len(coef_names):].tolist()
        
        # 生成系数图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(coef_values)), coef_values, color='gold')
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('有序逻辑回归系数')
        plt.xticks(range(len(coef_values)), coef_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存系数图
        coef_plot_filename = f"ordinal_regression_coefficients_{uuid.uuid4().hex}.png"
        coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
        plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        
        # 组织结果
        result_dict = {
            "model_type": "有序逻辑回归",
            "number_of_observations": len(y),
            "number_of_variables": len(var_names) - 1,
            "number_of_categories": len(unique_vals),
            "category_names": unique_vals,
            "variable_names": var_names[1:],
            "coefficients": dict(zip(coef_names, coef_values)),
            "p_values": dict(zip(coef_names, result.pvalues[:len(coef_names)].tolist())),
            "thresholds": dict(zip(threshold_names, threshold_values)),
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_ordinal_interpretation(result, coef_names)
        }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"有序逻辑回归分析失败: {str(e)}") from e


def _perform_poisson_regression(data: pd.DataFrame, var_names: List[str]) -> dict:
    """
    执行泊松回归
    """
    try:
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 添加常数项
        X_with_const = pd.concat([pd.Series(np.ones(len(X)), name='const'), X], axis=1)
        
        # 检查因变量是否为非负整数
        if not all(isinstance(val, (int, np.integer)) and val >= 0 for val in y):
            raise ValueError("泊松回归要求因变量为非负整数")
        
        # 拟合泊松回归模型
        model = Poisson(y, X_with_const)
        result = model.fit(disp=0)
        
        # 获取系数名称
        coef_names = ['常数项'] + var_names[1:]
        coef_values = result.params.tolist()
        
        # 生成系数图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(coef_values)), coef_values, color='lightgreen')
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('泊松回归系数')
        plt.xticks(range(len(coef_values)), coef_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存系数图
        coef_plot_filename = f"poisson_regression_coefficients_{uuid.uuid4().hex}.png"
        coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
        plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        
        # 计算拟合优度指标
        y_pred = result.predict()
        deviance = np.sum(2 * (y * np.log(y / (y_pred + 1e-10)) - (y - y_pred)))
        
        # 组织结果
        result_dict = {
            "model_type": "泊松回归",
            "number_of_observations": len(y),
            "number_of_variables": len(var_names) - 1,
            "variable_names": var_names[1:],
            "coefficients": dict(zip(coef_names, coef_values)),
            "p_values": dict(zip(coef_names, result.pvalues.tolist())),
            "incidence_rate_ratios": dict(zip(coef_names, np.exp(coef_values).tolist())),
            "deviance": float(deviance),
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_poisson_interpretation(result, deviance)
        }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"泊松回归分析失败: {str(e)}") from e


def _perform_negative_binomial_regression(data: pd.DataFrame, var_names: List[str]) -> dict:
    """
    执行负二项回归
    """
    try:
        y = data[var_names[0]]
        X = data[var_names[1:]]
        
        # 添加常数项
        # X_with_const = pd.concat([pd.Series(np.ones(len(X)), name='const'), X], axis=1)
        
        # 检查因变量是否为非负整数
        if not all(isinstance(val, (int, np.integer)) and val >= 0 for val in y):
            raise ValueError("负二项回归要求因变量为非负整数")
        
        # 拟合负二项回归模型
        # model = NegativeBinomial(y, X_with_const)
        model = sm.GLM(y, sm.add_constant(X), family=sm.families.NegativeBinomial())
        result = model.fit(disp=0)
        
        # 获取系数名称
        coef_names = ['常数项'] + var_names[1:]
        coef_values = result.params.tolist()
        
        # 生成系数图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(coef_values)), coef_values, color='orchid')
        plt.xlabel('变量')
        plt.ylabel('系数值')
        plt.title('负二项回归系数')
        plt.xticks(range(len(coef_values)), coef_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存系数图
        coef_plot_filename = f"negative_binomial_coefficients_{uuid.uuid4().hex}.png"
        coef_plot_filepath = os.path.join(OUTPUT_DIR, coef_plot_filename)
        plt.savefig(coef_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        coef_plot_url = f"{PUBLIC_FILE_BASE_URL}/{coef_plot_filename}"
        
        # 计算拟合优度指标
        y_pred = result.predict()
        deviance = np.sum(2 * (y * np.log(y / (y_pred + 1e-10)) - (y - y_pred)))
        
        # 组织结果
        result_dict = {
            "model_type": "负二项回归",
            "number_of_observations": len(y),
            "number_of_variables": len(var_names) - 1,
            "variable_names": var_names[1:],
            "coefficients": dict(zip(coef_names, coef_values)),
            "p_values": dict(zip(coef_names, result.pvalues.tolist())),
            "incidence_rate_ratios": dict(zip(coef_names, np.exp(coef_values).tolist())),
            "deviance": float(deviance),
            "coefficient_plot_url": coef_plot_url,
            "interpretation": _get_negative_binomial_interpretation(result, deviance)
        }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"负二项回归分析失败: {str(e)}") from e


def _get_logistic_interpretation(result, accuracy: float) -> str:
    """
    生成逻辑回归结果解释
    """
    interpretation = f"二分类逻辑回归分析完成。模型准确率为{accuracy:.2%}。"
    
    significant_vars = []
    for i, p_value in enumerate(result.pvalues):
        if p_value < 0.05:
            significant_vars.append(i)
    
    if len(significant_vars) > 0:
        interpretation += f"模型中存在{len(significant_vars)}个显著变量。"
    else:
        interpretation += "模型中没有显著变量。"
    
    return interpretation


def _get_multinomial_interpretation(result, category_names: List[str]) -> str:
    """
    生成多项逻辑回归结果解释
    """
    interpretation = f"多项逻辑回归分析完成，因变量包含{len(category_names)}个类别。"
    
    return interpretation


def _get_ordinal_interpretation(result, var_names: List[str]) -> str:
    """
    生成有序逻辑回归结果解释
    """
    interpretation = f"有序逻辑回归分析完成。"
    
    return interpretation


def _get_poisson_interpretation(result, deviance: float) -> str:
    """
    生成泊松回归结果解释
    """
    interpretation = f"泊松回归分析完成，模型偏差为{deviance:.4f}。"
    
    return interpretation


def _get_negative_binomial_interpretation(result, deviance: float) -> str:
    """
    生成负二项回归结果解释
    """
    interpretation = f"负二项回归分析完成，模型偏差为{deviance:.4f}。"
    
    return interpretation