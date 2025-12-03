import os
from typing import List, Union, Optional
import fastmcp
# 打印fastmcp的版本
print(f"FastMCP Version: {fastmcp.__version__}")
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

# 加载配置
from config import (
    OUTPUT_DIR,
    PUBLIC_FILE_BASE_URL,
    BASE_URL,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_PATH,
    SERVER_TRANSPORT,
    SERVER_LOG_LEVEL
)
print(f"当前工作目录: {os.getcwd()}")
print(f"OUTPUT_DIR 绝对路径: {os.path.abspath(OUTPUT_DIR)}")

from cleanup import start_cleanup_scheduler

# 导入工具函数
from mcp_tools.multiple_regression_tool import perform_multiple_regression
from mcp_tools.stepwise_regression_tool import perform_stepwise_regression
from mcp_tools.hotelling_t2_tool import perform_hotelling_t2_test
from mcp_tools.multivariate_normality_tool import perform_multivariate_normality_test
from mcp_tools.covariance_homogeneity_tool import perform_covariance_homogeneity_test
from mcp_tools.manova_tool import perform_manova
from mcp_tools.box_m_test_tool import perform_box_m_test
from mcp_tools.categorical_independence_test_tool import perform_categorical_independence_test
from mcp_tools.continuous_independence_test_tool import perform_continuous_independence_test
from mcp_tools.fisher_discriminant_analysis_tool import perform_fisher_discriminant_analysis
from mcp_tools.distance_discriminant_analysis_tool import perform_distance_discriminant_analysis
from mcp_tools.bayes_discriminant_analysis_tool import perform_bayes_discriminant_analysis
from mcp_tools.generalized_square_distance_discriminant_analysis_tool import perform_generalized_square_distance_discriminant_analysis
from mcp_tools.stepwise_discriminant_analysis_tool import perform_stepwise_discriminant_analysis
# 添加聚类分析工具导入
from mcp_tools.kmeans_clustering_tool import perform_kmeans_clustering
from mcp_tools.kmedoids_clustering_tool import perform_kmedoids_clustering
from mcp_tools.dbscan_clustering_tool import perform_dbscan_clustering
from mcp_tools.hierarchical_clustering_tool import perform_hierarchical_clustering
from mcp_tools.gmm_clustering_tool import perform_gmm_clustering
# 添加主成分分析工具导入
from mcp_tools.pca_tool import perform_pca
# 添加因子分析工具导入
from mcp_tools.factor_analysis_tool import perform_factor_analysis
# 添加对应分析工具导入
from mcp_tools.correspondence_analysis_tool import perform_correspondence_analysis
# 添加典型相关分析工具导入
from mcp_tools.canonical_correlation_analysis_tool import perform_canonical_correlation_analysis
# 添加偏最小二乘回归分析工具导入
from mcp_tools.pls_regression_tool import perform_pls_regression
# 添加时间序列分析工具导入
from mcp_tools.time_series_analysis_tool import perform_time_series_analysis
# 添加时间序列预处理检验工具导入
from mcp_tools.time_series_preprocessing_test_tool import perform_time_series_preprocessing_tests
# 添加正则化回归工具导入
from mcp_tools.regularized_regression_tool import perform_regularized_regression
# 添加广义线性模型工具导入
from mcp_tools.generalized_linear_model_tool import perform_generalized_linear_model
# 添加结构方程模型工具导入
from mcp_tools.sem_tool import perform_sem
# 添加分位数回归工具导入
from mcp_tools.quantile_regression_tool import perform_quantile_regression
# 添加多维尺度分析工具导入
from mcp_tools.mds_tool import perform_mds
# 添加UMAP非线性降维工具导入
from mcp_tools.umap_tool import perform_umap
# 添加稳健回归工具导入
from mcp_tools.robust_regression_tool import perform_robust_regression
# 添加HDBSCAN聚类工具导入
from mcp_tools.hdbscan_clustering_tool import perform_hdbscan_clustering
# 添加时间序列协整和Granger因果检验工具导入
from mcp_tools.time_series_cointegration_granger_tool import perform_time_series_cointegration_granger_tests
# 添加数据标准化工具导入
from mcp_tools.data_standardization_tool import perform_data_standardization
# 添加多重共线性诊断工具导入
from mcp_tools.multicollinearity_diagnosis_tool import perform_multicollinearity_diagnosis
# 添加混合效应模型工具导入
from mcp_tools.mixed_effects_model_tool import perform_mixed_effects_model
# 添加多重插补工具导入
from mcp_tools.multiple_imputation_tool import perform_multiple_imputation
# 添加生存分析工具导入
from mcp_tools.survival_analysis_tool import perform_survival_analysis
# 添加因果推断工具导入
from mcp_tools.causal_inference_tool import perform_causal_inference
# 添加非参数回归工具导入
from mcp_tools.nonparametric_regression_tool import perform_nonparametric_regression

# 添加数据分析所需库
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import uuid
import json

# ========== 初始化 FastMCP ==========
mcp = FastMCP("multivariate-statistics-mcp-server 📊")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 尝试写一个测试文件，确保输出目录可写
test_file = os.path.join(OUTPUT_DIR, ".write_test")
try:
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
except Exception as e:
    print(f"FATAL: 无法写入 {OUTPUT_DIR}: {e}")
    exit(1)

# ===================== 工具函数定义（参数直接定义在函数签名中） =====================

@mcp.tool()
def perform_multiple_regression_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
):
    """
    执行多元线性回归分析，包含统计显著性检验和残差分析
    """
    return perform_multiple_regression(dependent_var, independent_vars, var_names)

@mcp.tool()
def perform_stepwise_regression_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("both", description="逐步回归方法(可选): 'forward'(向前), 'backward'(向后), 'both'(双向)"),
    significance_level_enter: float = Field(0.05, description="变量进入模型的显著性水平(可选)"),
    significance_level_remove: float = Field(0.10, description="变量移出模型的显著性水平(可选)"),
):
    """
    执行逐步回归分析，包含统计显著性检验和残差分析
    """
    return perform_stepwise_regression(
        dependent_var, 
        independent_vars, 
        var_names, 
        method, 
        significance_level_enter, 
        significance_level_remove
    )

@mcp.tool()
def perform_hotelling_t2_test_tool(
    group1_data: List[float] = Field(..., description="第一组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    group2_data: List[float] = Field(..., description="第二组样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行Hotelling's T²检验，用于检验两个多维正态分布总体的均值向量是否相等
    """
    return perform_hotelling_t2_test(group1_data, group2_data, var_names)

@mcp.tool()
def perform_multivariate_normality_test_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行多元正态性检验，检验多变量数据是否符合多元正态分布
    """
    return perform_multivariate_normality_test(data, var_names)

@mcp.tool()
def perform_covariance_homogeneity_test_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行协方差矩阵齐性检验(Bartlett检验)，用于检验多个多元正态分布总体的协方差矩阵是否相等
    """
    return perform_covariance_homogeneity_test(groups_data, group_names, var_names)

@mcp.tool()
def perform_manova_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行多元方差分析(MANOVA)，用于检验多个组在多个因变量上的均值向量是否存在显著差异
    """
    return perform_manova(groups_data, group_names, var_names)

@mcp.tool()
def perform_box_m_test_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行Box's M检验，用于检验多个多元正态分布总体的协方差矩阵是否相等
    """
    return perform_box_m_test(groups_data, group_names, var_names)

@mcp.tool()
def perform_categorical_independence_test_tool(
    observed_frequencies: List[int] = Field(..., description="观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)"),
    row_labels: List[str] = Field(..., description="行标签列表"),
    col_labels: List[str] = Field(..., description="列标签列表"),
):
    """
    执行卡方独立性检验，用于检验两个分类变量是否相互独立
    """
    return perform_categorical_independence_test(observed_frequencies, row_labels, col_labels)

@mcp.tool()
def perform_continuous_independence_test_tool(
    data: List[float] = Field(..., description="连续变量数据，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行连续变量相关性检验，用于检验多个连续变量之间的线性相关性
    """
    return perform_continuous_independence_test(data, var_names)

@mcp.tool()
def perform_fisher_discriminant_analysis_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
):
    """
    执行Fisher判别分析，用于寻找最能区分不同组的线性组合
    """
    return perform_fisher_discriminant_analysis(groups_data, group_names, var_names)

@mcp.tool()
def perform_distance_discriminant_analysis_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    test_data: List[float] = Field(None, description="待判别样本数据(可选)，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
):
    """
    执行距离判别分析，基于马氏距离进行分类判别
    """
    return perform_distance_discriminant_analysis(groups_data, group_names, var_names, test_data)

@mcp.tool()
def perform_bayes_discriminant_analysis_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    test_data: List[float] = Field(None, description="待判别样本数据(可选)，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
    prior_probabilities: List[float] = Field(None, description="先验概率列表(可选)，长度应与组数相同"),
):
    """
    执行贝叶斯判别分析，基于贝叶斯定理和多元正态分布假设进行分类判别
    """
    return perform_bayes_discriminant_analysis(groups_data, group_names, var_names, test_data, prior_probabilities)

@mcp.tool()
def perform_generalized_square_distance_discriminant_analysis_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    test_data: List[float] = Field(None, description="待判别样本数据(可选)，所有变量的值按变量拼接成一维数组(先放完 X1、再放 X2 ...)"),
):
    """
    执行广义平方距离判别分析，考虑各组协方差矩阵不等的情况进行分类判别
    """
    return perform_generalized_square_distance_discriminant_analysis(groups_data, group_names, var_names, test_data)

@mcp.tool()
def perform_stepwise_discriminant_analysis_tool(
    groups_data: List[float] = Field(..., description="多组样本数据，所有组和变量的值按组和变量拼接成一维数组(先放组1的 X1、X2...，再放组2的X1、X2...)"),
    group_names: List[str] = Field(..., description="组名称列表"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("wilks", description="逐步选择方法: 'wilks'(Wilks' Lambda), 'f'(F检验)"),
    significance_level_enter: float = Field(0.05, description="变量进入模型的显著性水平"),
    significance_level_remove: float = Field(0.10, description="变量移出模型的显著性水平"),
):
    """
    执行逐步判别分析，通过逐步选择变量构建最优判别函数
    """
    return perform_stepwise_discriminant_analysis(
        groups_data, 
        group_names, 
        var_names, 
        method, 
        significance_level_enter, 
        significance_level_remove
    )

@mcp.tool()
def perform_kmeans_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_clusters: int = Field(3, description="聚类数量"),
    init_method: str = Field("k-means++", description="初始化方法: 'k-means++' 或 'random'"),
    max_iter: int = Field(300, description="最大迭代次数"),
    n_init: int = Field(10, description="运行算法的次数，返回最好的结果"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
):
    """
    执行K-Means聚类分析
    """
    return perform_kmeans_clustering(data, var_names, n_clusters, init_method, max_iter, n_init, random_state)

@mcp.tool()
def perform_kmedoids_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_clusters: int = Field(3, description="聚类数量"),
    max_iter: int = Field(300, description="最大迭代次数"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
):
    """
    执行K-Medoids聚类分析
    """
    return perform_kmedoids_clustering(data, var_names, n_clusters, max_iter, random_state)

@mcp.tool()
def perform_dbscan_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    eps: float = Field(0.5, description="邻域半径"),
    min_samples: int = Field(5, description="核心点邻域中的最小样本数"),
):
    """
    执行DBSCAN聚类分析
    """
    return perform_dbscan_clustering(data, var_names, eps, min_samples)

@mcp.tool()
def perform_hierarchical_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_clusters: int = Field(3, description="聚类数量"),
    linkage_method: str = Field("ward", description="链接方法: 'ward', 'complete', 'average', 'single'"),
):
    """
    执行凝聚式层次聚类分析
    """
    return perform_hierarchical_clustering(data, var_names, n_clusters, linkage_method)

@mcp.tool()
def perform_gmm_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_components: int = Field(3, description="混合成分数量（相当于聚类数）"),
    covariance_type: str = Field("full", description="协方差类型: 'full', 'tied', 'diag', 'spherical'"),
    max_iter: int = Field(100, description="最大迭代次数"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
):
    """
    执行高斯混合模型(GMM)聚类分析
    """
    return perform_gmm_clustering(data, var_names, n_components, covariance_type, max_iter, random_state)

# 添加主成分分析工具函数
@mcp.tool()
def perform_pca_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_components: Optional[int] = Field(None, description="主成分数量，如果不指定则保留所有成分"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行主成分分析(PCA)
    """
    return perform_pca(data, var_names, n_components, standardize)

# 添加因子分析工具函数
@mcp.tool()
def perform_factor_analysis_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_factors: Optional[int] = Field(None, description="因子数量，如果不指定则使用默认方法确定"),
    rotation: str = Field("varimax", description="因子旋转方法: 'varimax', 'promax', None"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行因子分析
    """
    return perform_factor_analysis(data, var_names, n_factors, rotation, standardize)

@mcp.tool()
def perform_correspondence_analysis_tool(
    observed_frequencies: List[int] = Field(..., description="观测频数表，所有行的值按行拼接成一维数组(先放第1行的值，再放第2行的值，...)"),
    row_labels: List[str] = Field(..., description="行标签列表"),
    col_labels: List[str] = Field(..., description="列标签列表"),
):
    """
    执行对应分析
    """
    return perform_correspondence_analysis(observed_frequencies, row_labels, col_labels)

@mcp.tool()
def perform_canonical_correlation_analysis_tool(
    x_data: List[float] = Field(..., description="第一组变量数据，所有第一组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    y_data: List[float] = Field(..., description="第二组变量数据，所有第二组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    x_var_names: List[str] = Field(..., description="第一组变量名称列表"),
    y_var_names: List[str] = Field(..., description="第二组变量名称列表"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行典型相关分析
    """
    return perform_canonical_correlation_analysis(x_data, y_data, x_var_names, y_var_names, standardize)

@mcp.tool()
def perform_pls_regression_tool(
    x_data: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    y_data: List[float] = Field(..., description="因变量数据，所有因变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    x_var_names: List[str] = Field(..., description="自变量名称列表"),
    y_var_names: List[str] = Field(..., description="因变量名称列表"),
    n_components: Optional[int] = Field(None, description="成分数量，如果不指定则自动选择"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行偏最小二乘回归分析
    """
    return perform_pls_regression(x_data, y_data, x_var_names, y_var_names, n_components, standardize)

@mcp.tool()
def perform_time_series_analysis_tool(
    time_series: List[float] = Field(..., description="时间序列数据"),
    time_labels: List[str] = Field(..., description="时间标签列表"),
    model_type: str = Field("auto_arima", description="模型类型: 'auto_arima', 'sarima', 'exponential_smoothing', 'manual_arima'"),
    forecast_steps: int = Field(10, description="预测步数"),
    seasonal_period: Optional[int] = Field(None, description="季节性周期（如月度数据为12，季度数据为4）"),
    order: Optional[List[int]] = Field(None, description="ARIMA模型的(p,d,q)参数，格式为[p,d,q]"),
    seasonal_order: Optional[List[int]] = Field(None, description="季节性ARIMA模型的(P,D,Q,s)参数，格式为[P,D,Q,s]"),
):
    """
    执行时间序列分析
    """
    return perform_time_series_analysis(time_series, time_labels, model_type, forecast_steps, seasonal_period, order, seasonal_order)

@mcp.tool()
def perform_time_series_preprocessing_tests_tool(
    time_series: List[float] = Field(..., description="时间序列数据"),
    time_labels: List[str] = Field(..., description="时间标签列表"),
    seasonal_period: Optional[int] = Field(None, description="季节性周期（如月度数据为12，季度数据为4）"),
    test_types: List[str] = Field(["adf", "kpss", "normality", "autocorrelation"], description="要执行的检验类型列表: 'adf'(ADF平稳性检验), 'kpss'(KPSS平稳性检验), 'normality'(正态性检验), 'autocorrelation'(自相关检验)"),
):
    """
    执行时间序列预处理检验，用于判断时间序列是否适合建模
    """
    return perform_time_series_preprocessing_tests(time_series, time_labels, seasonal_period, test_types)

@mcp.tool()
def perform_regularized_regression_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("ridge", description="正则化方法: 'ridge'(岭回归), 'lasso'(套索回归), 'elastic_net'(弹性网络)"),
    alpha: float = Field(1.0, description="正则化强度参数"),
    l1_ratio: float = Field(0.5, description="Elastic Net中L1正则化的比例 (仅用于elastic_net方法)"),
    standardize: bool = Field(True, description="是否标准化数据"),
    fit_intercept: bool = Field(True, description="是否计算截距"),
):
    """
    执行正则化回归分析（岭回归、套索回归、Elastic Net回归）
    """
    return perform_regularized_regression(dependent_var, independent_vars, var_names, method, alpha, l1_ratio, standardize, fit_intercept)

@mcp.tool()
def perform_generalized_linear_model_tool(
    dependent_var: List = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    model_type: str = Field("logistic", description="模型类型: 'logistic'(二分类逻辑回归), 'multinomial'(多项逻辑回归), 'ordinal'(有序逻辑回归), 'poisson'(泊松回归), 'negative_binomial'(负二项回归)"),
):
    """
    执行广义线性模型分析
    """
    return perform_generalized_linear_model(dependent_var, independent_vars, var_names, model_type)

@mcp.tool()
def perform_quantile_regression_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    quantiles: List[float] = Field([0.25, 0.5, 0.75], description="分位数列表，每个值应在0到1之间"),
    alpha: float = Field(1.0, description="正则化强度参数"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行分位数回归分析
    """
    return perform_quantile_regression(dependent_var, independent_vars, var_names, quantiles, alpha, standardize)

@mcp.tool()
def perform_sem_tool(
    data: List[float] = Field(..., description="观测数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    model_description: str = Field(..., description="模型描述字符串，使用semopy语法定义测量模型和结构模型"),
    group_var: Optional[str] = Field(None, description="分组变量名称，用于多组分析"),
):
    """
    执行结构方程模型分析
    """
    return perform_sem(data, var_names, model_description, group_var)

@mcp.tool()
def perform_mds_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_components: int = Field(2, description="降维后的维度数"),
    metric: bool = Field(True, description="是否使用度量MDS，True表示经典MDS，False表示非度量MDS"),
    n_init: int = Field(4, description="初始化次数，用于寻找最佳解"),
    max_iter: int = Field(300, description="最大迭代次数"),
    dissimilarity: str = Field("euclidean", description="距离度量方法: 'euclidean'(欧氏距离), 'precomputed'(预先计算的距离矩阵)"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行多维尺度分析(MDS)
    """
    return perform_mds(data, var_names, n_components, metric, n_init, max_iter, dissimilarity, standardize)

@mcp.tool()
def perform_umap_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    n_neighbors: int = Field(15, description="邻居数量，控制局部与全局结构的平衡"),
    n_components: int = Field(2, description="降维后的维度数"),
    min_dist: float = Field(0.1, description="最小距离，控制簇的紧密程度"),
    metric: str = Field("euclidean", description="距离度量方法"),
    random_state: Optional[int] = Field(42, description="随机种子，用于结果可重现"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行UMAP非线性降维
    """
    return perform_umap(data, var_names, n_neighbors, n_components, min_dist, metric, random_state, standardize)

@mcp.tool()
def perform_robust_regression_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为自变量名"),
    method: str = Field("ransac", description="稳健回归方法: 'ransac'(RANSAC回归), 'huber'(Huber回归)"),
    standardize: bool = Field(True, description="是否标准化数据"),
    fit_intercept: bool = Field(True, description="是否计算截距"),
    min_samples: Optional[int] = Field(None, description="RANSAC算法中随机样本的最小数量"),
    residual_threshold: Optional[float] = Field(None, description="RANSAC算法中样本被视为内点的最大残差"),
    max_trials: int = Field(100, description="RANSAC算法的最大迭代次数"),
    epsilon: float = Field(1.35, description="Huber回归的参数，决定对异常值的敏感度"),
    alpha: float = Field(0.0001, description="Huber回归的正则化强度"),
):
    """
    执行稳健回归分析
    """
    return perform_robust_regression(
        dependent_var, independent_vars, var_names, method, standardize, fit_intercept,
        min_samples, residual_threshold, max_trials, epsilon, alpha
    )

@mcp.tool()
def perform_hdbscan_clustering_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    min_cluster_size: int = Field(5, description="形成簇所需的最小样本数"),
    min_samples: Optional[int] = Field(None, description="核心点邻域中的最小样本数，如果为None则默认等于min_cluster_size"),
    cluster_selection_method: str = Field("eom", description="簇选择方法: 'eom'(超额质量算法), 'leaf'(叶簇选择)"),
    allow_single_cluster: bool = Field(False, description="是否允许将所有点归为一个簇"),
    alpha: float = Field(1.0, description="用于计算不稳定性的参数"),
    metric: str = Field("euclidean", description="距离度量方法"),
    standardize: bool = Field(True, description="是否标准化数据"),
):
    """
    执行HDBSCAN聚类分析
    """
    return perform_hdbscan_clustering(
        data, var_names, min_cluster_size, min_samples, cluster_selection_method, 
        allow_single_cluster, alpha, metric, standardize
    )

@mcp.tool()
def perform_time_series_cointegration_granger_tests_tool(
    time_series_list: List[float] = Field(..., description="时间序列数据，所有时间序列的值按时间序列拼接成一维数组(列优先，先放完时间序列1、再放时间序列2 ...)"),
    series_names: List[str] = Field(..., description="时间序列名称列表"),
    max_lag: int = Field(5, description="Granger因果检验的最大滞后阶数"),
    significance_level: float = Field(0.05, description="显著性水平"),
):
    """
    执行时间序列协整检验和Granger因果检验
    """
    return perform_time_series_cointegration_granger_tests(
        time_series_list, series_names, max_lag, significance_level
    )

@mcp.tool()
def perform_data_standardization_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("zscore", description="标准化方法: 'zscore'(Z-score标准化), 'minmax'(Min-Max标准化)"),
    feature_range: List[float] = Field([0, 1], description="Min-Max标准化的目标范围，格式为[min, max]"),
):
    """
    执行数据标准化（Z-score标准化或Min-Max标准化）
    """
    return perform_data_standardization(data, var_names, method, feature_range)

@mcp.tool()
def perform_multicollinearity_diagnosis_tool(
    independent_vars: List[float] = Field(..., description="自变量数据，所有自变量的值按自变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="自变量名称列表"),
    vif_threshold: float = Field(5.0, description="VIF阈值，用于判断是否存在多重共线性"),
):
    """
    执行多重共线性诊断（方差膨胀因子VIF分析）
    """
    return perform_multicollinearity_diagnosis(independent_vars, var_names, vif_threshold)

@mcp.tool()
def perform_mixed_effects_model_tool(
    dependent_var: List[float] = Field(..., description="因变量数据"),
    independent_vars: List[float] = Field(..., description="固定效应自变量数据，所有固定效应自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    random_effects_vars: List[float] = Field(..., description="随机效应变量数据，所有随机效应变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    grouping_vars: List[float] = Field(..., description="分组变量数据，所有分组变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表，第一个为因变量名，后续为固定效应自变量名"),
    random_effects_names: List[str] = Field(..., description="随机效应变量名称列表"),
    grouping_names: List[str] = Field(..., description="分组变量名称列表"),
    fit_method: str = Field("ml", description="拟合方法: 'ml'(最大似然), 'reml'(受限最大似然)"),
):
    """
    执行混合效应模型分析（Mixed Effects Models / Hierarchical Models）
    """
    return perform_mixed_effects_model(
        dependent_var, independent_vars, random_effects_vars, grouping_vars,
        var_names, random_effects_names, grouping_names, fit_method
    )

@mcp.tool()
def perform_multiple_imputation_tool(
    data: List[Union[float, None]] = Field(..., description="数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)，缺失值用null表示"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    method: str = Field("mice", description="插补方法: 'mice'(多重插补), 'mean'(均值插补), 'median'(中位数插补), 'mode'(众数插补), 'knn'(K近邻插补)"),
    n_imputations: int = Field(5, description="插补次数，仅对'mice'方法有效"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
):
    """
    执行缺失值多重插补分析
    """
    return perform_multiple_imputation(data, var_names, method, n_imputations, random_state)

@mcp.tool()
def perform_survival_analysis_tool(
    time_var: List[float] = Field(..., description="时间变量"),
    event_var: List[int] = Field(..., description="事件指示变量（1表示事件发生，0表示删失）"),
    group_var: Optional[List[int]] = Field(None, description="分组变量（用于Kaplan-Meier曲线和Log-rank检验）"),
    covariates: Optional[List[float]] = Field(None, description="协变量数据，所有协变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    covariate_names: Optional[List[str]] = Field(None, description="协变量名称列表"),
    confidence_level: float = Field(0.95, description="置信区间水平，如0.95表示95%置信区间"),
):
    """
    执行生存分析，包括Kaplan-Meier生存曲线估计、Log-rank检验和Cox比例风险模型
    """
    return perform_survival_analysis(time_var, event_var, group_var, covariates, covariate_names, confidence_level)

@mcp.tool()
def perform_causal_inference_tool(
    data: List[float] = Field(..., description="样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)"),
    var_names: List[str] = Field(..., description="变量名称列表"),
    treatment_var: str = Field(..., description="处理变量名称"),
    outcome_var: str = Field(..., description="结果变量名称"),
    confounding_vars: List[str] = Field(..., description="混杂变量名称列表"),
    method: str = Field("ipw", description="因果推断方法: 'ipw'(逆概率加权), 'matching'(匹配), 'regression'(回归调整)"),
    bootstrap_samples: int = Field(1000, description="Bootstrap抽样次数，用于计算置信区间"),
    random_state: Optional[int] = Field(None, description="随机种子，用于结果可重现"),
):
    """
    执行因果推断分析
    """
    return perform_causal_inference(data, var_names, treatment_var, outcome_var, confounding_vars, method, bootstrap_samples, random_state)

@mcp.tool()
def perform_nonparametric_regression_tool(
    x_data: List[float] = Field(..., description="自变量数据"),
    y_data: List[float] = Field(..., description="因变量数据"),
    method: str = Field("loess", description="非参数回归方法: 'loess'(局部加权回归), 'spline'(样条回归)"),
    loess_frac: float = Field(0.3, description="LOESS方法中用于局部回归的窗口大小比例，值越大越平滑"),
    loess_it: int = Field(3, description="LOESS方法的迭代次数"),
    spline_degree: int = Field(3, description="样条回归的多项式阶数"),
    spline_smooth_factor: Optional[float] = Field(None, description="样条回归的平滑因子，值越大越平滑，None表示自动选择"),
    confidence_level: float = Field(0.95, description="置信区间水平，如0.95表示95%置信区间"),
):
    """
    执行非参数/半参数回归分析（LOESS/LOWESS、样条回归）
    """
    return perform_nonparametric_regression(x_data, y_data, method, loess_frac, loess_it, 
                                          spline_degree, spline_smooth_factor, confidence_level)

# 添加静态文件路由
@mcp.custom_route("/generated_files/{filename:path}", methods=["GET"])
async def serve_static_files(request: Request):
    filename = request.path_params["filename"]
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        # 返回404响应
        return JSONResponse({"error": "File not found"}, status_code=404)

# 添加健康检查端点
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    return JSONResponse({
        "status": "healthy",
        "service": "multivariate-statistics-mcp-server",
        "version": "1.0.0"
    })


# # 添加根路径健康检查端点
# @mcp.custom_route("/", methods=["GET"])
# async def root_health_check(request: Request):
#     return JSONResponse({
#         "status": "healthy",
#         "service": "multivariate-statistics-mcp-server",
#         "version": "1.0.0"
#     })

def main():
    # 启动文件自动清理任务
    start_cleanup_scheduler()
    
    # 如果环境变量指定了SERVER_HOST，则使用环境变量的值
    host = os.environ.get("MCP_STATS_SERVER_HOST", SERVER_HOST)
    port = int(os.environ.get("MCP_STATS_SERVER_PORT", SERVER_PORT))
    transport = os.environ.get("MCP_STATS_SERVER_TRANSPORT", SERVER_TRANSPORT)
    
    print(f"Starting server on {host}:{port} with transport {transport}")
    print(f"Base URL: {BASE_URL}")
    
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=SERVER_PATH,
        log_level=SERVER_LOG_LEVEL,
        # strict_accept=False,
    )


if __name__ == "__main__":
    main()