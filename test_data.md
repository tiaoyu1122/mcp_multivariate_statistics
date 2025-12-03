# 多元统计分析工具测试数据

本文档包含了所有工具的测试数据示例，用于验证各个统计分析工具的功能。

## 1. 多元线性回归分析 (Multiple Regression)

### 工具名称
perform_multiple_regression

### 功能描述
执行多元线性回归分析，包含统计显著性检验和残差分析

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，每个子列表代表一个自变量
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名

### 测试数据
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2]
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses"]
}
```

## 2. 逐步回归分析 (Stepwise Regression)

### 工具名称
perform_stepwise_regression

### 功能描述
执行逐步回归分析，包含统计显著性检验和残差分析

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，每个子列表代表一个自变量
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名
- method: 逐步回归方法 ('forward', 'backward', 'both')
- significance_level_enter: 变量进入模型的显著性水平
- significance_level_remove: 变量移出模型的显著性水平

### 测试数据
```json
{
  "dependent_var": [12.3, 15.7, 18.2, 21.5, 24.8, 27.1, 30.4, 33.6, 36.9, 40.2, 43.5, 46.8],
  "independent_vars": [
    [1.5, 2.1, 2.8, 3.6, 4.2, 4.9, 5.7, 6.3, 7.1, 7.8, 8.5, 9.2],
    [0.8, 1.2, 1.7, 2.1, 2.6, 3.0, 3.5, 3.9, 4.4, 4.8, 5.3, 5.7],
    [2.1, 2.9, 3.8, 4.6, 5.5, 6.3, 7.2, 8.0, 8.9, 9.7, 10.6, 11.4]
  ],
  "var_names": ["production", "worker_salary", "raw_material_cost", "equipment_depreciation"],
  "method": "both",
  "significance_level_enter": 0.05,
  "significance_level_remove": 0.10
}
```

## 3. Hotelling's T²检验 (Hotelling's T² Test)

### 工具名称
perform_hotelling_t2_test

### 功能描述
执行Hotelling's T²检验，用于检验两个多维正态分布总体的均值向量是否相等

### 参数说明
- group1_data: 第一组样本数据，每个子列表代表一个变量的所有观测值
- group2_data: 第二组样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表

### 测试数据
```json
{
  "group1_data": [
    [12.5, 13.2, 14.1, 13.8, 12.9, 13.5, 14.0, 13.7, 12.8, 13.3],
    [8.1, 8.5, 9.2, 8.9, 8.3, 8.7, 9.0, 8.8, 8.2, 8.6],
    [15.2, 16.1, 15.8, 16.3, 15.5, 15.9, 16.0, 15.7, 15.4, 15.6]
  ],
  "group2_data": [
    [10.5, 11.2, 10.8, 11.5, 10.9, 11.3, 11.0, 11.4, 10.7, 11.1],
    [7.2, 7.8, 7.5, 8.0, 7.6, 7.9, 7.7, 7.4, 7.3, 7.1],
    [13.5, 14.2, 13.8, 14.5, 14.0, 14.1, 13.9, 14.3, 13.7, 14.4]
  ],
  "var_names": ["height", "weight", "vital_capacity"]
}
```

## 4. 多元正态性检验 (Multivariate Normality Test)

### 工具名称
perform_multivariate_normality_test

### 功能描述
执行多元正态性检验，检验多变量数据是否符合多元正态分布

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表

### 测试数据
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8, 9.4, 10.1]
  ],
  "var_names": ["variable_a", "variable_b", "variable_c"]
}
```

## 5. 协方差矩阵齐性检验 (Covariance Homogeneity Test)

### 工具名称
perform_covariance_homogeneity_test

### 功能描述
执行协方差矩阵齐性检验(Bartlett检验)，用于检验多个多元正态分布总体的协方差矩阵是否相等

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2, 5.1],
      [0.5, 1.0, 1.2, 1.8, 2.1, 2.7],
      [3.1, 4.2, 4.8, 5.3, 6.1, 6.8]
    ],
    [
      [2.2, 3.0, 3.7, 4.4, 5.1, 5.9, 6.6],
      [1.3, 1.8, 2.1, 2.7, 3.0, 3.6, 3.9],
      [4.0, 5.1, 5.7, 6.2, 6.9, 7.6, 8.1]
    ],
    [
      [0.8, 1.5, 2.2, 2.9, 3.6],
      [0.3, 0.8, 1.1, 1.6, 2.0],
      [2.5, 3.4, 4.0, 4.6, 5.2]
    ]
  ],
  "group_names": ["group_a", "group_b", "group_c"],
  "var_names": ["variable_x", "variable_y", "variable_z"]
}
```

## 6. 多元方差分析 (MANOVA)

### 工具名称
perform_manova

### 功能描述
执行多元方差分析(MANOVA)，用于检验多个组在多个因变量上的均值向量是否存在显著差异

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表

### 测试数据
```json
{
  "groups_data": [
    [
      [85, 87, 80, 90, 88, 82],
      [75, 78, 72, 80, 79, 74],
      [92, 95, 88, 96, 94, 90]
    ],
    [
      [78, 80, 75, 82, 81, 77, 79],
      [70, 72, 68, 74, 73, 69, 71],
      [85, 88, 82, 89, 87, 84, 86]
    ],
    [
      [90, 92, 88, 94, 93],
      [82, 84, 80, 86, 85],
      [96, 98, 94, 100, 99]
    ]
  ],
  "group_names": ["class_a", "class_b", "class_c"],
  "var_names": ["chinese_score", "math_score", "english_score"]
}
```

## 7. Box's M检验 (Box's M Test)

### 工具名称
perform_box_m_test

### 功能描述
执行Box's M检验，用于检验多个多元正态分布总体的协方差矩阵是否相等

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表

### 测试数据
```json
{
  "groups_data": [
    [
      [2.5, 3.1, 2.8, 3.5, 3.2, 2.9],
      [1.2, 1.5, 1.3, 1.8, 1.6, 1.4],
      [4.1, 4.5, 4.2, 4.8, 4.6, 4.3]
    ],
    [
      [3.2, 3.8, 3.5, 4.1, 3.9, 3.6, 4.0],
      [1.8, 2.1, 1.9, 2.4, 2.2, 2.0, 2.3],
      [5.2, 5.6, 5.3, 5.9, 5.7, 5.4, 5.8]
    ],
    [
      [1.8, 2.2, 2.0, 2.5, 2.3],
      [0.9, 1.1, 1.0, 1.3, 1.2],
      [3.5, 3.9, 3.7, 4.2, 4.0]
    ]
  ],
  "group_names": ["region_a", "region_b", "region_c"],
  "var_names": ["temperature", "humidity", "pressure"]
}
```

## 8. 分类变量独立性检验 (Categorical Independence Test)

### 工具名称
perform_categorical_independence_test

### 功能描述
执行卡方独立性检验，用于检验两个分类变量是否相互独立

### 参数说明
- observed_frequencies: 观测频数表，二维列表，每个子列表代表一行
- row_labels: 行标签列表
- col_labels: 列标签列表

### 测试数据
```json
{
  "observed_frequencies": [
    [20, 15, 10],
    [12, 25, 18],
    [8, 10, 22]
  ],
  "row_labels": ["age_group_young", "age_group_middle", "age_group_old"],
  "col_labels": ["support", "oppose", "neutral"]
}
```

## 9. 连续变量相关性检验 (Continuous Independence Test)

### 工具名称
perform_continuous_independence_test

### 功能描述
执行连续变量相关性检验，用于检验多个连续变量之间的线性相关性

### 参数说明
- data: 连续变量数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表

### 测试数据
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8]
  ],
  "var_names": ["sales_volume", "advertising_investment", "customer_satisfaction"]
}
```

## 10. Fisher判别分析 (Fisher Discriminant Analysis)

### 工具名称
perform_fisher_discriminant_analysis

### 功能描述
执行Fisher判别分析，用于寻找最能区分不同组的线性组合

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2],
      [0.5, 1.0, 1.2, 1.8, 2.1],
      [3.1, 4.2, 4.8, 5.3, 6.1]
    ],
    [
      [5.2, 6.1, 6.8, 7.5, 8.2],
      [2.5, 3.0, 3.2, 3.8, 4.1],
      [7.1, 8.2, 8.8, 9.3, 10.1]
    ],
    [
      [9.2, 10.1, 10.8, 11.5, 12.2],
      [4.5, 5.0, 5.2, 5.8, 6.1],
      [11.1, 12.2, 12.8, 13.3, 14.1]
    ]
  ],
  "group_names": ["low_income", "middle_income", "high_income"],
  "var_names": ["consumption_expenditure", "savings_amount", "investment_amount"]
}
```

## 11. 距离判别分析 (Distance Discriminant Analysis)

### 工具名称
perform_distance_discriminant_analysis

### 功能描述
执行距离判别分析，基于马氏距离进行分类判别

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表
- test_data: 待判别样本数据，每个子列表代表一个变量的所有观测值

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2],
      [0.5, 1.0, 1.2, 1.8, 2.1],
      [3.1, 4.2, 4.8, 5.3, 6.1]
    ],
    [
      [5.2, 6.1, 6.8, 7.5, 8.2],
      [2.5, 3.0, 3.2, 3.8, 4.1],
      [7.1, 8.2, 8.8, 9.3, 10.1]
    ],
    [
      [9.2, 10.1, 10.8, 11.5, 12.2],
      [4.5, 5.0, 5.2, 5.8, 6.1],
      [11.1, 12.2, 12.8, 13.3, 14.1]
    ]
  ],
  "group_names": ["low_income", "middle_income", "high_income"],
  "var_names": ["consumption_expenditure", "savings_amount", "investment_amount"],
  "test_data": [
    [3.2, 4.1, 5.0],
    [1.5, 2.0, 2.5],
    [5.2, 6.0, 6.8]
  ]
}
```

## 12. 贝叶斯判别分析 (Bayes Discriminant Analysis)

### 工具名称
perform_bayes_discriminant_analysis

### 功能描述
执行贝叶斯判别分析，基于贝叶斯定理和多元正态分布假设进行分类判别

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表
- test_data: 待判别样本数据，每个子列表代表一个变量的所有观测值
- prior_probabilities: 先验概率列表，长度应与组数相同

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2],
      [0.5, 1.0, 1.2, 1.8, 2.1],
      [3.1, 4.2, 4.8, 5.3, 6.1]
    ],
    [
      [5.2, 6.1, 6.8, 7.5, 8.2],
      [2.5, 3.0, 3.2, 3.8, 4.1],
      [7.1, 8.2, 8.8, 9.3, 10.1]
    ],
    [
      [9.2, 10.1, 10.8, 11.5, 12.2],
      [4.5, 5.0, 5.2, 5.8, 6.1],
      [11.1, 12.2, 12.8, 13.3, 14.1]
    ]
  ],
  "group_names": ["low_income", "middle_income", "high_income"],
  "var_names": ["consumption_expenditure", "savings_amount", "investment_amount"],
  "test_data": [
    [3.2, 4.1, 5.0],
    [1.5, 2.0, 2.5],
    [5.2, 6.0, 6.8]
  ],
  "prior_probabilities": [0.3, 0.4, 0.3]
}
```

## 13. 广义平方距离判别分析 (Generalized Square Distance Discriminant Analysis)

### 工具名称
perform_generalized_square_distance_discriminant_analysis

### 功能描述
执行广义平方距离判别分析，考虑各组协方差矩阵不等的情况

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表
- test_data: 待判别样本数据，每个子列表代表一个变量的所有观测值

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2],
      [0.5, 1.0, 1.2, 1.8, 2.1],
      [3.1, 4.2, 4.8, 5.3, 6.1]
    ],
    [
      [5.2, 6.1, 6.8, 7.5, 8.2],
      [2.5, 3.0, 3.2, 3.8, 4.1],
      [7.1, 8.2, 8.8, 9.3, 10.1]
    ],
    [
      [9.2, 10.1, 10.8, 11.5, 12.2],
      [4.5, 5.0, 5.2, 5.8, 6.1],
      [11.1, 12.2, 12.8, 13.3, 14.1]
    ]
  ],
  "group_names": ["low_income", "middle_income", "high_income"],
  "var_names": ["consumption_expenditure", "savings_amount", "investment_amount"],
  "test_data": [
    [3.2, 4.1, 5.0],
    [1.5, 2.0, 2.5],
    [5.2, 6.0, 6.8]
  ]
}
```

## 14. 逐步判别分析 (Stepwise Discriminant Analysis)

### 工具名称
perform_stepwise_discriminant_analysis

### 功能描述
执行逐步判别分析，通过逐步选择变量构建最优判别函数

### 参数说明
- groups_data: 多组样本数据，每个元素是一组数据，每组数据中每个子列表代表一个变量的所有观测值
- group_names: 组名称列表
- var_names: 变量名称列表
- method: 逐步选择方法 ('wilks' 或 'f')
- significance_level_enter: 变量进入模型的显著性水平
- significance_level_remove: 变量移出模型的显著性水平

### 测试数据
```json
{
  "groups_data": [
    [
      [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8],
      [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0],
      [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2],
      [2.1, 3.0, 3.5, 4.0, 4.8, 5.2, 5.7]
    ],
    [
      [5.2, 6.1, 6.8, 7.5, 8.2, 8.9, 9.5],
      [2.5, 3.0, 3.2, 3.8, 4.1, 4.6, 5.0],
      [7.1, 8.2, 8.8, 9.3, 10.1, 10.7, 11.2],
      [4.2, 5.0, 5.5, 6.1, 6.8, 7.3, 7.9]
    ],
    [
      [9.2, 10.1, 10.8, 11.5, 12.2, 12.9, 13.5],
      [4.5, 5.0, 5.2, 5.8, 6.1, 6.6, 7.0],
      [11.1, 12.2, 12.8, 13.3, 14.1, 14.7, 15.2],
      [6.5, 7.2, 7.8, 8.3, 9.0, 9.5, 10.1]
    ]
  ],
  "group_names": ["low_income", "middle_income", "high_income"],
  "var_names": ["consumption_expenditure", "savings_amount", "investment_amount", "education_expenditure"],
  "method": "wilks",
  "significance_level_enter": 0.05,
  "significance_level_remove": 0.10
}
```

## 15. K-Means聚类分析 (K-Means Clustering)

### 工具名称
perform_kmeans_clustering

### 功能描述
执行K-Means聚类分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_clusters: 聚类数量
- init_method: 初始化方法 ('k-means++' 或 'random')
- max_iter: 最大迭代次数
- n_init: 运行算法的次数，返回最好的结果
- random_state: 随机种子，用于结果可重现

### 测试数据
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c"],
  "n_clusters": 3,
  "init_method": "k-means++",
  "max_iter": 300,
  "n_init": 10,
  "random_state": 42
}
```

## 16. K-Medoids聚类分析 (K-Medoids Clustering)

### 工具名称
perform_kmedoids_clustering

### 功能描述
执行K-Medoids聚类分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_clusters: 聚类数量
- max_iter: 最大迭代次数
- random_state: 随机种子，用于结果可重现

### 测试数据
```json
{
  "data": [
    [1.5, 2.3, 3.1, 4.2, 5.0, 5.8, 6.7, 7.5, 8.2],
    [0.8, 1.2, 1.9, 2.5, 3.0, 3.7, 4.1, 4.8, 5.3],
    [2.2, 3.5, 4.0, 5.1, 6.2, 7.0, 7.8, 8.5, 9.1]
  ],
  "var_names": ["attribute_x", "attribute_y", "attribute_z"],
  "n_clusters": 3,
  "max_iter": 300,
  "random_state": 42
}
```

## 17. DBSCAN聚类分析 (DBSCAN Clustering)

### 工具名称
perform_dbscan_clustering

### 功能描述
执行DBSCAN聚类分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- eps: 邻域半径
- min_samples: 核心点邻域中的最小样本数

### 测试数据
```json
{
  "data": [
    [1.0, 1.5, 2.0, 2.5, 3.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    [1.0, 1.2, 1.8, 2.2, 2.8, 5.8, 6.2, 6.8, 7.2, 7.8],
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  ],
  "var_names": ["dimension_x", "dimension_y", "dimension_z"],
  "eps": 1.0,
  "min_samples": 3
}
```

## 18. 层次聚类分析 (Hierarchical Clustering)

### 工具名称
perform_hierarchical_clustering

### 功能描述
执行凝聚式层次聚类分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_clusters: 聚类数量
- linkage_method: 链接方法 ('ward', 'complete', 'average', 'single')

### 测试数据
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2]
  ],
  "var_names": ["metric_a", "metric_b", "metric_c"],
  "n_clusters": 3,
  "linkage_method": "ward"
}
```

## 19. 高斯混合模型聚类分析 (GMM Clustering)

### 工具名称
perform_gmm_clustering

### 功能描述
执行高斯混合模型(GMM)聚类分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_components: 混合成分数量（相当于聚类数）
- covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
- max_iter: 最大迭代次数
- random_state: 随机种子，用于结果可重现

### 测试数据
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["variable_x", "variable_y", "variable_z"],
  "n_components": 3,
  "covariance_type": "full",
  "max_iter": 100,
  "random_state": 42
}
```

## 20. 主成分分析 (Principal Component Analysis)

### 工具名称
perform_pca

### 功能描述
执行主成分分析(PCA)

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_components: 主成分数量，如果不指定则保留所有成分
- standardize: 是否标准化数据

### 测试数据
```json
{
  "data": [
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4],
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c", "feature_d"],
  "n_components": 2,
  "standardize": true
}
```

## 21. 因子分析 (Factor Analysis)

### 工具名称
perform_factor_analysis

### 功能描述
执行因子分析

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_factors: 因子数量，如果不指定则使用默认方法确定
- rotation: 因子旋转方法 ('varimax', 'promax', None)
- standardize: 是否标准化数据

### 测试数据
```json
{
  "data": [
    [2.1, 3.2, 4.1, 5.0, 6.2, 7.1, 8.0, 9.1],
    [1.5, 2.3, 3.0, 3.8, 4.6, 5.4, 6.2, 7.0],
    [0.8, 1.4, 2.0, 2.6, 3.2, 3.8, 4.4, 5.0],
    [3.2, 4.3, 5.1, 6.0, 6.8, 7.7, 8.5, 9.4],
    [1.0, 1.8, 2.5, 3.2, 4.0, 4.7, 5.5, 6.2]
  ],
  "var_names": ["variable_1", "variable_2", "variable_3", "variable_4", "variable_5"],
  "n_factors": 2,
  "rotation": "varimax",
  "standardize": true
}
```

## 22. 对应分析 (Correspondence Analysis)

### 工具名称
perform_correspondence_analysis

### 功能描述
执行对应分析 (Correspondence Analysis)

### 参数说明
- observed_frequencies: 观测频数表，二维列表，每个子列表代表一行
- row_labels: 行标签列表
- col_labels: 列标签列表

### 测试数据
```json
{
  "observed_frequencies": [
    [15, 20, 10],
    [25, 10, 15],
    [10, 15, 20]
  ],
  "row_labels": ["group_a", "group_b", "group_c"],
  "col_labels": ["category_x", "category_y", "category_z"]
}
```

## 23. 典型相关分析 (Canonical Correlation Analysis)

### 工具名称
perform_canonical_correlation_analysis

### 功能描述
执行典型相关分析 (Canonical Correlation Analysis)

### 参数说明
- x_data: 第一组变量数据，每个子列表代表一个变量的所有观测值
- y_data: 第二组变量数据，每个子列表代表一个变量的所有观测值
- x_var_names: 第一组变量名称列表
- y_var_names: 第二组变量名称列表
- standardize: 是否标准化数据

### 测试数据
```json
{
  "x_data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2]
  ],
  "y_data": [
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1],
    [1.8, 2.5, 3.0, 3.8, 4.5, 5.2, 5.9]
  ],
  "x_var_names": ["x_variable_1", "x_variable_2", "x_variable_3"],
  "y_var_names": ["y_variable_1", "y_variable_2"],
  "standardize": true
}
```

## 24. 偏最小二乘回归分析 (PLS Regression)

### 工具名称
perform_pls_regression

### 功能描述
执行偏最小二乘回归分析 (PLS Regression)

### 参数说明
- x_data: 自变量数据，每个子列表代表一个变量的所有观测值
- y_data: 因变量数据，每个子列表代表一个变量的所有观测值
- x_var_names: 自变量名称列表
- y_var_names: 因变量名称列表
- n_components: 成分数量，如果不指定则自动选择
- standardize: 是否标准化数据

### 测试数据
```json
{
  "x_data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "y_data": [
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4],
    [1.8, 2.7, 3.2, 4.3, 5.4, 6.9, 7.2, 8.5]
  ],
  "x_var_names": ["predictor_a", "predictor_b", "predictor_c"],
  "y_var_names": ["response_var_1", "response_var_2"],
  "n_components": 2,
  "standardize": true
}
```

## 25. 时间序列分析 (Time Series Analysis)

### 工具名称
perform_time_series_analysis

### 功能描述
执行时间序列分析

### 参数说明
- time_series: 时间序列数据
- time_labels: 时间标签列表
- model_type: 模型类型 ('auto_arima', 'sarima', 'exponential_smoothing', 'manual_arima')
- forecast_steps: 预测步数
- seasonal_period: 季节性周期
- order: ARIMA模型的(p,d,q)参数
- seasonal_order: 季节性ARIMA模型的(P,D,Q,s)参数

### 测试数据
```json
{
  "time_series": [10, 12, 13, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55],
  "time_labels": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08"],
  "model_type": "auto_arima",
  "forecast_steps": 5,
  "seasonal_period": 12
}
```

## 26. 时间序列预处理检验 (Time Series Preprocessing Tests)

### 工具名称
perform_time_series_preprocessing_tests

### 功能描述
执行时间序列预处理检验，用于判断时间序列是否适合建模，包括平稳性检验、正态性检验、自相关检验等

### 参数说明
- time_series: 时间序列数据
- time_labels: 时间标签列表
- seasonal_period: 季节性周期（如月度数据为12，季度数据为4）
- test_types: 要执行的检验类型列表: 'adf'(ADF平稳性检验), 'kpss'(KPSS平稳性检验), 'normality'(正态性检验), 'autocorrelation'(自相关检验)

### 测试数据
```json
{
  "time_series": [10, 12, 13, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55],
  "time_labels": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08"],
  "seasonal_period": 12,
  "test_types": ["adf", "kpss", "normality", "autocorrelation"]
}
```

## 27. 正则化回归分析 (Regularized Regression)

### 工具名称
perform_regularized_regression

### 功能描述
执行正则化回归分析（岭回归、套索回归、弹性网络回归），用于处理多重共线性问题和变量选择

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，每个子列表代表一个自变量
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名
- method: 正则化方法: 'ridge'(岭回归), 'lasso'(套索回归), 'elastic_net'(弹性网络)
- alpha: 正则化强度参数
- l1_ratio: Elastic Net中L1正则化的比例 (仅用于elastic_net方法)
- standardize: 是否标准化数据
- fit_intercept: 是否计算截距

### 测试数据（Ridge回归）
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8]
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses", "market_potential"],
  "method": "ridge",
  "alpha": 1.0,
  "l1_ratio": 0.5,
  "standardize": true,
  "fit_intercept": true
}
```

### 测试数据 (Lasso回归)
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8]
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses", "market_potential"],
  "method": "lasso",
  "alpha": 0.1,
  "l1_ratio": 0.5,
  "standardize": true,
  "fit_intercept": true
}
```

### 测试数据 (Elastic Net回归)
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8]
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses", "market_potential"],
  "method": "elastic_net",
  "alpha": 0.5,
  "l1_ratio": 0.7,
  "standardize": true,
  "fit_intercept": true
}
```

## 28. 广义线性模型分析 (Generalized Linear Model)

### 工具名称
perform_generalized_linear_model

### 功能描述
执行广义线性模型分析，包括逻辑回归、多项逻辑回归、有序逻辑回归、泊松回归和负二项回归

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，每个子列表代表一个自变量
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名
- model_type: 模型类型: 'logistic'(二分类逻辑回归), 'multinomial'(多项逻辑回归), 'ordinal'(有序逻辑回归), 'poisson'(泊松回归), 'negative_binomial'(负二项回归)

### 测试数据 (二分类逻辑回归)
```json
{
  "dependent_var": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["outcome", "variable_a", "variable_b"],
  "model_type": "logistic"
}
```

### 测试数据 (多项逻辑回归)
```json
{
  "dependent_var": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["category", "variable_a", "variable_b"],
  "model_type": "multinomial"
}
```

### 测试数据 (有序逻辑回归)
```json
{
  "dependent_var": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["rating", "variable_a", "variable_b"],
  "model_type": "ordinal"
}
```

### 测试数据 (泊松回归)
```json
{
  "dependent_var": [0, 1, 2, 1, 3, 2, 1, 0, 2, 3, 1, 2],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["count_events", "variable_a", "variable_b"],
  "model_type": "poisson"
}
```

### 测试数据 (负二项回归)
```json
{
  "dependent_var": [0, 1, 2, 1, 3, 2, 1, 0, 2, 3, 1, 2],
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["count_events", "variable_a", "variable_b"],
  "model_type": "negative_binomial"
}
```

## 29. 多维尺度分析 (Multidimensional Scaling)

### 工具名称
perform_mds

### 功能描述
执行多维尺度分析(MDS)，用于降维和可视化高维数据，保持样本间的距离关系

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_components: 降维后的维度数，默认为2
- metric: 是否使用度量MDS，True表示经典MDS，False表示非度量MDS，默认为True
- n_init: 初始化次数，用于寻找最佳解，默认为4
- max_iter: 最大迭代次数，默认为300
- dissimilarity: 距离度量方法: 'euclidean'(欧氏距离), 'precomputed'(预先计算的距离矩阵)
- standardize: 是否标准化数据，默认为True

### 测试数据 (经典MDS)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c"],
  "n_components": 2,
  "metric": true,
  "n_init": 4,
  "max_iter": 300,
  "dissimilarity": "euclidean",
  "standardize": true
}
```

### 测试数据 (非度量MDS)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c"],
  "n_components": 2,
  "metric": false,
  "n_init": 4,
  "max_iter": 300,
  "dissimilarity": "euclidean",
  "standardize": true
}
```

## 30. UMAP非线性降维分析 (UMAP Nonlinear Dimensionality Reduction)

### 工具名称
perform_umap

### 功能描述
执行UMAP非线性降维分析，用于高维数据的可视化和降维，能够更好地保持数据的局部和全局结构

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- n_neighbors: 邻居数量，控制局部与全局结构的平衡，默认为15
- n_components: 降维后的维度数，默认为2
- min_dist: 最小距离，控制簇的紧密程度，默认为0.1
- metric: 距离度量方法，默认为"euclidean"
- random_state: 随机种子，用于结果可重现，默认为42
- standardize: 是否标准化数据，默认为True

### 测试数据 (2D降维)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c"],
  "n_neighbors": 15,
  "n_components": 2,
  "min_dist": 0.1,
  "metric": "euclidean",
  "random_state": 42,
  "standardize": true
}
```

### 测试数据 (3D降维)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["feature_a", "feature_b", "feature_c"],
  "n_neighbors": 15,
  "n_components": 3,
  "min_dist": 0.1,
  "metric": "euclidean",
  "random_state": 42,
  "standardize": true
}
```

## 42. 稳健回归分析 (Robust Regression)

### 工具名称
perform_robust_regression

### 功能描述
执行稳健回归分析（RANSAC回归或Huber回归），用于处理包含异常值的数据集，提高回归模型的鲁棒性

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名
- method: 稳健回归方法: 'ransac'(RANSAC回归), 'huber'(Huber回归)
- standardize: 是否标准化数据，默认为True
- fit_intercept: 是否计算截距，默认为True
- min_samples: RANSAC算法中随机样本的最小数量
- residual_threshold: RANSAC算法中样本被视为内点的最大残差
- max_trials: RANSAC算法的最大迭代次数，默认为100
- epsilon: Huber回归的参数，决定对异常值的敏感度，默认为1.35
- alpha: Huber回归的正则化强度，默认为0.0001

### 测试数据 (RANSAC回归)
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 20.0],
  "independent_vars": [
    1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 9.0,
    0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 1.0
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses"],
  "method": "ransac",
  "standardize": true,
  "fit_intercept": true,
  "max_trials": 100
}
```

### 测试数据 (Huber回归)
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 20.0],
  "independent_vars": [
    1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 9.0,
    0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 1.0
  ],
  "var_names": ["sales_volume", "advertising_investment", "promotion_expenses"],
  "method": "huber",
  "standardize": true,
  "fit_intercept": true,
  "epsilon": 1.35,
  "alpha": 0.0001
}
```
  "time_series_list": [
    [10, 12, 13, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55],
    [8, 10, 11, 13, 16, 18, 20, 23, 26, 28, 30, 33, 36, 38, 40, 43, 46, 48, 50, 53],
    [12, 14, 15, 17, 20, 22, 24, 27, 30, 32, 34, 37, 40, 42, 44, 47, 50, 52, 54, 57]
  ],
  "series_names": ["gdp", "consumption", "investment"],
  "max_lag": 3,
  "significance_level": 0.05
}
```

## 33. 数据标准化 (Data Standardization)

### 工具名称
perform_data_standardization

### 功能描述
执行数据标准化（Z-score标准化或Min-Max标准化）

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- method: 标准化方法 ('zscore', 'minmax')
- feature_range: Min-Max标准化的目标范围，格式为[min, max]

### 测试数据 (Z-score标准化)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2]
  ],
  "var_names": ["sales_revenue", "advertising_costs", "employee_count"],
  "method": "zscore",
  "feature_range": [0, 1]
}
```

### 测试数据 (Min-Max标准化)
```json
{
  "data": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2]
  ],
  "var_names": ["sales_revenue", "advertising_costs", "employee_count"],
  "method": "minmax",
  "feature_range": [0, 1]
}
```

## 34. 多重共线性诊断 (Multicollinearity Diagnosis)

### 工具名称
perform_multicollinearity_diagnosis

### 功能描述
执行多重共线性诊断（方差膨胀因子VIF分析）

### 参数说明
- independent_vars: 自变量数据，每个子列表代表一个自变量
- var_names: 自变量名称列表
- vif_threshold: VIF阈值，用于判断是否存在多重共线性

### 测试数据
```json
{
  "independent_vars": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7],
    [1.0, 2.0, 2.9, 3.6, 4.3, 5.2, 5.9, 6.8],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9]
  ],
  "var_names": ["advertising_expenditure", "marketing_budget", "staff_count", "product_price"],
  "vif_threshold": 5.0
}
```

## 35. 混合效应模型 (Mixed Effects Model)

### 工具名称
perform_mixed_effects_model

### 功能描述
执行混合效应模型分析（Mixed Effects Models / Hierarchical Models）

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 固定效应自变量数据，每个子列表代表一个自变量
- random_effects_vars: 随机效应变量数据，每个子列表代表一个随机效应变量
- grouping_vars: 分组变量数据，每个子列表代表一个分组变量
- var_names: 变量名称列表，第一个为因变量名，后续为固定效应自变量名
- random_effects_names: 随机效应变量名称列表
- grouping_names: 分组变量名称列表
- fit_method: 拟合方法 ('ml', 'reml')

### 测试数据
```json
{
  "dependent_var": [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
  "independent_vars": [
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0],
    [3.1, 4.2, 4.8, 5.3, 6.1, 6.8, 7.2, 7.9, 8.3, 8.8, 9.4, 10.1]
  ],
  "random_effects_vars": [
    [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
  ],
  "grouping_vars": [
    [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
  ],
  "var_names": ["test_score", "study_hours", "attendance_rate"],
  "random_effects_names": ["intercept"],
  "grouping_names": ["class_id"],
  "fit_method": "reml"
}
```

## 36. 多重插补 (Multiple Imputation)

### 工具名称
perform_multiple_imputation

### 功能描述
执行缺失值多重插补分析

### 参数说明
- data: 数据，每个子列表代表一个变量的所有观测值，缺失值用null表示
- var_names: 变量名称列表
- method: 插补方法 ('mice', 'mean', 'median', 'mode', 'knn')
- n_imputations: 插补次数，仅对'mice'方法有效
- random_state: 随机种子，用于结果可重现

### 测试数据
```json
{
  "data": [
    [1.2, null, 2.8, 3.5, 4.2, null, 5.8, 6.7],
    [0.5, 1.0, null, 1.8, 2.1, 2.7, 3.0, null],
    [3.1, 4.2, 4.8, null, 6.1, 6.8, null, 7.9]
  ],
  "var_names": ["variable_a", "variable_b", "variable_c"],
  "method": "mice",
  "n_imputations": 5,
  "random_state": 42
}
```

## 37. 生存分析 (Survival Analysis)

### 工具名称
perform_survival_analysis

### 功能描述
执行生存分析，包括Kaplan-Meier生存曲线估计、Log-rank检验和Cox比例风险模型

### 参数说明
- time_var: 时间变量
- event_var: 事件指示变量（1表示事件发生，0表示删失）
- group_var: 分组变量（用于Kaplan-Meier曲线和Log-rank检验）
- covariates: 协变量数据（用于Cox比例风险模型），每个子列表代表一个协变量
- covariate_names: 协变量名称列表
- confidence_level: 置信区间水平，如0.95表示95%置信区间

### 测试数据
```json
{
  "time_var": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
  "event_var": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
  "group_var": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
  "covariates": [
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2]
  ],
  "covariate_names": ["treatment_group", "age"],
  "confidence_level": 0.95
}
```



## 38. 因果推断分析 (Causal Inference Analysis)

### 工具名称
perform_causal_inference

### 功能描述
执行因果推断分析，支持逆概率加权(IPW)、匹配(matching)和回归调整(regression)三种方法

### 参数说明
- data: 样本数据，每个子列表代表一个变量的所有观测值
- var_names: 变量名称列表
- treatment_var: 处理变量名称
- outcome_var: 结果变量名称
- confounding_vars: 混杂变量名称列表
- method: 因果推断方法: 'ipw'(逆概率加权), 'matching'(匹配), 'regression'(回归调整)
- bootstrap_samples: Bootstrap抽样次数，用于计算置信区间
- random_state: 随机种子，用于结果可重现

### 测试数据 (逆概率加权方法)
```json
{
  "data": [
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["treatment", "outcome", "confounder_a", "confounder_b"],
  "treatment_var": "treatment",
  "outcome_var": "outcome",
  "confounding_vars": ["confounder_a", "confounder_b"],
  "method": "ipw",
  "bootstrap_samples": 1000,
  "random_state": 42
}
```

### 测试数据 (匹配方法)
```json
{
  "data": [
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["treatment", "outcome", "confounder_a", "confounder_b"],
  "treatment_var": "treatment",
  "outcome_var": "outcome",
  "confounding_vars": ["confounder_a", "confounder_b"],
  "method": "matching",
  "bootstrap_samples": 1000,
  "random_state": 42
}
```

### 测试数据 (回归调整方法)
```json
{
  "data": [
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
    [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
    [0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0]
  ],
  "var_names": ["treatment", "outcome", "confounder_a", "confounder_b"],
  "treatment_var": "treatment",
  "outcome_var": "outcome",
  "confounding_vars": ["confounder_a", "confounder_b"],
  "method": "regression",
  "bootstrap_samples": 1000,
  "random_state": 42
}
```



## 39. 非参数回归分析 (Nonparametric Regression Analysis)

### 工具名称
perform_nonparametric_regression

### 功能描述
执行非参数/半参数回归分析，支持局部加权回归(LOESS)和样条回归两种方法

### 参数说明
- x_data: 自变量数据
- y_data: 因变量数据
- method: 非参数回归方法: 'loess'(局部加权回归), 'spline'(样条回归)
- loess_frac: LOESS方法中用于局部回归的窗口大小比例，值越大越平滑
- loess_it: LOESS方法的迭代次数
- spline_degree: 样条回归的多项式阶数
- spline_smooth_factor: 样条回归的平滑因子，值越大越平滑，None表示自动选择
- confidence_level: 置信区间水平，如0.95表示95%置信区间

### 测试数据 (LOESS方法)
```json
{
  "x_data": [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
  "y_data": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
  "method": "loess",
  "loess_frac": 0.3,
  "loess_it": 3,
  "spline_degree": 3,
  "spline_smooth_factor": null,
  "confidence_level": 0.95
}
```

### 测试数据 (样条回归方法)
```json
{
  "x_data": [1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5],
  "y_data": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
  "method": "spline",
  "loess_frac": 0.3,
  "loess_it": 3,
  "spline_degree": 3,
  "spline_smooth_factor": null,
  "confidence_level": 0.95
}
```

## 40. 分位数回归分析 (Quantile Regression)

### 工具名称
perform_quantile_regression

### 功能描述
执行分位数回归分析，用于研究自变量对因变量在不同分位数下的影响

### 参数说明
- dependent_var: 因变量数据
- independent_vars: 自变量数据，所有自变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
- var_names: 变量名称列表，第一个为因变量名，后续为自变量名
- quantiles: 分位数列表，每个值应在0到1之间，默认为[0.25, 0.5, 0.75]
- alpha: 正则化强度参数，默认为1.0
- standardize: 是否标准化数据，默认为True

### 测试数据
```json
{
  "dependent_var": [2.5, 3.6, 4.1, 5.2, 6.3, 7.8, 8.1, 9.4, 10.2, 11.5, 12.0, 13.2],
  "independent_vars": [
    1.2, 2.1, 2.8, 3.5, 4.2, 5.1, 5.8, 6.7, 7.3, 8.2, 8.9, 9.5,
    0.5, 1.0, 1.2, 1.8, 2.1, 2.7, 3.0, 3.5, 3.8, 4.2, 4.7, 5.0
  ],
  "var_names": ["house_price", "house_area", "location_score"],
  "quantiles": [0.25, 0.5, 0.75],
  "alpha": 1.0,
  "standardize": true
}
```

## 41. HDBSCAN聚类分析 (HDBSCAN Clustering)

### 工具名称
perform_hdbscan_clustering

### 功能描述
执行HDBSCAN聚类分析，一种基于密度的聚类算法，能够自动确定簇的数量并识别噪声点

### 参数说明
- data: 样本数据，所有变量的值按变量拼接成一维数组(列优先，先放完 X1、再放 X2 ...)
- var_names: 变量名称列表
- min_cluster_size: 形成簇所需的最小样本数，默认为5
- min_samples: 核心点邻域中的最小样本数，如果为None则默认等于min_cluster_size
- cluster_selection_method: 簇选择方法: 'eom'(超额质量算法), 'leaf'(叶簇选择)，默认为'eom'
- allow_single_cluster: 是否允许将所有点归为一个簇，默认为False
- alpha: 用于计算不稳定性的参数，默认为1.0
- metric: 距离度量方法，默认为"euclidean"
- standardize: 是否标准化数据，默认为True

### 测试数据
```json
{
  "data": [
    1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8, 8.0, 8.5, 7.8, 8.2, 8.7, 8.9, 7.5, 8.1, 4.0, 4.5, 3.8, 4.2, 4.7, 4.9, 3.5, 4.1,
    1.2, 2.2, 1.7, 2.7, 1.4, 2.4, 1.9, 2.9, 8.2, 8.7, 8.0, 8.4, 8.9, 9.1, 7.7, 8.3, 4.2, 4.7, 4.0, 4.4, 4.9, 5.1, 3.7, 4.3
  ],
  "var_names": ["feature_x", "feature_y"],
  "min_cluster_size": 3,
  "min_samples": null,
  "cluster_selection_method": "eom",
  "allow_single_cluster": false,
  "alpha": 1.0,
  "metric": "euclidean",
  "standardize": true
}
```

