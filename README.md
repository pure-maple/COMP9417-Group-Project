# COMP9417_GROUP_PROJECT

## 数据预处理

[预处理数据 Demo](./preprocessing/preprocessing.ipynb)

1. 去除高缺失率特征（暂定阈值为90%, 60%）
2. 去除一元特征
3. 对特征进行分类：离散型、连续浮点型、连续整型
    - 离散型：判断标准采用总观测值的5%作为阈值
4. 根据不同类型的特征进行缺失值处理
    - 离散型：使用众数进行填充（或者后续可以，根据数据分布情况进行插值处理）
    - 连续浮点型：使用平均值进行填充
    - 连续整型，使用四舍五入后的平均值进行填充
5. 对outlier的数据进行修改

处理数据文件说明：

1. `X_drop.npy`: 去除高缺失率（90%）特征和一元特征
2. `X_filled.npy`: 填充缺失值
3. `X_processed.npy`: 处理离群值，预处理完毕后保存的数据文件

## Feature Selection

[feature selection](./preprocessing/feature_selection.ipynb)

1. 对数据集进行分别处理，去除和目标变量无关的特征。

## Update Messages

- [2024/04/07 Maple]
    - 缺失率阈值按90%，60%进行分类处理，相关数据在`data_set`文件夹中的`drop_90`和`drop_60`中。
    - `processing.ipynb`和`feature_selection.ipynb`中相关函数已经封装。

后续建模可能用到的函数：

`classify_features`将数据集中的特征分类为离散型、连续性整型和连续性浮点型三种类型。

```python
import numpy as np


def classify_features(X_dataset, threshold_ratio=0.05):
    """
    Classify features of the dataset into discrete, continuous integer, and continuous float categories.

    Parameters:
    - X_dataset: The dataset to classify features for.
    - threshold_ratio: The threshold ratio to determine if a feature is discrete or continuous.

    Returns:
    - discrete_feature_indices: Indices of discrete features.
    - continuous_integer_indices: Indices of continuous integer features.
    - continuous_float_indices: Indices of continuous float features.
    """
    # Determine the unique counts for each feature
    unique_counts = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, X_dataset)

    # Determine feature types based on the proportion of unique values

    discrete_feature_indices = np.where((unique_counts / X_dataset.shape[0]) < threshold_ratio)[0]
    continuous_feature_indices = np.where((unique_counts / X_dataset.shape[0]) >= threshold_ratio)[0]

    # Initialize lists to save indices of continuous integer and float features
    continuous_integer_indices = []
    continuous_float_indices = []

    # Classify continuous features into integer and float
    for index in continuous_feature_indices:
        if np.all(np.mod(X_dataset[:, index][~np.isnan(X_dataset[:, index])], 1) == 0):
            continuous_integer_indices.append(index)
        else:
            continuous_float_indices.append(index)

    return discrete_feature_indices, continuous_integer_indices, continuous_float_indices


# Example usage
X_drop = np.load('../data_set/drop_90/X_drop.npy')

# Classify features of the dataset
discrete_indices, continuous_int_indices, continuous_float_indices = classify_features(X_drop)

print("Discrete feature indices:", discrete_indices)
print("Continuous integer feature indices:", continuous_int_indices)
print("Continuous float feature indices:", continuous_float_indices)

```

