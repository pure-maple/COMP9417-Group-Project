# COMP9417_GROUP_PROJECT

## 数据预处理

[预处理数据 Demo](./preprocessing/preprocessing.ipynb)

### 使用文件

- `data_set/preprocessed/01_remove/Y_remove.npy`: 移除高缺失率样本数据后的Y数据集
- `data_set/preprocessed/03_outliers`: 处理后的X数据集
    - 移除高缺失率的特征及样本
    - 移除一元特征
    - 调整outlier数据
- `data_set/preprocessed/04_processed_features`: 针对每个label进行特征相关性分析，移除了无关特征

### 工具函数

#### 特征分类
```python
def classify_features(X, threshold_ratio=0.05):
  """
  将数据集的特征分为离散型、连续型、连续整型和连续浮点型。

  Parameters:
  - X: 要对特征进行分类的数据集。
  - threshold_ratio: 用于确定特征是离散还是连续的阈值比率。

  Returns:
  - discrete_feature_indices: 离散特征的索引。
  - continuous_feature_indices: 连续特征的索引。
  - binary_feature_indices: 二元特征的索引。
  - multi_feature_indices: 多元特征的索引。
  - continuous_integer_indices: 连续整型特征的索引。
  - continuous_float_indices: 连续浮点型特征的索引。
  """
  # 计算每个特征的唯一值数量
  unique_counts = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, X)

  # 根据阈值比率划分离散和连续特征
  discrete_feature_indices = np.where((unique_counts / X.shape[0]) < threshold_ratio)[0]
  continuous_feature_indices = np.where((unique_counts / X.shape[0]) >= threshold_ratio)[0]

  # 初始化列表，以保存连续整数和浮点特征的索引
  continuous_integer_indices = []
  continuous_float_indices = []

  # 将连续特征分为整数和浮点两种
  for index in continuous_feature_indices:
    if np.all(np.mod(X[:, index][~np.isnan(X[:, index])], 1) == 0):
      continuous_integer_indices.append(index)
    else:
      continuous_float_indices.append(index)

  # 初始化列表，以保存二元和多元特征的索引
  binary_feature_indices = []
  multi_feature_indices = []

  # 将离散特征分为二元和多元两种
  for index in discrete_feature_indices:
    if len(np.unique(X[:, index][~np.isnan(X[:, index])])) == 2:
      binary_feature_indices.append(index)
    else:
      multi_feature_indices.append(index)

  return discrete_feature_indices, continuous_feature_indices, binary_feature_indices, multi_feature_indices, continuous_integer_indices, continuous_float_indices


def print_feature_classification(X):
  """
  输出特征的分类情况。

  Parameters:
  - X: 要输出特征分类的数据集。
  """
  discrete_feature_indices, continuous_feature_indices, binary_feature_indices, multi_feature_indices, continuous_integer_indices, continuous_float_indices = classify_features(
    X)

  print("离散特征数量: {}".format(len(discrete_feature_indices)))
  print("连续特征数量: {}".format(len(continuous_feature_indices)))
  print("二元特征数量: {}".format(len(binary_feature_indices)))
  print("多元特征数量: {}".format(len(multi_feature_indices)))
  print("连续整型特征数量: {}".format(len(continuous_integer_indices)))
  print("连续浮点型特征数量: {}".format(len(continuous_float_indices)))

  print("\n离散特征索引: {}".format(discrete_feature_indices))
  print("连续特征索引: {}".format(continuous_feature_indices))
  print("二元特征索引: {}".format(binary_feature_indices))
  print("多元特征索引: {}".format(multi_feature_indices))
  print("连续整型特征索引: {}".format(continuous_integer_indices))
  print("连续浮点型特征索引: {}".format(continuous_float_indices))
```

