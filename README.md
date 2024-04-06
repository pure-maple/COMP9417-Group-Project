# COMP9417_GROUP_PROJECT

## 数据预处理

[预处理数据 Demo](./preprocessing/preprocessing_0406.ipynb)

1. 去除高缺失率特征（暂定阈值为90%）
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