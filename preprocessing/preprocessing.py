import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练集数据
X_train = np.load('../data_set/X_train.npy')

# 计算每个特征的缺失值比例
missing_ratio = np.mean(np.isnan(X_train), axis=0)

# 定义高缺失率阈值为90%
high_missing_ratio_threshold = 0.9

# 标记应删除的特征
features_to_drop_high_missing = np.where(missing_ratio > high_missing_ratio_threshold)[0]

# 计算每个特征的唯一值数量
unique_counts = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, X_train)

# 一元特征的索引
single_value_feature_indices = np.where(unique_counts == 1)[0]

# 删除高缺失率特征和一元特征
X_dropped_high_missing = np.delete(X_train,
                                   np.concatenate([features_to_drop_high_missing, single_value_feature_indices]),
                                   axis=1)

# 加载删除高缺失率特征后的数据
X_train = X_dropped_high_missing

# 确定每个特征的唯一值数量
unique_counts = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, X_dropped_high_missing)

# 使用唯一值数量与样本总数的比例来推断特征类型
# 设定一个阈值
# 如果唯一值占总样本数的比例小于5%是离散型特征，否则是连续型特征
threshold_ratio = 0.05
discrete_feature_indices = np.where((unique_counts / X_train.shape[0]) < threshold_ratio)[0]
continuous_feature_indices = np.where((unique_counts / X_train.shape[0]) >= threshold_ratio)[0]

# 初始化列表来保存整数型和浮点型连续特征的索引
continuous_integer_indices = []
continuous_float_indices = []

# 遍历所有连续型特征，检查它们是整数型还是浮点型
for index in continuous_feature_indices:
    if np.all(np.mod(X_train[:, index][~np.isnan(X_train[:, index])], 1) == 0):
        continuous_integer_indices.append(index)
    else:
        continuous_float_indices.append(index)

# 对离散型特征使用众数填充
imputer_mode = SimpleImputer(strategy='most_frequent')
X_train[:, discrete_feature_indices] = imputer_mode.fit_transform(X_train[:, discrete_feature_indices])

# 对连续浮点型特征使用均值填充
imputer_mean = SimpleImputer(strategy='mean')
X_train[:, continuous_float_indices] = imputer_mean.fit_transform(X_train[:, continuous_float_indices])

# 对连续整数型特征使用均值填充并四舍五入
X_continuous_integer = imputer_mean.fit_transform(X_train[:, continuous_integer_indices])
X_train[:, continuous_integer_indices] = np.round(X_continuous_integer)

# 加载填充后的数据
X_cleaned = X_train


# 定义处理离群值的函数
def cap_outliers(data, lower_bound, upper_bound):
    data = np.where(data < lower_bound, lower_bound, data)
    data = np.where(data > upper_bound, upper_bound, data)
    return data


# 遍历每个连续特征
for feature_index in continuous_feature_indices:  # 使用之前确定的连续特征索引列表
    feature_data = X_cleaned[:, feature_index]
    q1, q3 = np.percentile(feature_data[~np.isnan(feature_data)], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 检查是否有超出上下界的离群值
    outlier_indices = np.where((feature_data < lower_bound) | (feature_data > upper_bound))[0]
    if len(outlier_indices) > 0:
        # 可视化特征的箱线图（处理前）
        sns.boxplot(x=feature_data)
        plt.title('Box Plot Before Outlier Capping - Feature Index {}'.format(feature_index))
        plt.show()

        # 处理离群值
        X_cleaned[:, feature_index] = cap_outliers(feature_data, lower_bound, upper_bound)

        # 可视化特征的箱线图（处理后）
        sns.boxplot(x=X_cleaned[:, feature_index])
        plt.title('Box Plot After Outlier Capping - Feature Index {}'.format(feature_index))
        plt.show()
