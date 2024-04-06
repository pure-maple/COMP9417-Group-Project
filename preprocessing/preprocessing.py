import numpy as np

# 加载训练集数据
X_train = np.load('../data_set/X_train.npy')

# 计算每个特征的缺失值比例
missing_ratio = np.mean(np.isnan(X_train), axis=0)

# 定义高缺失率阈值为90%
high_missing_ratio_threshold = 0.9

# 标记应删除的特征
features_to_drop_high_missing = np.where(missing_ratio > high_missing_ratio_threshold)[0]

# 删除高缺失率的特征
X_dropped_high_missing = np.delete(X_train, features_to_drop_high_missing, axis=1)

# 保存删除高缺失率特征后的数据
np.save('../data_set/processed_data_set/X_dropped_90.npy', X_dropped_high_missing)
