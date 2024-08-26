import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.metrics import Precision, Recall
from sklearn.metrics import precision_score, recall_score
import os


class ModelConfig:
    def __init__(self, batch_size, epochs, dropout_rate, learning_rate, n_splits):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_splits = n_splits

def nn_model(feature_dim, label_dim, config):
    print(f"X:{feature_dim}   Y:{label_dim}")
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(label_dim, activation='sigmoid'))
    adam_optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    return model

class F1ScoreCallback(Callback):
    def __init__(self, test_data, result_storage):
        self.test_data = test_data
        self.best_f1 = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_epoch = 0
        self.result_storage = result_storage

    def on_epoch_end(self, epoch, logs=None):
        X_test, Y_test = self.test_data
        Y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        current_f1 = f1_score(Y_test, Y_pred, average='micro')
        current_precision = precision_score(Y_test, Y_pred, average='micro', zero_division=0)
        current_recall = recall_score(Y_test, Y_pred, average='micro', zero_division=0)

        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_precision = current_precision
            self.best_recall = current_recall
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        self.result_storage['best_f1'] = self.best_f1
        self.result_storage['best_precision'] = self.best_precision
        self.result_storage['best_recall'] = self.best_recall
        self.result_storage['best_epoch'] = self.best_epoch


def train_model(X_train, X_test, Y_train, Y_test, config, data_name, cv_round, result_storage):
    print(f"Training {data_name}, CV Round: {cv_round}")
    model = nn_model(X_train.shape[1], Y_train.shape[1], config)
    # model = nn_model(X_train.shape[1], Y_train.shape[1], config)

    f1_callback = F1ScoreCallback((X_test, Y_test), result_storage)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, Y_test), callbacks=[f1_callback, early_stopping])
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # 加载预测数据
    X_test_knn = np.load('./data_preprocessed/X_test_filled_knn.npy')
    # 进行预测
    predictions = model.predict(X_test_knn)
    # 应用阈值将概率转换为0或1
    predictions = (predictions > 0.5).astype(int)   
    
    # 保存预测结果
    results_dir = './prediction_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    np.save(os.path.join(results_dir, f'predictions_{data_name}_cv_{cv_round}.npy'), predictions)

    return model

def cross_validate_model(X, Y, config, data_name):
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    results = []
    for i, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        result_storage = {
            'config': config,
            'data_name': data_name,
            'cv_round': i
        }
        train_model(X_train, X_test, Y_train, Y_test, config, data_name, i, result_storage)
        results.append(result_storage)
        print(f"Best F1 Score for {data_name}, CV Round {i}: {result_storage['best_f1']}")
    return results

def train_total():
    Y = np.load('./data_preprocessed/Y_remove.npy')
    datasets = {
        'X_outliers_regression_reverse': np.load('./data_preprocessed/03_outliers/X_outliers_regression_reverse.npy')
    }

    # datasets = {
    #     'X_outliers_knn': np.load('./data_preprocessed/03_outliers/X_outliers_knn.npy'),
    #     'X_outliers_knn': np.load('./data_preprocessed/03_outliers/X_outliers_knn.npy'),
    #     'X_outliers_naive': np.load('./data_preprocessed/03_outliers/X_outliers_naive.npy'),
    #     'X_outliers_regression_naive': np.load('./data_preprocessed/03_outliers/X_outliers_regression_naive.npy'),
    #     'X_outliers_regression_reverse': np.load('./data_preprocessed/03_outliers/X_outliers_regression_reverse.npy'),
    #     'X_outliers_regression_rf': np.load('./data_preprocessed/03_outliers/X_outliers_regression_rf.npy'),
    #     'X_outliers_regression_svr': np.load('./data_preprocessed/03_outliers/X_outliers_regression_svr.npy')
    # }

    # batch_sizes = [8, 16, 32, 64]
    # learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    # dropout_rates = [0, 0.05, 0.1, 0.2, 0.3]
    # epochs = 100
    # n_splits = 5

    # batch_sizes = [16, 32]
    # learning_rates = [0.00001]
    # dropout_rates = [0.3]
    # epochs = 100
    # n_splits = 5

    # batch_sizes = [16]
    # learning_rates = [0.000001]
    # dropout_rates = [0.3]
    # epochs = 100
    # n_splits = 5

    # batch_sizes = [16]
    # learning_rates = [0.0000005, 0.0000004, 0.0000003, 0.0000002, 0.0000001, 0.00000005, 0.00000001]
    # dropout_rates = [0.4, 0.5]
    # epochs = 300
    # n_splits = 5

    batch_sizes = [16]
    learning_rates = [0.000001]
    dropout_rates = [0.3]
    epochs = 100
    n_splits = 5

    all_results = []
    
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for dropout_rate in dropout_rates:
                config = ModelConfig(batch_size, epochs, dropout_rate, learning_rate, n_splits)
                for name, X in datasets.items():
                    results = cross_validate_model(X, Y, config, name)
                    all_results.extend(results)

    return all_results

# 运行训练
all_results = train_total()