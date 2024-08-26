import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

# Load shared feature set and labels for all tasks
X_all = np.load('X_outliers_regression_naive.npy')
y_all = np.load('Y_remove.npy')
n_tasks = y_all.shape[1]  # Assume each column in y_all represents a label for a task

# Choose sampling method: 'oversample' or 'undersample'
sampling_method = 'undersample'


# Noise injection and feature perturbation function
def add_noise_and_perturbation(data, noise_level=0.01, perturbation_level=0.05):
    noise = np.random.normal(0, np.std(data, axis=0) * noise_level, data.shape)
    perturbation = np.random.uniform(-1, 1, data.shape) * np.std(data, axis=0) * perturbation_level
    return data + noise + perturbation


# Define the parameter grid to be optimized
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [5, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': ['balanced', 'balanced_subsample']  # Add class weight to address imbalance
}
total_accuracy = 0  # Initialize cumulative accuracy variable
total_f1 = 0
total_crossentropy = 0
count_binary = 0  # Used to count the number of binary classification tasks

for task_id in range(n_tasks):
    y_task = y_all[:, task_id]  # Get current task labels from y_all

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_task, test_size=0.3, random_state=42)
    if sampling_method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    else:
        sampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    # Create a RandomForest classifier
    classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Train model
    grid_search.fit(X_resampled, y_resampled)

    # Feature importance analysis and feature selection
    selector = SelectFromModel(grid_search.best_estimator_, threshold="median")
    X_important_train = selector.fit_transform(X_resampled, y_resampled)
    X_important_test = selector.transform(X_test)

    # Retrain model with selected important features
    grid_search.best_estimator_.fit(X_important_train, y_resampled)

    # Predict with the best parameter model
    y_pred = grid_search.best_estimator_.predict(X_important_test)

    # Calculate accuracy and accumulate
    accuracy = accuracy_score(y_test, y_pred)
    total_accuracy += accuracy

    if len(np.unique(y_task)) == 2:  # Binary classification tasks
        count_binary += 1
        y_pred_proba = grid_search.best_estimator_.predict_proba(X_important_test)[:, 1]
        binary_crossentropy = log_loss(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        total_crossentropy += binary_crossentropy
        total_f1 += f1
        print(f"Task {task_id}: Accuracy = {accuracy}, Binary Cross-Entropy = {binary_crossentropy}, F1 Score = {f1}")
    else:
        # Multiclass tasks
        metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        total_f1 += metrics[2]
        print(
            f"Task {task_id}: Accuracy = {accuracy}, Precision = {metrics[0]}, Recall = {metrics[1]}, F1 Score = {metrics[2]}")

# Output averages
average_accuracy = total_accuracy / n_tasks
print(f"Average Accuracy across all tasks: {average_accuracy:.4f}")
print(f"Average F1 Score = {total_f1 / n_tasks}")
print(
    f"Average Binary Cross-Entropy (Binary tasks) = {total_crossentropy / count_binary if count_binary > 0 else 'N/A'}")
