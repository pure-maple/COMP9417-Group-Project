import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Load the data
X_train = np.load('../data_set/X_train.npy')
Y_train = np.load('../data_set/Y_train.npy')


def missing_ratio(X):
    """
    Calculate the missing ratio of each feature and sample
    :param X: X numpy array
    :return: feature_hist, sample_hist
    """
    # Calculate the missing ratio of each feature and sample
    feature_missing_ratio = np.mean(np.isnan(X), axis=0)
    sample_missing_ratio = np.mean(np.isnan(X), axis=1)

    # Calculate the histogram of missing ratio
    bins = np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.2, ..., 1.0]

    # Calculate the histogram of missing ratio
    feature_hist, _ = np.histogram(feature_missing_ratio, bins=bins)
    sample_hist, _ = np.histogram(sample_missing_ratio, bins=bins)

    return feature_hist, sample_hist


# Print the shape of the data
print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))


def print_missing_ratio(X):
    """
    Print the missing ratio of each feature and sample
    :param X: X numpy array
    :return: None
    """
    feature_hist, sample_hist = missing_ratio(X)

    print("Feature missing ratio:")
    for i in range(len(feature_hist)):
        print("[{:.0%} - {:.0%}]: {}".format(i * 0.1, (i + 1) * 0.1, feature_hist[i]))

    print("\nSample missing ratio:")
    for i in range(len(sample_hist)):
        print("[{:.0%} - {:.0%}]: {}".format(i * 0.1, (i + 1) * 0.1, sample_hist[i]))


# Print the missing ratio of the data
print_missing_ratio(X_train)


# Remove the features and samples with high missing ratio
def remove_high_missing(X, Y, feature_threshold, sample_threshold):
    """
    Remove the features and samples with high missing ratio
    :param X: X numpy array
    :param Y: Y numpy array
    :param feature_threshold: feature missing ratio threshold
    :param sample_threshold: sample missing ratio threshold
    :return: X, Y
    """
    # Calculate the missing ratio of each feature and sample
    feature_missing_ratio = np.mean(np.isnan(X), axis=0)
    sample_missing_ratio = np.mean(np.isnan(X), axis=1)

    # Remove the features and samples with high missing ratio
    feature_idx = np.where(feature_missing_ratio > feature_threshold)[0]
    X = np.delete(X, feature_idx, axis=1)

    # Remove the samples with high missing ratio
    sample_idx = np.where(sample_missing_ratio > sample_threshold)[0]
    X = np.delete(X, sample_idx, axis=0)
    Y = np.delete(Y, sample_idx, axis=0)

    return X, Y


# Remove the features and samples with high missing ratio
X_remove_high_missing, Y_remove_high_missing = remove_high_missing(X_train, Y_train, 0.6, 0.2)

# Print the shape of the data after removing the features and samples with high missing ratio
print("X_remove_high_missing shape: {}".format(X_remove_high_missing.shape))
print("Y_remove_high_missing shape: {}".format(Y_remove_high_missing.shape))

# Verify whether the processed data still contains missing values
print_missing_ratio(X_remove_high_missing)

# Save the processed data
np.save('../data_set/preprocessed/01_remove/X_remove_high_missing.npy', X_remove_high_missing)
np.save('../data_set/preprocessed/01_remove/Y_remove_high_missing.npy', Y_remove_high_missing)


def remove_unary_features(X):
    """
    Remove the unary features
    :param X: X numpy array
    :return: X
    """
    # Initialize the list of unary feature indices
    unary_feature_indices = []

    # Iterate over each feature
    for i in range(X.shape[1]):
        # Get the unique values of the feature
        unique_values = np.unique(X[:, i])

        # If the feature has only one unique value, add the index to the list
        if len(unique_values) <= 1:
            unary_feature_indices.append(i)

    # Remove the unary features
    X = np.delete(X, unary_feature_indices, axis=1)

    return X


# Remove the unary features
X_remove_unary = remove_unary_features(X_remove_high_missing)

# Print the shape of the data after removing the unary features
print("X_remove_unary shape: {}".format(X_remove_unary.shape))

# Save the processed data
np.save('../data_set/preprocessed/01_remove/X_remove.npy', X_remove_unary)
np.save('../data_set/preprocessed/01_remove/Y_remove.npy', Y_remove_high_missing)


def classify_features(X, threshold_ratio=0.05):
    """
    Classify the features into discrete, continuous, binary, multi, continuous integer, and continuous float.

    Parameters:
    - X: The dataset to classify the features.
    - threshold_ratio: The threshold ratio to determine whether a feature is discrete or continuous.

    Returns:
    - discrete_feature_indices: List of discrete feature indices.
    - continuous_feature_indices: List of continuous feature indices.
    - binary_feature_indices: List of binary feature indices.
    - multi_feature_indices: List of multi-feature indices.
    - continuous_integer_indices: List of continuous integer feature indices.
    - continuous_float_indices: List of continuous float feature indices.
    """
    # Calculate the number of unique values for each feature
    unique_counts = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, X)

    # Classify the features into discrete and continuous
    discrete_feature_indices = np.where((unique_counts / X.shape[0]) < threshold_ratio)[0]
    continuous_feature_indices = np.where((unique_counts / X.shape[0]) >= threshold_ratio)[0]

    # Initialize lists to save the indices of binary and multi features
    continuous_integer_indices = []
    continuous_float_indices = []

    # Classify the continuous features into integer and float
    for index in continuous_feature_indices:
        if np.all(np.mod(X[:, index][~np.isnan(X[:, index])], 1) == 0):
            continuous_integer_indices.append(index)
        else:
            continuous_float_indices.append(index)

    # Initialize lists to save the indices of binary and multi features
    binary_feature_indices = []
    multi_feature_indices = []

    # Classify the discrete features into binary and multi
    for index in discrete_feature_indices:
        if len(np.unique(X[:, index][~np.isnan(X[:, index])])) == 2:
            binary_feature_indices.append(index)
        else:
            multi_feature_indices.append(index)

    return discrete_feature_indices, continuous_feature_indices, binary_feature_indices, multi_feature_indices, continuous_integer_indices, continuous_float_indices


def print_feature_classification(X):
    """
    Print the classification of features.

    Parameters:
    - X: X numpy array
    """
    discrete_feature_indices, continuous_feature_indices, binary_feature_indices, multi_feature_indices, continuous_integer_indices, continuous_float_indices = classify_features(
        X)

    print("Discrete features: {}".format(len(discrete_feature_indices)))
    print("Continuous features: {}".format(len(continuous_feature_indices)))
    print("Binary features: {}".format(len(binary_feature_indices)))
    print("Multi features: {}".format(len(multi_feature_indices)))
    print("Continuous integer features: {}".format(len(continuous_integer_indices)))
    print("Continuous float features: {}".format(len(continuous_float_indices)))

    print("\nDiscrete feature indices: {}".format(discrete_feature_indices))
    print("Continuous feature indices: {}".format(continuous_feature_indices))
    print("Binary feature indices: {}".format(binary_feature_indices))
    print("Multi feature indices: {}".format(multi_feature_indices))
    print("Continuous integer feature indices: {}".format(continuous_integer_indices))
    print("Continuous float feature indices: {}".format(continuous_float_indices))


# Print the classification of features
print_feature_classification(X_remove_unary)


def naive_fill(X, discrete_indices, continuous_int_indices, continuous_float_indices):
    """
    Fill the missing values with the naive method.
    :param X: X numpy array
    :param discrete_indices: list of discrete feature indices
    :param continuous_int_indices: list of continuous integer feature indices
    :param continuous_float_indices: list of continuous float feature indices
    :return:
    """
    imputer_mode = SimpleImputer(strategy='most_frequent')
    imputer_median = SimpleImputer(strategy='median')
    imputer_mean = SimpleImputer(strategy='mean')

    if len(discrete_indices) > 0:  # Check if there are discrete indices
        X[:, discrete_indices] = imputer_mode.fit_transform(X[:, discrete_indices])
    if len(continuous_int_indices) > 0:  # Check if there are continuous integer indices
        X[:, continuous_int_indices] = imputer_median.fit_transform(X[:, continuous_int_indices])
    if len(continuous_float_indices) > 0:  # Check if there are continuous float indices
        X[:, continuous_float_indices] = imputer_mean.fit_transform(X[:, continuous_float_indices])

    return X


# Classify the features
discrete_feature_indices, continuous_feature_indices, binary_feature_indices, multi_feature_indices, continuous_integer_indices, continuous_float_indices = classify_features(
    X_remove_unary)

# Fill the missing values with the naive method
X_filled_naive = naive_fill(X_remove_unary, discrete_feature_indices, continuous_integer_indices,
                            continuous_float_indices)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_naive)

# Save the processed data
np.save('../data_set/preprocessed/02_fill/X_filled_naive.npy', X_filled_naive)

from sklearn.impute import KNNImputer


def knn_fill(X, n_neighbors=5):
    """
    Fills missing values using K-Nearest Neighbors.

    Parameters:
    - X: The dataset with missing values.
    - n_neighbors: Number of neighboring samples to use for imputation.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(X)


# Fill the missing values with the KNN method
X_filled_knn = knn_fill(X_remove_unary)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_knn)

# Save the processed data
np.save('../data_set/preprocessed/02_filled/X_filled_knn.npy', X_filled_knn)


def regression_fill_naive(X):
    """
    Fills missing values using regression imputation with a Linear Regression model.
    """
    imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0)
    return imputer.fit_transform(X)


# Fill the missing values with the regression method
X_filled_regression_naive = regression_fill_naive(X_remove_unary)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_regression_naive)

# Save the processed data
np.save('../data_set/preprocessed/02_filled/X_filled_regression_naive.npy', X_filled_regression_naive)

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def reverse_regression_fill(X, Y):
    """
    Fills missing values in X using the target variable y with Linear Regression.

    Parameters:
    - X: Feature matrix with missing values (numpy array).
    - Y: Target variable (numpy array).

    Returns:
    - X_filled: Feature matrix with missing values filled (numpy array).
    """
    # Reshape Y to 2D array
    Y = Y.reshape(-1, 1)

    # Copy X to X_filled
    X_filled = X.copy()

    # Iterate over each feature
    for i in range(X.shape[1]):
        missing_indices = np.where(np.isnan(X[:, i]))[0]
        if missing_indices.size > 0:
            # Initialize the Linear Regression model
            model = LinearRegression()

            # Get the indices of non-missing values
            not_missing_indices = np.where(~np.isnan(X[:, i]))[0]

            # If all values are missing, fill with the mean
            if len(not_missing_indices) == 0:
                imputer = SimpleImputer(strategy='mean')
                X_filled[:, i] = imputer.fit_transform(X[:, i].reshape(-1, 1)).ravel()
                continue

            # Prepare the input data for the model
            inputs = np.hstack((Y[not_missing_indices], X[not_missing_indices, :]))

            # Fit the model
            model.fit(inputs, X[not_missing_indices, i])

            # Predict the missing values
            inputs_missing = np.hstack((Y[missing_indices], X[missing_indices, :]))
            X_filled[missing_indices, i] = model.predict(inputs_missing)

    return X_filled


# Fill the missing values with the reverse regression method
X_filled_regression_reverse = reverse_regression_fill(X_remove_unary, Y_remove_high_missing)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_regression_reverse)

# Save the processed data
np.save('../data_set/preprocessed/02_filled/X_filled_regression_reverse.npy', X_filled_regression_reverse)


def regression_fill_rf(X, Y):
    """
    Fills missing values in X using random forest regression based on target Y.
    """
    # Combine Y and X for imputation, assuming Y could help predict X
    data = np.hstack((Y, X))

    # Create the imputer with a RandomForestRegressor that supports multi-output
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
                               max_iter=10, random_state=0, initial_strategy='mean')

    # Perform the imputation
    data_filled = imputer.fit_transform(data)

    # Extract the filled X
    X_filled = data_filled[:, Y.shape[1]:]  # Skip the Y columns

    return X_filled


# Fill the missing values with the regression method
X_filled_regression_rf = regression_fill_rf(X_remove_unary, Y_remove_high_missing)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_regression_rf)

# Save the processed data
np.save('../data_set/preprocessed/02_filled/X_filled_regression_rf.npy', X_filled_regression_rf)


def fill_features_svr(X, Y):
    """
    Fills missing values in each feature of X using the target y with SVR.

    Parameters:
    - X: Feature matrix with missing values (2D numpy array).
    - Y: Target variable (1D numpy array).

    Returns:
    - X_filled: Feature matrix with missing values filled (numpy array).
    """
    # Ensure y is the correct shape
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    # Initialize the filled data array
    X_filled = X.copy()

    # Iterate over each feature in X
    for i in range(X.shape[1]):
        # Identify missing values for this feature
        missing_indices = np.where(np.isnan(X[:, i]))[0]
        if missing_indices.size > 0:
            # Create a SVR model with a pipeline that includes scaling
            model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

            # Fit the model using y as input and the available values of the current feature
            not_missing_indices = np.where(~np.isnan(X[:, i]))[0]
            model.fit(Y[not_missing_indices], X[not_missing_indices, i])

            # Predict missing values
            X_filled[missing_indices, i] = model.predict(Y[missing_indices])

    return X_filled


# Fill the missing values with the regression method
X_filled_regression_svr = fill_features_svr(X_remove_unary, Y_remove_high_missing)

# Print the missing ratio of the data after filling the missing values
print_missing_ratio(X_filled_regression_svr)

# Save the processed data
np.save('../data_set/preprocessed/02_filled/X_filled_regression_svr.npy', X_filled_regression_svr)

# Print the shape of the data after filling the missing values
print("X_filled_naive shape: {}".format(X_filled_naive.shape))
print("X_filled_knn shape: {}".format(X_filled_knn.shape))
print("X_filled_regression_naive shape: {}".format(X_filled_regression_naive.shape))
print("X_filled_regression_reverse shape: {}".format(X_filled_regression_reverse.shape))
print("X_filled_regression_rf shape: {}".format(X_filled_regression_rf.shape))
print("X_filled_regression_svr shape: {}".format(X_filled_regression_svr.shape))


def process_outliers(X):
    """
    Processes outliers for specified continuous features and reports details.

    Parameters:
    - X: The dataset to process outliers for.

    Returns:
    - X_processed: The dataset with outliers processed.
    - reports: A dictionary containing reports on processed features.
    """

    X = X.copy()

    _, continuous_feature_indices, _, _, _, _ = classify_features(X)

    # Initialize the reports dictionary
    reports = {}

    for feature_index in continuous_feature_indices:
        feature_data = X[:, feature_index]
        q1, q3 = np.percentile(feature_data[~np.isnan(feature_data)], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Detect outliers
        outlier_indices = np.where((feature_data < lower_bound) | (feature_data > upper_bound))[0]
        outlier_count = len(outlier_indices)

        # Visualize the box plot before capping
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=feature_data)
        plt.title(f'Before Capping - Feature {feature_index}')

        # Cap the outliers
        feature_data_capped = np.where(feature_data < lower_bound, lower_bound,
                                       np.where(feature_data > upper_bound, upper_bound, feature_data))
        X[:, feature_index] = feature_data_capped

        # Visualize the box plot after capping
        plt.subplot(1, 2, 2)
        sns.boxplot(x=feature_data_capped)
        plt.title(f'After Capping - Feature {feature_index}')
        plt.show()

        # Update reports
        reports[feature_index] = {
            'outlier_count': outlier_count,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    return X, reports


# Process outliers for naive filled data
print("Processing outliers for naive filled data:")
X_processed_outliers_naive, reports_naive = process_outliers(X_filled_naive)

# Process outliers for KNN filled data
print("\nProcessing outliers for KNN filled data:")
X_processed_outliers_knn, reports_knn = process_outliers(X_filled_knn)

# Process outliers for regression filled data
print("\nProcessing outliers for regression filled data:")
X_processed_outliers_regression_naive, reports_regression_naive = process_outliers(X_filled_regression_naive)

# Process outliers for reverse regression filled data
print("\nProcessing outliers for reverse regression filled data:")
X_processed_outliers_regression_reverse, reports_regression_reverse = process_outliers(X_filled_regression_reverse)

# Process outliers for random forest regression filled data
print("\nProcessing outliers for random forest regression filled data:")
X_processed_outliers_regression_rf, reports_regression_rf = process_outliers(X_filled_regression_rf)

# Process outliers for SVR regression filled data
print("\nProcessing outliers for SVR regression filled data:")
X_processed_outliers_regression_svr, reports_regression_svr = process_outliers(X_filled_regression_svr)

# Print the shape of the data after processing outliers
print("Naive filled data outliers processing reports:")
for feature_index, report in reports_naive.items():
    print(f"Feature {feature_index}: {report}")

print("\nKNN filled data outliers processing reports:")
for feature_index, report in reports_knn.items():
    print(f"Feature {feature_index}: {report}")

print("\nRegression filled data outliers processing reports:")
for feature_index, report in reports_regression_naive.items():
    print(f"Feature {feature_index}: {report}")

print("\nReverse regression filled data outliers processing reports:")
for feature_index, report in reports_regression_reverse.items():
    print(f"Feature {feature_index}: {report}")

print("\nRandom forest regression filled data outliers processing reports:")
for feature_index, report in reports_regression_rf.items():
    print(f"Feature {feature_index}: {report}")

# Print the shape of the data after processing outliers
print("X_processed_outliers_naive shape: {}".format(X_processed_outliers_naive.shape))
print("X_processed_outliers_knn shape: {}".format(X_processed_outliers_knn.shape))
print("X_processed_outliers_regression_naive shape: {}".format(X_processed_outliers_regression_naive.shape))
print("X_processed_outliers_regression_reverse shape: {}".format(X_processed_outliers_regression_reverse.shape))
print("X_processed_outliers_regression_rf shape: {}".format(X_processed_outliers_regression_rf.shape))
print("X_processed_outliers_regression_svr shape: {}".format(X_processed_outliers_regression_svr.shape))

# Save the processed data
np.save('../data_set/preprocessed/03_outliers/X_outliers_naive.npy', X_processed_outliers_naive)
np.save('../data_set/preprocessed/03_outliers/X_outliers_knn.npy', X_processed_outliers_knn)
np.save('../data_set/preprocessed/03_outliers/X_outliers_regression_naive.npy', X_processed_outliers_regression_naive)
np.save('../data_set/preprocessed/03_outliers/X_outliers_regression_reverse.npy',
        X_processed_outliers_regression_reverse)
np.save('../data_set/preprocessed/03_outliers/X_outliers_regression_rf.npy', X_processed_outliers_regression_rf)
np.save('../data_set/preprocessed/03_outliers/X_outliers_regression_svr.npy', X_processed_outliers_regression_svr)


def process_features_for_label(x_data, y_data, label_index):
    """
    Analyzes and processes features based on the given label index.

    Parameters:
    - x_data: feature data set
    - y_data: label data set
    - label_index: label index to process

    Returns:
    - relevant_features_indices: indices of features relevant to the label
    - relevant_scores: mutual information scores of relevant features
    - irrelevant_features_indices: indices of features irrelevant to the label
    - x_data_relevant: feature data set with irrelevant features removed
    """
    # Calculate the mutual information scores between features and the label
    scores = mutual_info_classif(x_data, y_data[:, label_index])

    # Identify relevant and irrelevant features
    relevant_features_info = [(i, score) for i, score in enumerate(scores) if score > 0]
    irrelevant_features_indices = [i for i, score in enumerate(scores) if score == 0]

    # Print relevant feature information
    print(f"Label {label_index + 1}:")
    print("Relevant features and their scores:")
    for i, score in relevant_features_info:
        print(f"Feature {i + 1}: {score}")
    print(f"Total number of relevant features: {len(relevant_features_info)}")

    # Print irrelevant feature information
    print("Irrelevant features indices:")
    print([i + 1 for i in irrelevant_features_indices])
    print(f"Total number of irrelevant features: {len(irrelevant_features_indices)}")
    print("\n")

    # Remove irrelevant features
    relevant_features_indices = [i for i, _ in relevant_features_info]
    x_data_processed = x_data[:, relevant_features_indices]

    return x_data_processed


# Load the data
y_data = np.load('../data_set/preprocessed/01_remove/y_remove.npy')

# Load the processed data
x_naive = np.load('../data_set/preprocessed/03_outliers/X_outliers_naive.npy')
x_knn = np.load('../data_set/preprocessed/03_outliers/X_outliers_knn.npy')
x_regression_naive = np.load('../data_set/preprocessed/03_outliers/X_outliers_regression_naive.npy')
x_regression_reverse = np.load('../data_set/preprocessed/03_outliers/X_outliers_regression_reverse.npy')
x_regression_rf = np.load('../data_set/preprocessed/03_outliers/X_outliers_regression_rf.npy')
x_regression_svr = np.load('../data_set/preprocessed/03_outliers/X_outliers_regression_svr.npy')

# Iterate over all labels and save the processed data
for label_index in range(y_data.shape[1]):
    x_data_relevant_naive = process_features_for_label(x_naive, y_data, label_index)
    x_data_relevant_knn = process_features_for_label(x_knn, y_data, label_index)
    x_data_relevant_regression_naive = process_features_for_label(x_regression_naive, y_data, label_index)
    x_data_relevant_regression_reverse = process_features_for_label(x_regression_reverse, y_data, label_index)
    x_data_relevant_regression_rf = process_features_for_label(x_regression_rf, y_data, label_index)
    x_data_relevant_regression_svr = process_features_for_label(x_regression_svr, y_data, label_index)

    # Save the processed data with a unique file name for each label
    np.save(f'../data_set/preprocessed/04_processed_features/naive/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_naive)
    np.save(f'../data_set/preprocessed/04_processed_features/knn/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_knn)
    np.save(f'../data_set/preprocessed/04_processed_features/regression/naive/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_regression_naive)
    np.save(f'../data_set/preprocessed/04_processed_features/regression/reverse/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_regression_reverse)
    np.save(f'../data_set/preprocessed/04_processed_features/regression/rf/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_regression_rf)
    np.save(f'../data_set/preprocessed/04_processed_features/regression/svr/X_relevant_label_{label_index + 1}.npy',
            x_data_relevant_regression_svr)
