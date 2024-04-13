import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
X_train = np.load('processed_data.npy')
y_train = np.load('y_train.npy')

# Convert the NumPy array to DataFrame for easier handling
X_df = pd.DataFrame(X_train)

# Create an imputer object with a strategy to fill missing values
imputer = SimpleImputer(strategy='median')  # You can change 'median' to 'mean' or 'most_frequent' as needed

# Apply the imputer to the DataFrame
X_df_imputed = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

# Iterate over columns to predict missing values using RandomForest
for column in X_df.columns:
    if X_df[column].isnull().any():  # Check if the original DataFrame had missing values
        # Split data into sets with and without missing values
        train_data = X_df_imputed[X_df[column].notnull()]
        predict_data = X_df_imputed[X_df[column].isnull()]

        # Prepare feature vectors
        train_features = train_data.drop(column, axis=1)
        train_labels = train_data[column]

        predict_features = predict_data.drop(column, axis=1)

        # Initialize the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestClassifier if it's categorical

        # Train the model
        model.fit(train_features, train_labels)

        # Predict the missing values
        predicted_values = model.predict(predict_features)

        # Fill in the missing values in the original DataFrame
        X_df.loc[X_df[column].isnull(), column] = predicted_values

# Convert the filled DataFrame back to a NumPy array
X_train_filled = X_df.values

# Save the filled data to a new .npy file
np.save('X_train_filled.npy', X_train_filled)
