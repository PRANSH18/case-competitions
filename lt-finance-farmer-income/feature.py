import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer # Added KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import logging # Import the logging module
from sklearn.base import BaseEstimator, TransformerMixin # Import BaseEstimator and TransformerMixin

# Import ensemble models for stacking
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression # For meta-model and potentially a base model
from sklearn.cluster import KMeans # Keeping KMeans for clustering feature engineering
from xgboost import XGBRegressor # Import XGBoost Regressor for a base model

# Import feature selection
from sklearn.feature_selection import SelectKBest, f_regression # Added SelectKBest and f_regression

# --- Configure Logging ---
# Create and configure logger
LOG_FILE = 'farmer_income_prediction.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'w' overwrites the log file each time, 'a' appends
)
logger = logging.getLogger()
# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Starting farmer income prediction script.")

# --- 1. Load Data ---
# Load the training and test datasets
try:
    # Changed to pd.read_excel as per your query's context
    train_df = pd.read_excel('LTF Challenge data with dictionary - Copy.xlsx', sheet_name='TrainData')
    test_df = pd.read_excel('LTF Challenge data with dictionary - Copy.xlsx', sheet_name='TestData')
    dictionary_df = pd.read_excel('LTF Challenge data with dictionary - Copy.xlsx', sheet_name='Dictionary', header=1) # Header is on the second row
    logger.info("Data loaded successfully from Excel files.")
except FileNotFoundError as e:
    logger.error(f"Error loading file: {e}. Make sure the Excel file 'LTF Challenge data with dictionary - Copy.xlsx' is in the same directory.")
    exit()
except Exception as e:
    logger.error(f"An unexpected error occurred during data loading: {e}")
    exit()


# Display initial info for training data
logger.info("\n--- Train Data Info (Initial) ---")
train_df.info(buf=logger.handlers[0].stream) # Direct info to log file
logger.info("\n--- Test Data Info (Initial) ---")
test_df.info(buf=logger.handlers[0].stream) # Direct info to log file

# --- 2. Data Preprocessing and Feature Engineering ---

def preprocess_data(df):
    """
    Applies preprocessing and feature engineering steps to the DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # --- 2.1 Handle 'Location' column (Latitude, Longitude) ---
    # Split 'Location' into 'Latitude' and 'Longitude'
    # Handle cases where 'Location' might be missing or malformed
    if 'Location' in df.columns:
        df[['Latitude', 'Longitude']] = df['Location'].str.split(',', expand=True)
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.drop('Location', axis=1) # Drop original Location column
    else:
        logger.warning(" 'Location' column not found in DataFrame. Skipping Latitude/Longitude extraction.")


    # --- 2.2 Handle Ambient Temperature Columns ---
    # Columns like 'K022-Ambient temperature (min & max)' are strings "min / max"
    temp_cols = [col for col in df.columns if 'Ambient temperature (min & max)' in col]
    if temp_cols:
        for col in temp_cols:
            min_temp_col = col.replace(' (min & max)', '_Min')
            max_temp_col = col.replace(' (min & max)', '_Max')
            # Split and convert to numeric, coercing errors to NaN
            df[[min_temp_col, max_temp_col]] = df[col].str.split('/', expand=True)
            df[min_temp_col] = pd.to_numeric(df[min_temp_col], errors='coerce')
            df[max_temp_col] = pd.to_numeric(df[max_temp_col], errors='coerce')
            df = df.drop(col, axis=1) # Drop original combined temperature column
    else:
        logger.warning("No 'Ambient temperature (min & max)' columns found. Skipping temperature extraction.")


    # --- 2.3 Clean up other columns with non-numeric values that should be numeric ---
    # 'Non_Agriculture_Income' might have '[None]' string
    if 'Non_Agriculture_Income' in df.columns:
        df['Non_Agriculture_Income'] = df['Non_Agriculture_Income'].replace('[None]', np.nan)
        df['Non_Agriculture_Income'] = pd.to_numeric(df['Non_Agriculture_Income'], errors='coerce')
    else:
        logger.warning("'Non_Agriculture_Income' column not found. Skipping its cleaning.")


    # Many other columns might have non-numeric values that need to be coerced to numeric
    # Iterate through all columns and attempt to convert to numeric if appropriate
    for col in df.columns:
        # Exclude known categorical or identifier columns and the target variable
        if col not in ['FarmerID', 'State', 'REGION', 'SEX', 'CITY', 'DISTRICT', 'VILLAGE',
                       'MARITAL_STATUS', 'Address type', 'Ownership',
                       'K022-Village category based on Agri parameters (Good, Average, Poor)',
                       'K022-Nearest Mandi Name',
                       'K022-Village category based on socio-economic parameters (Good, Average, Poor)',
                       'R022-Village category based on Agri parameters (Good, Average, Poor)',
                       'Kharif Seasons Type of soil in 2022',
                       'Rabi Seasons Type of soil in 2022',
                       'Kharif Seasons Type of water bodies in hectares 2022',
                       'Rabi Seasons Type of water bodies in hectares 2022',
                       'Type of soil in 2020', 'Type of soil in 2021', 'Type of soil in 2022',
                       'Type of water bodies in hectares 2020', 'Type of water bodies in hectares 2021',
                       'Type of water bodies in hectares 2022',
                       'Target_Variable', 'Target_Variable/Total Income']: # Added 'Target_Variable/Total Income'
            # Try converting to numeric, if it fails, it will remain as is or become NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

logger.info("Starting data preprocessing and feature engineering.")
# Apply preprocessing to both train and test datasets
train_df_processed = preprocess_data(train_df)
test_df_processed = preprocess_data(test_df)
logger.info("Data preprocessing and feature engineering completed.")


# Determine the correct target variable column name
target_variable_col = None
if 'Target_Variable' in train_df_processed.columns:
    target_variable_col = 'Target_Variable'
elif 'Target_Variable/Total Income' in train_df_processed.columns:
    target_variable_col = 'Target_Variable/Total Income'
else:
    logger.error("Target variable column ('Target_Variable' or 'Target_Variable/Total Income') not found in training data.")
    exit()

# --- Target Variable Transformation ---
logger.info(f"Applying log1p transformation to target variable '{target_variable_col}'.")
# Apply log1p transformation to the target variable
train_df_processed[target_variable_col] = np.log1p(train_df_processed[target_variable_col])
logger.info("Target variable transformation completed.")


# Separate target variable from training data
X = train_df_processed.drop(target_variable_col, axis=1)
y = train_df_processed[target_variable_col]
logger.info(f"Target variable '{target_variable_col}' separated.")

# --- Clustering Step: Add Cluster Labels as a Feature ---
logger.info("Starting clustering to create farmer segments.")

# Identify numerical features for clustering from the current X DataFrame
# These features will be imputed and scaled before clustering
clustering_numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Create a temporary pipeline for numerical preprocessing specifically for clustering
clustering_numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Using SimpleImputer for clustering data prep
    ('scaler', StandardScaler())
])

# Fit and transform numerical features for clustering on training data
X_numerical_scaled_for_clustering = clustering_numerical_pipeline.fit_transform(X[clustering_numerical_features])

# Transform numerical features for clustering on test data using the fitted pipeline
test_numerical_scaled_for_clustering = clustering_numerical_pipeline.transform(test_df_processed[clustering_numerical_features])

# Determine optimal number of clusters (e.g., using Elbow method, or just pick a reasonable number)
# For demonstration, let's pick 5 clusters. In a real scenario, this would be tuned.
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robust centroid initialization

# Fit KMeans on the training data's scaled numerical features
kmeans.fit(X_numerical_scaled_for_clustering)

# Predict cluster labels for both training and test data
cluster_labels_train = kmeans.predict(X_numerical_scaled_for_clustering)
cluster_labels_test = kmeans.predict(test_numerical_scaled_for_clustering)

# Add cluster labels as a new categorical feature to the original (preprocessed) DataFrames
X['Cluster_Label'] = cluster_labels_train.astype(str) # Convert to string to treat as categorical
test_df_processed['Cluster_Label'] = cluster_labels_test.astype(str) # Convert to string to treat as categorical
logger.info(f"Clustering completed with {n_clusters} clusters. 'Cluster_Label' added as a new feature.")
# --- End Clustering Step ---


# Align columns between X (training features) and test_df_processed (test features)
# This is crucial after feature engineering (including adding Cluster_Label) and before one-hot encoding,
# as some columns might exist in one but not the other due to data sparsity.
# We will use the common columns for model training and prediction.
common_cols = list(set(X.columns) & set(test_df_processed.columns))
X = X[common_cols]
test_df_processed = test_df_processed[common_cols]
logger.info(f"Aligned columns for training and test sets. Number of common features: {len(common_cols)}")


# Identify numerical and categorical features for the preprocessor
# Re-identify features after adding 'Cluster_Label' so it's picked up as categorical
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

logger.info(f"Numerical features identified: {numerical_features}")
logger.info(f"Categorical features identified: {categorical_features}")

# --- Custom Transformer for Outlier Handling (Capping/Flooring) ---
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                col_data = X[col].dropna() # Get non-NaN values
                if not col_data.empty: # Check if there are any non-NaN values
                    self.lower_bounds[col] = np.percentile(col_data, self.lower_percentile)
                    self.upper_bounds[col] = np.percentile(col_data, self.upper_percentile)
                else:
                    # If all values are NaN, store NaN as bounds to indicate no capping
                    self.lower_bounds[col] = np.nan
                    self.upper_bounds[col] = np.nan
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            # Check if bounds were set for this column and they are not NaN
            if col in self.lower_bounds and not np.isnan(self.lower_bounds[col]):
                X_transformed[col] = np.clip(X_transformed[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_transformed

# --- 3. Create Preprocessing Pipelines ---
logger.info("Creating preprocessing pipelines with advanced features.")
# Numerical pipeline: Outlier capping, KNN Imputation, then scaling, then feature selection
numerical_transformer = Pipeline(steps=[
    ('outlier_capper', OutlierCapper(lower_percentile=1, upper_percentile=99)), # Outlier handling
    ('imputer', KNNImputer(n_neighbors=5)), # Advanced Imputation
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_regression, k=20)) # Feature Selection: Select top 20 features
])

# Categorical pipeline: Impute missing values with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'handle_unknown' to manage new categories in test set
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like FarmerID) if not transformed
)
logger.info("Preprocessing pipelines created.")

# --- 4. Model Training with Stacking Regressor ---
logger.info("--- Starting Stacking Regressor Training ---")

# Define base models for stacking
# Using a few diverse models typically works best
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
]

# Define the meta-model (final estimator)
# A simple model like LinearRegression or Ridge is often used here
final_estimator = LinearRegression()

# Create the StackingRegressor
# cv=3 for cross-validation within stacking, ensures base models' predictions are out-of-fold
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=3, # Number of cross-validation folds for base models
    n_jobs=-1 # Use all available cores for parallel processing of base models
)

# Create the full pipeline: Preprocessing + Stacking Regressor
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', stacking_regressor)])

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Splitting training data for validation.")

logger.info("--- Training Stacking Regressor ---")
full_pipeline.fit(X_train, y_train)
logger.info("Stacking Regressor training complete.")

# --- 5. Model Evaluation (on validation set using the stacking model) ---
logger.info("--- Evaluating Stacking Regressor on Validation Set ---")
y_pred_val_transformed = full_pipeline.predict(X_val)
# Inverse transform predictions for evaluation
y_pred_val = np.expm1(y_pred_val_transformed)
y_val_original = np.expm1(y_val) # Inverse transform original y_val for accurate metrics

mae = mean_absolute_error(y_val_original, y_pred_val)
r2 = r2_score(y_val_original, y_pred_val)

# Calculate MAPE
epsilon = 1e-10 # Small constant to avoid division by zero
mape = np.mean(np.abs((y_val_original - y_pred_val) / (y_val_original + epsilon))) * 100

logger.info(f"Mean Absolute Error (MAE) on validation set (Stacking Regressor): {mae:.2f}")
logger.info(f"R-squared (R2) on validation set (Stacking Regressor): {r2:.2f}")
logger.info(f"Mean Absolute Percentage Error (MAPE) on validation set (Stacking Regressor): {mape:.2f}%")
logger.info("Note: MAE indicates the average absolute difference between predicted and actual incomes.")
logger.info("R-squared indicates the proportion of variance in the dependent variable that is predictable from the independent variables.")
logger.info("MAPE indicates the average percentage difference between predicted and actual incomes.")

# --- 6. Make Predictions on the Test Data ---
logger.info("--- Making Predictions on Test Data ---")

# Predict income for the test dataset using the trained stacking pipeline
test_predictions_transformed = full_pipeline.predict(test_df_processed)
# Inverse transform predictions for final output
test_predictions = np.expm1(test_predictions_transformed)


# Convert predictions to integers (rounded) for saving
test_predictions_int = np.round(test_predictions).astype(int)

# Create a DataFrame for the predictions (FarmerID and Predicted_Income)
predictions_df = pd.DataFrame({
    'FarmerID': test_df['FarmerID'], # Use original test_df's FarmerID
    'Predicted_Income': test_predictions_int # Use integer predictions
})

logger.info("--- Sample Predictions for Test Data (first 10 rows) ---")
# Format 'Predicted_Income' to display as a whole number with commas for logging
predictions_df_display = predictions_df.copy()
predictions_df_display['Predicted_Income'] = predictions_df_display['Predicted_Income'].apply(lambda x: f"{x:,}")
logger.info(predictions_df_display.head(10).to_string()) # Display first 10 rows for logging

# --- 7. Store Predictions in Excel File ---
logger.info("--- Storing Predictions in Excel File ---")
output_excel_file = 'LTF Challenge data with dictionary - Copy.xlsx'
output_sheet_name = 'Predictions_Stacking_AdvPreproc' # Changed sheet name to reflect advanced preprocessing

try:
    # Merge predictions back into the original test_df for a comprehensive output
    # Ensure the merge is done on a copy to avoid SettingWithCopyWarning
    test_df_with_predictions = test_df.merge(predictions_df, on='FarmerID', how='left')

    # Load the existing Excel file to append a new sheet
    # 'openpyxl' engine is required for .xlsx files
    # 'mode='a'' is for append mode, 'if_sheet_exists='replace'' will overwrite if sheet exists
    with pd.ExcelWriter(output_excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        test_df_with_predictions.to_excel(writer, sheet_name=output_sheet_name, index=False)
    logger.info(f"Predictions successfully stored in '{output_excel_file}' on sheet '{output_sheet_name}'.")
except Exception as e:
    logger.error(f"Error saving predictions to Excel: {e}")

logger.info("Script finished.")
