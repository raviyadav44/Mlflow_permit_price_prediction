import pandas as pd
import numpy as np
import os
import mlflow
import logging
import pickle
import re
import matplotlib.pyplot as plt
import shap

from fuzzywuzzy import fuzz
from scipy.stats import randint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import (
    train_test_split, cross_val_score, LeaveOneOut, 
    KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlflow.models.signature import infer_signature


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('workflow_logger')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'workflow_logger.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url, keep_default_na=True, 
                         na_values=['', 'NA', 'N/A', 'null'])
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # Columns to drop initially
        cols_to_drop = [
            'id', 'first_name', 'last_name', 'contact_person',
            'client_company_name', 'client_trn', 'client_address_street','event_audience_type',
            'event_mode','event_entertainment_theatre_category', 'event_entertainment_dance_category',
            'event_entertainment_dj_category', 'event_entertainment_amusement_category',
            'event_entertainment_music_category', 'ticketing_delegates_data','event_beneficial_organisation',
            'client_address_line2', 'client_city', 'client_country','event_vip', 'event_fund_raising',
            'client_phone', 'client_email', 'event_name', 'event_website','event_is_crypto', 'event_multi_venue',
            'event_organizer_name', 'event_organizer_details', 'event_max_speakers',
            'event_manager', 'event_owner', 'created_at', 'updated_at','event_max_attendees','event_sub_venue',
            'event_crypto_names','Obs','ticketing_selling_link','event_profile','ticketing_category_prices', 'ticketing_category_sell_estimate' #since ticeting_selling_link is not used in calculatign the permit price
        ]

        df.drop(columns = cols_to_drop, inplace = True)
        df = df.dropna(how='all')  # Drop rows that are all NaN
        df = df.drop_duplicates()  # Drop duplicate rows
        df=df.dropna(axis=1, how='all')  # Drop columns that are all NaN
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price']).copy() # Drop rows where 'Price' is NaN after conversion
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def type_safe_text_cleaner(X, columns):
    """
    Ensures specified text columns in a DataFrame are properly string-typed and cleaned.
    Equivalent to the TypeSafeTextCleaner class.
    """
    logger.info(f"type_safe_text_cleaner() - Columns to clean: {columns}")
    logger.info(f"type_safe_text_cleaner() - Input shape: {X.shape}")
    
    X = X.copy()

    for col in columns:
        if col in X.columns:
            # logger.debug(f"Cleaning column: {col}")
            original_sample = X[col].iloc[0] if len(X) > 0 else "EMPTY"

            X[col] = (
                X[col]
                .fillna("")  # avoid converting NaN to 'nan'
                .astype(str)
                .replace({'nan': '', 'None': '', '<NA>': ''})
                .str.strip()
                .str.lower()
            )

            cleaned_sample = X[col].iloc[0] if len(X) > 0 else "EMPTY"
            logger.debug(f"Column '{col}' cleaned - Before: '{original_sample}', After: '{cleaned_sample}'")
    
    logger.info(f"Text cleaning complete - Output shape: {X.shape}")
    return X


def CategoricalCleaner(X):
    """
    Cleans specific categorical columns with custom rules.
    Equivalent to the CategoricalCleaner class.
    """
    X = X.copy()

    # Clean event_business_category
    if 'event_business_category' in X.columns:
        X['event_business_category'] = (
            X['event_business_category']
            .replace(['', 'nan', '<nan>', '<blank>'], np.nan)
            .replace({
                'networking, conference': 'conference, networking',
                'meeting, conference': 'conference, meeting',
                'networking, congress': 'congress, networking',
                'congress, conference': 'conference, congress'
            })
        )

    # Clean event_emirate
    if 'event_emirate' in X.columns:
        valid_emirates = [
            'dubai', 'abu dhabi', 'sharjah', 'ajman',
            'umm al quwain', 'ras al khaimah', 'fujairah'
        ]

        X['event_emirate'] = (
            X['event_emirate']
            .replace('', np.nan)
            .replace([
                'dubai', 'DUBAI', 'Dubai, UAE', 'Dubai-UAE',
                'Sheraton Grand Dubai', 'Oud Metha Road.*',
                'Sheraton Grand Hotel.*'
            ], 'dubai')
            .replace([
                'no', 'na', 'test', 'yes', 'world trade center',
                'palm', 'de', 'select one', 'jafza one',
                'select state', '.*sheikh zayed road.*', '????? ????'
            ], np.nan)
            .apply(lambda x: x if x in valid_emirates else np.nan)
            .str.title()
        )

    return X

def extract_date_features(X, start_date_col, end_date_col):
    """
    Extracts features from date columns:
    - event_duration_days
    - days_until_event
    - event_month
    - event_day_of_week

    Equivalent to the DateFeatureExtractor class.
    """
    X = X.copy()

    X[start_date_col] = pd.to_datetime(X[start_date_col], errors='coerce')
    X[end_date_col] = pd.to_datetime(X[end_date_col], errors='coerce')

    X['event_duration_days'] = (X[end_date_col] - X[start_date_col]).dt.days
    X['days_until_event'] = (X[start_date_col] - pd.to_datetime('today')).dt.days
    X['event_month'] = X[start_date_col].dt.month
    X['event_day_of_week'] = X[start_date_col].dt.dayofweek

    return X.drop(columns=[start_date_col, end_date_col])




def clean_ticketing_badge_assist_column(df, column_name='ticketing_badge_assist'):
    """
    Comprehensive cleaning function for ticketing_badge_assist column.
    Performs all preprocessing steps and replaces the original column with cleaned data.
    
    Parameters:
    - df: Input DataFrame
    - column_name: Name of the column to clean (default: 'ticketing_badge_assist')
    
    Returns:
    - DataFrame with the cleaned column (all other columns preserved)
    """
    
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert to string, lowercase, and strip whitespace
    cleaned = df[column_name].astype(str).str.lower().str.strip()
    
    # Helper function for regex matching
    def regex_replace(text, patterns, replacement):
        for pattern in patterns:
            if re.search(pattern, str(text)):
                return replacement
        return text
    
    # Define patterns for each category
    no_patterns = [
        r'^no$',
        r'^no\b',
        r'no thank you',
        r'no need',
        r'not required',
        r'no, we use cvent',
        r'no, we only require',
        r'we already have the desk',
        r'no badge assistance required'
    ]
    
    yes_patterns = [
        r'^yes$',
        r'yes, for tickets? scanning',
        r'yes \d+ hostess',
        r'yes badge printing',
        r'yes for badge printing',
        r'yes, qr code',
        r'yes we will have',
        r'registration desk.*mark people as attended',
        r'please provide a table for',
        r'requires ticketing assistance',
        r'need help with badge'
    ]
    
    maybe_patterns = [
        r'maybe',
        r'possibly',
        r'potentially',
        r'tbc',
        r'tbd',
        r'assistance',
        r'we may need assistance',
        r'please provide a table',
        r'note - we plan on printing',
        r'open to exploring',
        r'to be confirmed',
        r'to be determined',
        r'still deciding',
        r'checking on requirements'
    ]
    
    # Standard replacements first
    cleaned = cleaned.replace({
        'no': 'no',
        'yes': 'yes',
        'maybe': 'maybe',
        'nan': np.nan,
        '': np.nan,
        'na': np.nan,
        'none': np.nan,
        'null': np.nan,
        '1': np.nan,
        '1250.0': np.nan,
        '321.0': np.nan,
        'please provide a table for the registration desk outside the ballroom.': 'yes',
        'we may need assistance checking individuals in.': 'maybe'
    })
    
    # Apply the regex replacements
    cleaned = cleaned.apply(
        lambda x: 'no' if regex_replace(x, no_patterns, 'no') == 'no' else x
    )
    cleaned = cleaned.apply(
        lambda x: 'yes' if regex_replace(x, yes_patterns, 'yes') == 'yes' else x
    )
    cleaned = cleaned.apply(
        lambda x: 'maybe' if regex_replace(x, maybe_patterns, 'maybe') == 'maybe' else x
    )
    
    # Convert to proper case
    cleaned = cleaned.str.capitalize()
    
    # Final validation - keep only our standardized values
    valid_values = ['Yes', 'No', 'Maybe', np.nan]
    cleaned = cleaned.apply(lambda x: x if x in valid_values else np.nan)
    
    # Replace the original column with cleaned data
    df[column_name] = cleaned
    
    return df


def standardize_venues(df, venue_column='event_venue', threshold=85):
    """
    Standardizes venue names in a DataFrame while preserving all columns.
    
    Parameters:
    - df: Input DataFrame
    - venue_column: Name of the column containing venue information
    - threshold: Fuzzy matching similarity threshold (0-100)
    
    Returns:
    - DataFrame with an additional 'standardized_venue' column and all original columns
    """
    
    # Make a copy of the original DataFrame to avoid modifying it
    df = df.copy()
    
    def preprocess_venue(text):
        if pd.isna(text):
            return ''
            
        text = str(text).lower()
        
        # Remove punctuation except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Standardize common terms
        replacements = {
            'uae': '',
            'united arab emirates': '',
            'dubai': '',
            'dubayy': '',
            'hotel': '',
            'resort': '',
            'spa': '',
            'conference centre': 'conference',
            'conference hall': 'conference',
            'tbc': 'to be confirmed',
            'tba': 'to be announced',
            'tbd': 'to be decided',
            '&': 'and',
            'palm jumeirah': 'palm',
            'jbr': 'jumeirah beach residence',
            'difc': 'dubai international financial centre'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove anything in parentheses
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove address-like components
        text = re.sub(r'\b(p\.?o\.? box|po|building|block|road|street)\b.*?\d+', '', text)
        text = re.sub(r'\d+', '', text)  # Remove any remaining numbers
        
        # Remove common stop words
        stop_words = {'the', 'at', 'in', 'on', 'by', 'a', 'an', 'of', 'and'}
        text = ' '.join(word for word in text.split() if word not in stop_words)
        
        # Collapse multiple spaces and trim
        text = ' '.join(text.split())
        
        return text.strip()
    
    # Preprocess all venues
    processed_venues = df[venue_column].apply(preprocess_venue)
    
    # Create a dictionary to map original venues to standardized versions
    venue_mapping = {}
    standardized_venues = []
    
    for original, processed in zip(df[venue_column], processed_venues):
        if pd.isna(original):
            standardized_venues.append('')
            continue
            
        # Find the best match among already standardized venues
        best_match = None
        highest_score = 0
        
        for standardized in venue_mapping.values():
            score = fuzz.ratio(processed, standardized)
            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = standardized
        
        if best_match is not None:
            standardized_venues.append(best_match)
        else:
            # If no good match found, use the processed version as new standard
            standardized = processed
            venue_mapping[original] = standardized
            standardized_venues.append(standardized)
    
    # Add the standardized venue column to the DataFrame
    df['standardized_venue'] = standardized_venues
    
    return df

def clean_ticketing_selling_option(df, column_name='ticketing_selling_option'):
    """
    Cleans and standardizes the ticketing_selling_option column.
    
    Parameters:
    - df: Input DataFrame
    - column_name: Name of the column to clean (default: 'ticketing_selling_option')
    
    Returns:
    - DataFrame with the cleaned column
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert to string, lowercase, and strip whitespace
    cleaned = df[column_name].astype(str).str.lower().str.strip()
    
    # Standardize values
    cleaned = cleaned.replace({
        'no': 'no',
        'yes': 'yes',
        'nan': np.nan,
        '': np.nan,
        'null': np.nan,
        'none': np.nan,
        'n/a': np.nan,
        'na': np.nan
    })
    
    # Convert anything that's not 'yes' or 'no' to 'other'
    cleaned = cleaned.apply(lambda x: 'other' if x not in ['yes', 'no', np.nan] else x)
    
    # Capitalize first letter for consistency
    cleaned = cleaned.str.capitalize()
    
    # Replace the original column
    df[column_name] = cleaned
    
    return df



def preprocess_datetime_features(df):
    df = df.copy()
    
    # Convert to datetime if not already
    df['ticketing_selling_start_date'] = pd.to_datetime(df['ticketing_selling_start_date'])
    df['ticketing_selling_end_date'] = pd.to_datetime(df['ticketing_selling_end_date'])
    
    # 1. Duration Features
    df['selling_duration_days'] = (df['ticketing_selling_end_date'] - df['ticketing_selling_start_date']).dt.days
    
    # Drop original date columns if desired
    df.drop(columns=['ticketing_selling_start_date', 'ticketing_selling_end_date'], inplace=True)
    
    return df

def transform_numeric_features(df, numeric_columns):
    """
    Transforms numeric features in a DataFrame:
    - Selects specified numeric columns
    - Imputes missing values with median
    - Applies StandardScaler
    Returns the full DataFrame with only numeric_columns modified
    """
    df = df.copy()
    X_numeric = df[numeric_columns].astype(np.float64)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Fit and transform the numeric columns
    X_imputed = imputer.fit_transform(X_numeric)
    X_scaled = scaler.fit_transform(X_imputed)

    # Replace the original numeric columns with the scaled values
    df[numeric_columns] = pd.DataFrame(X_scaled, columns=numeric_columns, index=df.index)

    return df

def comma_tokenizer(x):
    return x.split(',')

class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
        self.onehot_encoder = None
        self.count_vectorizers = {}
        self.scaler = None
        self.binary_columns_map = None
        
    def fit(self, df,y=None):
        """Fit all transformers on training data"""
        # Define column groups
        self.categorical_cols = [
            'event_emirate', 'ticketing_category', 
            'standardized_venue', 'ticketing_badge_assist',
            'ticketing_selling_option'
        ]
        
        self.multi_label_cols = [
            'event_type', 'event_entertainment_category',
            'event_business_category', 'event_industry_type'
        ]
        
        self.numeric_cols = [
            'event_max_performers', 'ticketing_is_complementary_pass',
            'event_duration_days', 'days_until_event', 
            'event_month', 'event_day_of_week', 'selling_duration_days'
        ]
        
        self.binary_cols = [
            'event_within_company', 'event_exhibition',
            'ticketing_is_event_free', 'ticketing_is_reg_desk',
            'ticketing_is_print_badges', 'ticketing_is_walkin_expect'
        ]
        
        # Fit OneHotEncoder for categorical columns
        # logger.info("Fitted OneHotEncoder on categorical columns: %s", self.categorical_cols)
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.onehot_encoder.fit(df[self.categorical_cols])
        
        
        # Fit CountVectorizers for multi-label columns
        for col in self.multi_label_cols:
            # logger.info("Fitted CountVectorizer on multi-label column: %s", col)
            vectorizer = CountVectorizer(
                tokenizer=comma_tokenizer,
                token_pattern=None,
                binary=True,
                lowercase=True
            )
            vectorizer.fit(df[col].fillna(''))
            self.count_vectorizers[col] = vectorizer
            
            
        # Fit StandardScaler for numeric columns
        # logger.info("Fitted StandardScaler on numeric columns: %s", self.numeric_cols)
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.numeric_cols])
        
        
        # Create binary columns mapping
        self.binary_columns_map = {
            'yes': 1,
            'no': 0,
            'true': 1,
            'false': 0,
            '1': 1,
            '0': 0,
            't': 1,
            'f': 0
        }
        
        self.fitted = True
        return self
    
    def transform(self, df):
        """Apply all transformations"""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet. Call fit() first.")
            
        df = df.copy()
        
        # Apply OneHotEncoding
        # logger.info("Transforming categorical columns: %s", self.categorical_cols)
        encoded = self.onehot_encoder.transform(df[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.onehot_encoder.get_feature_names_out(self.categorical_cols),
            index=df.index
        )
        df = df.drop(columns=self.categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
        
        # Apply CountVectorizer for multi-label
        for col in self.multi_label_cols:
            # logger.info("Transforming multi-label column: %s", col)
            vectorizer = self.count_vectorizers[col]
            transformed = vectorizer.transform(df[col].fillna(''))
            features = pd.DataFrame(
                transformed.toarray(),
                columns=[f"{col}__{feat}" for feat in vectorizer.get_feature_names_out()],
                index=df.index
            )
            df = df.drop(columns=[col])
            df = pd.concat([df, features], axis=1)
            
        # Apply StandardScaler
        # logger.info("Transforming numeric columns: %s", self.numeric_cols)
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        
        # Process binary columns
        logger.info("Processing binary columns: %s", self.binary_cols)
        for col in self.binary_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .map(self.binary_columns_map)
                    .fillna(0) ) # Treat missing/unknown as 0
                # Ensure values are either 0 or 1
                df[col] = np.where(df[col].isin([0, 1]), df[col], 0)
            
        return df
    
def main():
    # Set MLflow experiment name
    experiment_name = "random_forest_experiment_2"
    mlflow.set_experiment(experiment_name)
    
    # Enable automatic logging
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        log_datasets=True
    )
    
    try:
        # params = load_params(params_path='params.yaml')
        logger.info("Starting regression pipeline...")
        data_path = 'https://docs.google.com/spreadsheets/d/1sgMtUpG2YNNMAsLiPpJj8T3tZXvgtq19Qug18wiDwMY/export?format=csv'
        
        # Start MLflow parent run
        with mlflow.start_run(run_name="main_experiment") as run:
            # Load and log raw data
            df = load_data(data_url=data_path)

            # Save the loaded DataFrame as a CSV locally
            raw_data_path = "raw_data.csv"
            df.to_csv(raw_data_path, index=False)
            mlflow.log_artifact(raw_data_path, "raw_data")
            
            logger.info("Preprocessing data...")
            df = preprocess_data(df)
            df = standardize_venues(df, venue_column='event_venue')
            df = df.drop(columns=['event_venue'])
            
            # Clean specific columns
            df = clean_ticketing_badge_assist_column(df)
            df = clean_ticketing_selling_option(df)
            
            # Text cleaning
            text_columns = [
                'event_type', 'event_entertainment_category',
                'event_business_category', 'event_industry_type',
                'event_emirate', 'ticketing_category',
                'standardized_venue', 'ticketing_badge_assist',
                'ticketing_selling_option'
            ]
            logger.info("Cleaning text columns...")
            df = type_safe_text_cleaner(df, columns=text_columns)
            
            # Categorical cleaning
            logger.info("Processing categorical features...")
            df = CategoricalCleaner(df)
            
            # Date features
            logger.info("Extracting date features...")
            df = extract_date_features(df, 'event_start_date', 'event_end_date')
            df = preprocess_datetime_features(df)

            # Numeric feature transformation
            numeric_cols = [
                'event_max_performers', 'ticketing_is_complementary_pass',
                'event_duration_days', 'days_until_event', 
                'event_month', 'event_day_of_week', 'selling_duration_days'
            ]
            logger.info("Transforming numeric features...")
            df = transform_numeric_features(df, numeric_cols)

            # Save and log processed data
            processed_data_path = 'processed_data.csv'
            df.to_csv(processed_data_path, index=False)
            mlflow.log_artifact(processed_data_path, "processed_data")
            logger.info("Data preprocessing completed successfully.")

            X = df.drop(columns=['Price'])
            y = df['Price']
            
            # Log dataset stats
            mlflow.log_params({
                "dataset_size": len(df),
                "num_features": X.shape[1],
                "target_mean": y.mean(),
                "target_std": y.std()
            })
            
            # Log sample of data
            mlflow.log_text(df.head().to_csv(), "data_sample.csv")
            
            # Random Forest Pipeline
            model = Pipeline([
                ('preprocessor', FeaturePreprocessor()),
                ('regressor', RandomForestRegressor(random_state=42))
            ])

            # K-Fold Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            metrics_list = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    metrics = {
                        'MSE': mean_squared_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'R2': r2_score(y_test, y_pred)
                    }
                    metrics_list.append(metrics)
                    
                    # Log fold-specific artifacts
                    fold_results = pd.DataFrame({
                        'actual': y_test,
                        'predicted': y_pred
                    })
                    fold_results.to_csv(f"fold_{fold}_results.csv", index=False)
                    mlflow.log_artifact(f"fold_{fold}_results.csv")

            # Log aggregated metrics
            metrics_df = pd.DataFrame(metrics_list)
            avg_metrics = metrics_df.mean()
            mlflow.log_metrics({
                "cv_avg_mse": avg_metrics['MSE'],
                "cv_avg_rmse": avg_metrics['RMSE'],
                "cv_avg_mae": avg_metrics['MAE'],
                "cv_avg_r2": avg_metrics['R2']
            })
            
            # Hyperparameter Tuning
            with mlflow.start_run(run_name="hyperparameter_tuning", nested=True):
                param_dist = {
                    'regressor__n_estimators': randint(50, 500),
                    'regressor__max_depth': [None, 5, 10, 20, 30],
                    'regressor__min_samples_split': randint(2, 20),
                    'regressor__min_samples_leaf': randint(1, 10),
                    'regressor__max_features': ['sqrt', 'log2', None]
                }

                random_search = RandomizedSearchCV(
                    model,
                    param_distributions=param_dist,
                    n_iter=20,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    verbose=1,
                    random_state=42
                )
                random_search.fit(X, y)
                
                # Log best model to registry
                best_model = random_search.best_estimator_
                signature = infer_signature(X, best_model.predict(X))
                
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name="price_prediction_rf",
                    signature=signature,
                    input_example=X.iloc[:1],
                    metadata={
                        "model_type": "RandomForestRegressor",
                        "dataset": "permit_pricing"
                    }
                )
                
                # Log detailed feature importance
                importances = best_model.named_steps['regressor'].feature_importances_
                try:
                    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
                except AttributeError:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                fi_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                
                # Log SHAP values for model interpretability
                try:
                    explainer = shap.TreeExplainer(best_model.named_steps['regressor'])
                    shap_values = explainer.shap_values(best_model.named_steps['preprocessor'].transform(X))
                    
                    plt.figure()
                    shap.summary_plot(shap_values, 
                                    best_model.named_steps['preprocessor'].transform(X),
                                    feature_names=feature_names,
                                    show=False)
                    plt.savefig("shap_summary.png")
                    plt.close()
                    mlflow.log_artifact("shap_summary.png")
                except Exception as e:
                    logger.warning(f"Could not generate SHAP plot: {str(e)}")
                    mlflow.log_param("shap_error", str(e))

            logger.info("Pipeline completed successfully!")
            
            # Log environment info
            # mlflow.log_artifact("requirements.txt", "environment")
            
            # Tag the successful run
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("model_type", "random_forest")

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        mlflow.log_param("error", str(e))
        mlflow.set_tag("status", "failed")
        raise

if __name__ == "__main__":
    main()
    logger.info('Data ingestion and preprocessing completed successfully.')