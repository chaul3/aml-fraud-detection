import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

class Preprocessor:
    """
    Data preprocessing class for AML fraud detection.
    
    Handles:
    - Data cleaning and validation
    - Feature engineering
    - Feature selection
    - Data scaling and normalization
    - Handling imbalanced datasets
    - Train/validation/test splitting
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Log initial state
        self.logger.info(f"Initial data shape: {df_clean.shape}")
        self.logger.info(f"Missing values: {df_clean.isnull().sum().sum()}")
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Convert data types if needed
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        self.logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features for better fraud detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Time-based features
        if 'timestamp' in df_eng.columns:
            df_eng['hour'] = df_eng['timestamp'].dt.hour
            df_eng['day_of_week'] = df_eng['timestamp'].dt.dayofweek
            df_eng['is_weekend'] = (df_eng['day_of_week'] >= 5).astype(int)
            df_eng['is_night'] = ((df_eng['hour'] >= 22) | (df_eng['hour'] <= 6)).astype(int)
            
            # Sort by timestamp for rolling features
            df_eng = df_eng.sort_values('timestamp')
        
        # Transaction amount features
        if 'transaction_amount' in df_eng.columns:
            df_eng['log_transaction_amount'] = np.log1p(df_eng['transaction_amount'])
            df_eng['transaction_amount_squared'] = df_eng['transaction_amount'] ** 2
            
            # Rolling statistics by customer
            if 'customer_id' in df_eng.columns:
                df_eng['customer_avg_amount'] = df_eng.groupby('customer_id')['transaction_amount'].transform('mean')
                df_eng['customer_std_amount'] = df_eng.groupby('customer_id')['transaction_amount'].transform('std')
                df_eng['amount_deviation_from_customer_avg'] = (
                    df_eng['transaction_amount'] - df_eng['customer_avg_amount']
                ) / (df_eng['customer_std_amount'] + 1e-8)
                
                # Customer transaction frequency
                df_eng['customer_transaction_count'] = df_eng.groupby('customer_id').cumcount() + 1
        
        # Account balance features
        if 'account_balance' in df_eng.columns:
            df_eng['log_account_balance'] = np.log1p(df_eng['account_balance'])
            
            if 'transaction_amount' in df_eng.columns:
                df_eng['balance_to_transaction_ratio'] = (
                    df_eng['account_balance'] / (df_eng['transaction_amount'] + 1e-8)
                )
        
        # Risk score combinations
        risk_cols = [col for col in df_eng.columns if 'risk_score' in col]
        if len(risk_cols) > 1:
            df_eng['combined_risk_score'] = df_eng[risk_cols].mean(axis=1)
            df_eng['max_risk_score'] = df_eng[risk_cols].max(axis=1)
            df_eng['risk_score_std'] = df_eng[risk_cols].std(axis=1)
        
        # Interaction features
        if 'customer_age' in df_eng.columns and 'account_age_days' in df_eng.columns:
            df_eng['customer_account_age_ratio'] = (
                df_eng['customer_age'] * 365 / (df_eng['account_age_days'] + 1)
            )
        
        # Categorical encoding for any remaining categorical variables
        categorical_cols = df_eng.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['timestamp']]
        
        for col in categorical_cols:
            if col != 'is_fraud':  # Don't encode target variable
                # Use one-hot encoding for low cardinality, label encoding for high cardinality
                if df_eng[col].nunique() <= 10:
                    dummies = pd.get_dummies(df_eng[col], prefix=col, drop_first=True)
                    df_eng = pd.concat([df_eng, dummies], axis=1)
                    df_eng.drop(col, axis=1, inplace=True)
                else:
                    # Label encoding
                    df_eng[col] = pd.Categorical(df_eng[col]).codes
        
        self.logger.info(f"Feature engineering completed. New shape: {df_eng.shape}")
        return df_eng
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select the most relevant features for fraud detection.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with selected features
        """
        feature_config = self.config.get('feature_selection', {})
        method = feature_config.get('method', 'mutual_info')
        k_best = feature_config.get('k_best', 50)
        
        # Ensure k_best doesn't exceed number of features
        k_best = min(k_best, X.shape[1])
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        elif method == 'chi2':
            # Chi2 requires non-negative values
            X_min = X.min().min()
            if X_min < 0:
                X = X - X_min  # Shift to make all values non-negative
            selector = SelectKBest(score_func=chi2, k=k_best)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k_best)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k_best)
        else:
            self.logger.warning(f"Unknown feature selection method: {method}. Using mutual_info.")
            selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()
        else:
            # For RFE
            selected_features = X.columns[selector.support_].tolist()
        
        self.selected_features = selected_features
        self.feature_selector = selector
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.logger.info(f"Feature selection completed. Selected {len(selected_features)} features from {X.shape[1]}")
        
        return X_selected_df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using the specified scaling method.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled feature matrix
        """
        scaling_config = self.config.get('scaling', {})
        method = scaling_config.get('method', 'standard')
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
            scaler = StandardScaler()
        
        if fit:
            X_scaled = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.logger.info(f"Features scaled using {method} scaling")
        return X_scaled_df
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced dataset using resampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Resampled feature matrix and target variable
        """
        resampling_config = self.config.get('resampling', {})
        method = resampling_config.get('method', 'smote')
        ratio = resampling_config.get('ratio', 'auto')
        
        initial_counts = y.value_counts()
        self.logger.info(f"Initial class distribution: {initial_counts.to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=ratio, random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=ratio, random_state=42)
        elif method == 'random_oversample':
            sampler = RandomOverSampler(sampling_strategy=ratio, random_state=42)
        elif method == 'random_undersample':
            sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
        else:
            self.logger.warning(f"Unknown resampling method: {method}. No resampling applied.")
            return X, y
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            final_counts = y_resampled.value_counts()
            self.logger.info(f"Final class distribution after {method}: {final_counts.to_dict()}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error in resampling: {e}")
            self.logger.info("Returning original data without resampling")
            return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('validation_size', 0.1)
        random_state = self.config.get('random_state', 42)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"Train set: {X_train.shape}, Fraud rate: {y_train.mean():.3f}")
        self.logger.info(f"Validation set: {X_val.shape}, Fraud rate: {y_val.mean():.3f}")
        self.logger.info(f"Test set: {X_test.shape}, Fraud rate: {y_test.mean():.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary containing preprocessed data splits and metadata
        """
        self.logger.info("Starting preprocessing pipeline")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Engineer features
        df_engineered = self.engineer_features(df_clean)
        
        # Step 3: Separate features and target
        # Remove non-feature columns
        exclude_cols = [target_col, 'transaction_id', 'customer_id', 'timestamp']
        feature_cols = [col for col in df_engineered.columns if col not in exclude_cols]
        
        X = df_engineered[feature_cols]
        y = df_engineered[target_col]
        
        # Step 4: Split data first (before feature selection and scaling)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 5: Feature selection (fit on training data only)
        X_train_selected = self.select_features(X_train, y_train)
        X_val_selected = pd.DataFrame(
            self.feature_selector.transform(X_val), 
            columns=self.selected_features,
            index=X_val.index
        )
        X_test_selected = pd.DataFrame(
            self.feature_selector.transform(X_test), 
            columns=self.selected_features,
            index=X_test.index
        )
        
        # Step 6: Scale features (fit on training data only)
        X_train_scaled = self.scale_features(X_train_selected, fit=True)
        X_val_scaled = self.scale_features(X_val_selected, fit=False)
        X_test_scaled = self.scale_features(X_test_selected, fit=False)
        
        # Step 7: Handle imbalanced data (only on training set)
        X_train_resampled, y_train_resampled = self.handle_imbalanced_data(X_train_scaled, y_train)
        
        result = {
            'X_train': X_train_resampled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_resampled,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.selected_features,
            'original_features': feature_cols,
            'preprocessing_info': {
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': self.selected_features,
                'original_shape': df.shape,
                'final_train_shape': X_train_resampled.shape
            }
        }
        
        self.logger.info("Preprocessing pipeline completed successfully")
        return result
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed data ready for prediction
        """
        if self.scaler is None or self.feature_selector is None:
            raise ValueError("Preprocessors not fitted. Run preprocess_pipeline first.")
        
        # Apply same preprocessing steps
        df_clean = self.clean_data(df)
        df_engineered = self.engineer_features(df_clean)
        
        # Select same features as training
        # First get all original features (not just selected ones)
        available_features = [col for col in df_engineered.columns 
                            if col not in ['is_fraud', 'transaction_id', 'customer_id', 'timestamp']]
        
        # Make sure we have all the features that were used in training
        missing_features = [col for col in self.selected_features if col not in available_features]
        if missing_features:
            self.logger.warning(f"Missing features in new data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                df_engineered[feature] = 0
        
        # Select and scale features
        X = df_engineered[self.selected_features]
        X_scaled = self.scale_features(X, fit=False)
        
        return X_scaled
