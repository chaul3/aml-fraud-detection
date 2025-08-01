import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
from sklearn.datasets import make_classification
import requests
import zipfile
from pathlib import Path

class DataLoader:
    """
    Data loader class for AML fraud detection datasets.
    
    Supports loading from various sources including:
    - CSV files
    - Kaggle datasets
    - Synthetic data generation
    - Database connections
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"Loaded CSV data: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic AML fraud detection data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pandas DataFrame with synthetic transaction data
        """
        np.random.seed(self.config.get('random_state', 42))
        
        # Generate base features using make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=self.config.get('random_state', 42)
        )
        
        # Create feature names
        feature_names = [
            'transaction_amount', 'account_balance', 'transaction_frequency',
            'time_since_last_transaction', 'customer_age', 'account_age_days',
            'number_of_accounts', 'avg_transaction_amount', 'max_transaction_amount',
            'min_transaction_amount', 'transaction_amount_std', 'transaction_velocity',
            'cross_border_transactions', 'cash_transactions_ratio', 'weekend_transactions_ratio',
            'night_transactions_ratio', 'merchant_risk_score', 'customer_risk_score',
            'geographic_risk_score', 'payment_method_risk_score'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fraud'] = y
        
        # Add transaction ID and customer ID
        df['transaction_id'] = range(1, len(df) + 1)
        df['customer_id'] = np.random.randint(1, n_samples // 10, size=len(df))
        
        # Add timestamp
        start_date = pd.Timestamp('2023-01-01')
        df['timestamp'] = pd.date_range(
            start=start_date, 
            periods=len(df), 
            freq='1min'
        )
        
        # Transform some features to make them more realistic
        df['transaction_amount'] = np.exp(df['transaction_amount']) * 100
        df['account_balance'] = np.exp(df['account_balance']) * 1000
        df['customer_age'] = np.clip(np.abs(df['customer_age']) * 10 + 20, 18, 80)
        df['account_age_days'] = np.clip(np.abs(df['account_age_days']) * 100, 1, 3650)
        
        # Make percentages between 0 and 1
        percentage_cols = [
            'cash_transactions_ratio', 'weekend_transactions_ratio', 
            'night_transactions_ratio'
        ]
        for col in percentage_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Make risk scores between 0 and 100
        risk_cols = [
            'merchant_risk_score', 'customer_risk_score', 
            'geographic_risk_score', 'payment_method_risk_score'
        ]
        for col in risk_cols:
            df[col] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 100
        
        self.logger.info(f"Generated synthetic data: {df.shape}")
        self.logger.info(f"Fraud rate: {df['is_fraud'].mean():.3f}")
        
        return df
    
    def load_credit_card_fraud_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and process credit card fraud dataset.
        
        Args:
            file_path: Path to credit card fraud CSV file
            sample_size: Number of samples to return (None for all)
            
        Returns:
            pandas DataFrame
        """
        df = self.load_csv(file_path)
        
        # Rename target column to standard name
        if 'Class' in df.columns:
            df = df.rename(columns={'Class': 'is_fraud'})
        
        # Add transaction ID if not present
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = range(1, len(df) + 1)
        
        # Sample data if requested
        if sample_size and sample_size < len(df):
            # Stratified sampling to maintain fraud ratio
            fraud_samples = df[df['is_fraud'] == 1].sample(
                n=min(sample_size // 10, len(df[df['is_fraud'] == 1])),
                random_state=self.config.get('random_state', 42)
            )
            normal_samples = df[df['is_fraud'] == 0].sample(
                n=sample_size - len(fraud_samples),
                random_state=self.config.get('random_state', 42)
            )
            df = pd.concat([fraud_samples, normal_samples]).shuffle(
                random_state=self.config.get('random_state', 42)
            ).reset_index(drop=True)
        
        self.logger.info(f"Loaded credit card fraud data: {df.shape}")
        self.logger.info(f"Fraud rate: {df['is_fraud'].mean():.3f}")
        
        return df
    
    def download_sample_dataset(self, dataset_name: str, save_dir: str = "data/raw/") -> str:
        """
        Download sample datasets for testing.
        
        Args:
            dataset_name: Name of dataset to download
            save_dir: Directory to save downloaded data
            
        Returns:
            Path to downloaded file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Sample dataset URLs (these would be actual URLs in practice)
        datasets = {
            "sample_fraud": "https://example.com/sample_fraud.csv",
            "sample_transactions": "https://example.com/sample_transactions.csv"
        }
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not available")
        
        # For now, generate synthetic data instead of downloading
        if dataset_name == "sample_fraud":
            df = self.generate_synthetic_data(10000)
            file_path = os.path.join(save_dir, "sample_fraud.csv")
            df.to_csv(file_path, index=False)
            self.logger.info(f"Generated sample fraud dataset: {file_path}")
            return file_path
        
        return ""
    
    def load_data(self, data_source: str, **kwargs) -> pd.DataFrame:
        """
        Main method to load data from various sources.
        
        Args:
            data_source: Source of data ('csv', 'synthetic', 'credit_card')
            **kwargs: Additional arguments specific to data source
            
        Returns:
            pandas DataFrame
        """
        if data_source == 'synthetic':
            return self.generate_synthetic_data(
                kwargs.get('n_samples', self.config.get('sample_size', 10000))
            )
        elif data_source == 'csv':
            return self.load_csv(kwargs['file_path'], **kwargs)
        elif data_source == 'credit_card':
            return self.load_credit_card_fraud_data(
                kwargs['file_path'],
                kwargs.get('sample_size', self.config.get('sample_size'))
            )
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the dataset.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
        
        if 'is_fraud' in df.columns:
            info['fraud_rate'] = df['is_fraud'].mean()
            info['class_distribution'] = df['is_fraud'].value_counts().to_dict()
        
        return info
