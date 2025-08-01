# AML Fraud Detection Project

## Overview
This project implements an Anti-Money Laundering (AML) fraud detection system using dynamic thresholds and multiple machine learning algorithms including anomaly detection, clustering, and supervised learning.

## Objectives
- Develop dynamic thresholds for detecting suspicious activities based on historical user data
- Apply anomaly detection, clustering, and supervised learning algorithms to identify potential AML cases
- Focus on outlier detection with approximately 10k sample data

## Recommended Datasets

### 1. IEEE-CIS Fraud Detection Dataset (Primary Choice)
- **Source**: Kaggle IEEE-CIS Fraud Detection Competition
- **Size**: ~590,000 transactions (can be sampled to 10k)
- **Features**: 433 features including transaction amounts, time features, and anonymized categorical features
- **Labels**: Binary fraud classification (0/1)
- **Download**: `https://www.kaggle.com/c/ieee-fraud-detection/data`

### 2. Credit Card Fraud Detection (Alternative)
- **Source**: Kaggle
- **Size**: 284,807 transactions (easily sampled to 10k)
- **Features**: 30 features (mostly PCA-transformed for privacy)
- **Labels**: Binary fraud classification
- **Download**: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

### 3. Synthetic Financial Dataset (Backup)
- **Source**: PaySim - Financial Dataset for Fraud Detection
- **Size**: 6+ million transactions (easily sampled)
- **Features**: Transaction type, amount, origin/destination balances
- **Download**: `https://www.kaggle.com/datasets/ealaxi/paysim1`

### 4. AML Transaction Monitoring Dataset
- **Source**: Synthetic AML dataset
- **Features**: Customer demographics, transaction patterns, risk scores
- **Size**: Configurable (we'll generate ~10k samples)

## Project Structure

```
AML-Fraud-Detection/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and preprocessed data
│   └── samples/                # 10k sample datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_clustering_analysis.ipynb
│   ├── 05_supervised_learning.ipynb
│   └── 06_dynamic_thresholds.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py
│   │   ├── clustering.py
│   │   ├── supervised_models.py
│   │   └── dynamic_thresholds.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── config/
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Algorithms to Implement

### 1. Anomaly Detection
- **Isolation Forest**: For detecting outliers in transaction patterns
- **One-Class SVM**: For novelty detection in user behavior
- **Local Outlier Factor (LOF)**: For density-based anomaly detection
- **Autoencoders**: Deep learning approach for pattern recognition

### 2. Clustering
- **K-Means**: For customer segmentation and transaction grouping
- **DBSCAN**: For density-based clustering of suspicious activities
- **Gaussian Mixture Models**: For probabilistic clustering

### 3. Supervised Learning
- **Random Forest**: Ensemble method for fraud classification
- **XGBoost**: Gradient boosting for high performance
- **Logistic Regression**: Baseline interpretable model
- **Neural Networks**: Deep learning approach

### 4. Dynamic Thresholds
- **Moving Averages**: Time-based threshold adjustment
- **Statistical Process Control**: Control charts for monitoring
- **Adaptive Thresholding**: Machine learning-based threshold optimization

## Features for Historical Data Analysis
- Transaction frequency patterns
- Amount distribution changes over time
- Customer behavior evolution
- Seasonal trends in suspicious activities
- Risk score progression

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AML-Fraud-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Download and place datasets in `data/raw/`
2. **Data Preprocessing**: Run preprocessing notebooks/scripts
3. **Model Training**: Execute model training scripts
4. **Evaluation**: Run evaluation scripts to assess performance
5. **Threshold Optimization**: Use dynamic threshold algorithms

## Key Metrics
- Precision, Recall, F1-Score for fraud detection
- False Positive Rate (critical for AML compliance)
- AUC-ROC for model performance
- Anomaly scores distribution
- Threshold effectiveness over time

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
