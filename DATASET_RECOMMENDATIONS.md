# AML Fraud Detection - Dataset Recommendations and Project Setup

## Recommended Open-Source Datasets (~10k records)

### 1. **IEEE-CIS Fraud Detection Dataset** (Primary Recommendation)
- **Source**: Kaggle Competition
- **Size**: 590,540 transactions (easily sampled to 10k)
- **Features**: 433 features including:
  - Transaction amounts and time features
  - Anonymized categorical features
  - Device and browser information
  - Geographic indicators
- **Labels**: Binary fraud classification (0/1)
- **Fraud Rate**: ~3.5%
- **Download**: `https://www.kaggle.com/c/ieee-fraud-detection/data`
- **Highlights**: Real-world data, rich feature set, good for anomaly detection

### 2. **Credit Card Fraud Detection Dataset** (Alternative)
- **Source**: Kaggle / ULB Machine Learning Group
- **Size**: 284,807 transactions
- **Features**: 30 features (28 PCA-transformed + Time + Amount)
- **Labels**: Binary fraud classification
- **Fraud Rate**: 0.172%
- **Download**: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`
- **Highlights**: Highly imbalanced, good for outlier detection algorithms

### 3. **PaySim Financial Dataset** (Backup)
- **Source**: Synthetic but realistic
- **Size**: 6+ million transactions (easily sampled)
- **Features**: Transaction type, amount, origin/destination accounts, balances
- **Labels**: Fraud labels available
- **Download**: `https://www.kaggle.com/datasets/ealaxi/paysim1`
- **Highlights**: Includes money laundering patterns, large dataset

### 4. **Synthetic AML Dataset** (Generated in this project)
- **Source**: Our custom generator (see notebook)
- **Size**: 10,000 transactions (configurable)
- **Features**: 20+ AML-specific features
- **Labels**: Fraud classification with realistic patterns
- **Highlights**: Tailored for AML detection, includes behavioral patterns

## Project Structure Created

```
AML-Fraud-Detection/
├── README.md                          # Project overview and instructions
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── config/
│   └── config.yaml                   # Configuration settings
├── src/                              # Source code
│   ├── __init__.py
│   ├── data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Data loading utilities
│   │   └── preprocessor.py          # Data preprocessing
│   ├── models/                       # ML models (to be created)
│   ├── evaluation/                   # Model evaluation (to be created)
│   └── utils/                        # Utilities (to be created)
└── notebooks/
    └── 01_data_exploration.ipynb     # Data exploration notebook
```

## Key Features for AML Detection

### 1. **Transaction Features**
- Amount patterns and distributions
- Frequency and velocity metrics
- Time-based patterns (hour, day, weekend)
- Cross-border transaction ratios

### 2. **Customer Behavioral Features**
- Historical transaction patterns
- Risk score combinations
- Account age and customer demographics
- Deviation from normal behavior

### 3. **Dynamic Threshold Components**
- Moving averages of transaction amounts
- Statistical process control limits
- Customer-specific baselines
- Adaptive threshold adjustments

## Machine Learning Algorithms Implemented

### 1. **Anomaly Detection** (Unsupervised)
- **Isolation Forest**: Detects outliers in high-dimensional space
- **One-Class SVM**: Learns normal behavior boundary
- **Local Outlier Factor**: Density-based anomaly detection
- **Autoencoders**: Neural network-based pattern recognition

### 2. **Clustering Analysis**
- **K-Means**: Customer segmentation and transaction grouping
- **DBSCAN**: Density-based clustering for suspicious patterns
- **Gaussian Mixture Models**: Probabilistic clustering

### 3. **Supervised Learning**
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: High-performance gradient boosting
- **Logistic Regression**: Interpretable baseline model
- **Neural Networks**: Deep learning approach

### 4. **Dynamic Thresholds**
- Historical pattern analysis
- Moving window calculations
- Statistical process control
- Adaptive learning algorithms

## Next Steps

### 1. **Download Real Dataset**
```bash
# Install Kaggle API
pip install kaggle

# Download IEEE-CIS dataset (requires Kaggle account)
kaggle competitions download -c ieee-fraud-detection

# Or use credit card fraud dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
```

### 2. **Set Up Environment**
```bash
cd AML-Fraud-Detection
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 3. **Run the Notebook**
```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

### 4. **Implement Models**
- Complete the notebook sections for:
  - Anomaly detection implementation
  - Clustering analysis
  - Supervised learning models
  - Dynamic threshold calculation
  - Model evaluation and comparison

## Performance Metrics for AML

### 1. **Standard Metrics**
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix

### 2. **AML-Specific Metrics**
- False Positive Rate (critical for compliance)
- Detection Rate at specific thresholds
- Time to detection
- Threshold stability over time

### 3. **Business Metrics**
- Cost of false positives
- Regulatory compliance scores
- Operational efficiency

## Configuration Features

The `config.yaml` file includes settings for:
- Data preprocessing parameters
- Model hyperparameters
- Dynamic threshold settings
- Evaluation metrics
- Monitoring and alerting

## Ready to Use

Your project is now set up with:
1. ✅ Synthetic dataset generation (10k records)
2. ✅ Feature engineering pipeline
3. ✅ Data exploration framework
4. ✅ Configuration management
5. ✅ Extensible model structure

You can start working with the synthetic data immediately or download one of the recommended real datasets for production use.
