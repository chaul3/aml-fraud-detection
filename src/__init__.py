"""
AML Fraud Detection Package

This package provides tools for detecting anti-money laundering (AML) fraud
using dynamic thresholds and multiple machine learning algorithms.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data import DataLoader, Preprocessor
from .models import (
    AnomalyDetector,
    ClusteringModel,
    SupervisedModel,
    DynamicThresholds
)
from .evaluation import ModelEvaluator
from .utils import ConfigLoader, Logger

__all__ = [
    "DataLoader",
    "Preprocessor", 
    "AnomalyDetector",
    "ClusteringModel",
    "SupervisedModel",
    "DynamicThresholds",
    "ModelEvaluator",
    "ConfigLoader",
    "Logger"
]
