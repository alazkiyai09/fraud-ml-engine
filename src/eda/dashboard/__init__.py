"""
Fraud Detection EDA Dashboard

A production-grade interactive dashboard for exploratory data analysis
of credit card fraud detection datasets.

Author: [Your Name]
Portfolio: AI/ML for Financial Services
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.eda.dashboard.data_loader import load_fraud_data, preprocess_data
from src.eda.dashboard.utils import FRAUD_COLOR, LEGIT_COLOR

__all__ = [
    "load_fraud_data",
    "preprocess_data",
    "FRAUD_COLOR",
    "LEGIT_COLOR",
]
