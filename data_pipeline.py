"""
Data Pipeline for Climate-Resilient Crop Yield Forecasting
Handles loading, cleaning, missing value imputation, feature engineering,
temporal splitting, scaling, and version space discretization using NumPy and Pandas.
Aligned with Course Outcomes CO6, CO7.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger('DataPipeline')


class CustomStandardScaler:
    """
    Pure NumPy implementation of standard z-score normalization:
    z = (x - mean) / (std + eps)
    Fitted exclusively on training data to prevent future information leakage.
    """
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.scale_ = None
        self.eps = eps
        self.feature_names = None

    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'CustomStandardScaler':
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        self.feature_names = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("CustomStandardScaler must be fitted before transforming data.")
        return (X - self.mean_) / (self.scale_ + self.eps)

    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("CustomStandardScaler must be fitted before inverse transform.")
        return (X_scaled * (self.scale_ + self.eps)) + self.mean_


class AgroDataPipeline:
    """
    End-to-end data preparation, cleaning, feature engineering, and splitting pipeline.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.audit_log: Dict[str, Any] = {}
        self.imputation_values: Dict[str, float] = {}
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.scaler = CustomStandardScaler()
        self.feature_cols: List[str] = []
        self.target_col: str = 'Yield_tonnes_per_ha'

    def load_and_audit(self) -> pd.DataFrame:
        """Load dataset and perform initial structural audit."""
        logger.info(f"Loading raw dataset from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        self.audit_log['raw_row_count'] = int(len(df))
        self.audit_log['raw_col_count'] = int(df.shape[1])
        self.audit_log['initial_null_counts'] = df.isnull().sum().to_dict()
        
        logger.info(f"Raw data shape: {df.shape}. Initial nulls: {self.audit_log['initial_null_counts']}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, trim invalid bounds, and log removed records."""
        initial_len = len(df)
        
        # Deduplication
        df = df.drop_duplicates(subset=['Year', 'State', 'District', 'Crop', 'Season']).copy()
        dup_removed = initial_len - len(df)
        
        # Outlier / Invalid target filtering
        valid_mask = (df[self.target_col] > 0.0) & (df[self.target_col] < 200.0)
        invalid_target_count = int((~valid_mask).sum())
        df = df[valid_mask].copy()
        
        self.audit_log['duplicates_removed'] = dup_removed
        self.audit_log['invalid_target_rows_removed'] = invalid_target_count
        self.audit_log['cleaned_row_count'] = int(len(df))
        
        logger.info(f"Cleaned data: removed {dup_removed} duplicates, {invalid_target_count} invalid records. Retained: {len(df)}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive domain-specific agro-climatic indicators using Pandas & NumPy:
        1. Rainfall Anomaly (R_t - regional mean)
        2. Growing Degree Days (GDD) Proxy
        3. Soil Nutrient Composite Index
        4. Compound Climate Stress Index
        """
        df = df.copy()
        
        # 1. Regional Rainfall Baseline and Anomaly
        regional_rain_mean = df.groupby('State')['Annual_Rainfall_mm'].transform('mean')
        df['Rainfall_Anomaly_mm'] = df['Annual_Rainfall_mm'] - regional_rain_mean
        
        # 2. GDD (Growing Degree Day) Proxy: thermal sum above 10 deg C base temperature
        # Standard growing cycle approximated as 120 days
        base_temp = 10.0
        df['GDD_Proxy'] = np.maximum(df['Avg_Temperature_C'] - base_temp, 0.0) * 120.0
        
        # 3. Soil Nutrient Composite Index: Mean of normalized N, P, K ratios
        # Normalization references (standard optimums: N ~ 250, P ~ 35, K ~ 200 kg/ha)
        n_norm = df['Soil_N_kg_per_ha'] / 250.0
        p_norm = df['Soil_P_kg_per_ha'] / 35.0
        k_norm = df['Soil_K_kg_per_ha'] / 200.0
        df['Nutrient_Index'] = (n_norm + p_norm + k_norm) / 3.0
        
        # 4. Compound Climate Stress Index (Standardized rainfall deficit + thermal deviation)
        rain_z = (df['Rainfall_Anomaly_mm'] - df['Rainfall_Anomaly_mm'].mean()) / (df['Rainfall_Anomaly_mm'].std() + 1e-8)
        temp_dev = np.abs(df['Avg_Temperature_C'] - df.groupby('State')['Avg_Temperature_C'].transform('mean'))
        temp_z = (temp_dev - temp_dev.mean()) / (temp_dev.std() + 1e-8)
        df['Climate_Stress_Index'] = 0.5 * (-rain_z) + 0.5 * temp_z
        
        logger.info("Feature engineering complete: Added Rainfall_Anomaly_mm, GDD_Proxy, Nutrient_Index, Climate_Stress_Index")
        return df

    def temporal_split(self, df: pd.DataFrame, split_year: int = 2021) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal Train/Test partition to simulate real forecasting conditions without data leakage.
        Train: Year <= split_year (e.g., 2010–2021)
        Test:  Year > split_year  (e.g., 2022–2024)
        """
        train_df = df[df['Year'] <= split_year].copy()
        test_df = df[df['Year'] > split_year].copy()
        
        self.audit_log['train_size'] = int(len(train_df))
        self.audit_log['test_size'] = int(len(test_df))
        self.audit_log['split_year'] = split_year
        
        logger.info(f"Temporal Split at Year {split_year}: Train records = {len(train_df)}, Test records = {len(test_df)}")
        return train_df, test_df

    def impute_missing_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute missing values using training-set medians strictly computed from train_df.
        """
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        numeric_cols = [
            'Annual_Rainfall_mm', 'Avg_Temperature_C', 'Humidity_Percent',
            'Soil_N_kg_per_ha', 'Soil_P_kg_per_ha', 'Soil_K_kg_per_ha',
            'Soil_pH', 'Fertilizer_Usage_kg_per_ha', 'Pesticide_Usage_kg_per_ha',
            'Rainfall_Anomaly_mm', 'GDD_Proxy', 'Nutrient_Index', 'Climate_Stress_Index'
        ]
        
        for col in numeric_cols:
            if col in train_df.columns:
                median_val = float(train_df[col].median())
                self.imputation_values[col] = median_val
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                
        logger.info(f"Imputation completed for {len(self.imputation_values)} continuous features using training medians.")
        return train_df, test_df

    def encode_and_scale(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        One-hot encode categorical features and scale numeric features using CustomStandardScaler.
        Returns prepared NumPy matrices and feature metadata.
        """
        cat_cols = ['State', 'Crop', 'Season']
        num_cols = [
            'Area_Hectares', 'Annual_Rainfall_mm', 'Avg_Temperature_C', 'Humidity_Percent',
            'Soil_N_kg_per_ha', 'Soil_P_kg_per_ha', 'Soil_K_kg_per_ha', 'Soil_pH',
            'Fertilizer_Usage_kg_per_ha', 'Pesticide_Usage_kg_per_ha',
            'Rainfall_Anomaly_mm', 'GDD_Proxy', 'Nutrient_Index', 'Climate_Stress_Index'
        ]
        
        # Fit one-hot categories on training data
        train_encoded = pd.get_dummies(train_df[cat_cols], drop_first=True)
        test_encoded = pd.get_dummies(test_df[cat_cols], drop_first=True)
        
        # Align test columns with train columns
        test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
        
        # Continuous feature scaling
        X_train_num = train_df[num_cols].values.astype(np.float64)
        X_test_num = test_df[num_cols].values.astype(np.float64)
        
        self.scaler.fit(X_train_num, feature_names=num_cols)
        X_train_num_scaled = self.scaler.transform(X_train_num)
        X_test_num_scaled = self.scaler.transform(X_test_num)
        
        # Combine scaled numeric and encoded categorical features
        X_train = np.hstack([X_train_num_scaled, train_encoded.values.astype(np.float64)])
        X_test = np.hstack([X_test_num_scaled, test_encoded.values.astype(np.float64)])
        
        y_train = train_df[self.target_col].values.astype(np.float64)
        y_test = test_df[self.target_col].values.astype(np.float64)
        
        all_feature_names = num_cols + list(train_encoded.columns)
        self.feature_cols = all_feature_names
        
        logger.info(f"Transformed Feature Matrix: X_train {X_train.shape}, X_test {X_test.shape} with {len(all_feature_names)} features.")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_num_unscaled': X_train_num,
            'X_test_num_unscaled': X_test_num,
            'feature_names': all_feature_names,
            'num_feature_names': num_cols,
            'train_df': train_df,
            'test_df': test_df,
            'scaler': self.scaler
        }

    def prepare_version_space_data(self, df: pd.DataFrame, target_crop: str = 'Wheat') -> pd.DataFrame:
        """
        Discretize continuous climate, soil, and yield variables for Candidate-Elimination / Version Space learning (CO2).
        Converts instances into discrete attribute bins:
        - Rainfall_Bin: ['Low', 'Med', 'High']
        - Temp_Bin: ['Low', 'Med', 'High']
        - Humidity_Bin: ['Low', 'Med', 'High']
        - Soil_pH_Bin: ['Acidic', 'Neutral', 'Alkaline']
        - Nutrient_Bin: ['Low', 'Med', 'High']
        - Yield_Risk_Band: ['Low_Risk' (High Yield), 'Medium_Risk', 'High_Risk' (Low Yield)]
        """
        sub_df = df[df['Crop'] == target_crop].copy() if target_crop in df['Crop'].values else df.copy()
        
        vs_df = pd.DataFrame()
        
        # 1. Rainfall Binning
        vs_df['Rainfall_Bin'] = pd.qcut(sub_df['Annual_Rainfall_mm'], q=3, labels=['Low', 'Med', 'High']).astype(str)
        
        # 2. Temperature Binning
        vs_df['Temp_Bin'] = pd.qcut(sub_df['Avg_Temperature_C'], q=3, labels=['Low', 'Med', 'High']).astype(str)
        
        # 3. Humidity Binning
        vs_df['Humidity_Bin'] = pd.qcut(sub_df['Humidity_Percent'], q=3, labels=['Low', 'Med', 'High']).astype(str)
        
        # 4. Soil pH Binning: [< 6.2: Acidic, 6.2 - 7.5: Neutral, > 7.5: Alkaline]
        vs_df['Soil_pH_Bin'] = pd.cut(
            sub_df['Soil_pH'], 
            bins=[-np.inf, 6.2, 7.5, np.inf], 
            labels=['Acidic', 'Neutral', 'Alkaline']
        ).astype(str)
        
        # 5. Nutrient Binning
        vs_df['Nutrient_Bin'] = pd.qcut(sub_df['Nutrient_Index'], q=3, labels=['Low', 'Med', 'High']).astype(str)
        
        # 6. Yield Risk Target: High Yield = Low Risk (Positive class for high resilience)
        yield_33 = sub_df[self.target_col].quantile(0.33)
        yield_66 = sub_df[self.target_col].quantile(0.66)
        
        def assign_risk_band(y):
            if y >= yield_66:
                return 'Low_Risk'      # High Yield
            elif y >= yield_33:
                return 'Medium_Risk'   # Moderate Yield
            else:
                return 'High_Risk'     # Low Yield / Crop Vulnerability
                
        vs_df['Yield_Risk_Band'] = sub_df[self.target_col].apply(assign_risk_band)
        vs_df['High_Resilience'] = (vs_df['Yield_Risk_Band'] == 'Low_Risk').map({True: 'Yes', False: 'No'})
        
        return vs_df


def run_data_pipeline(data_path: str, split_year: int = 2021) -> Tuple[Dict[str, Any], AgroDataPipeline]:
    """Convenience runner for data pipeline."""
    pipeline = AgroDataPipeline(data_path)
    df_raw = pipeline.load_and_audit()
    df_clean = pipeline.clean_data(df_raw)
    df_eng = pipeline.engineer_features(df_clean)
    train_df, test_df = pipeline.temporal_split(df_eng, split_year=split_year)
    train_imp, test_imp = pipeline.impute_missing_values(train_df, test_df)
    dataset_dict = pipeline.encode_and_scale(train_imp, test_imp)
    return dataset_dict, pipeline
