"""
Main Pipeline Orchestrator for Climate-Resilient Crop Yield Forecasting
ITA0613 - Machine Learning Technical Report Pipeline
Coordinates Data Preprocessing, EDA, k-NN, LWR, Version Space, Scalability, and Visualizations.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Local imports
from src.data_pipeline import run_data_pipeline, AgroDataPipeline
from src.knn import (
    FromScratchKNNRegressor, 
    manual_k_validation_curve, 
    evaluate_regression
)
from src.lwr import FromScratchLWR, manual_tau_validation_curve
from src.version_space import run_version_space_analysis
from src.scalability import benchmark_scalability
from src.visualization import (
    plot_fig1_yield_distribution,
    plot_fig2_rainfall_vs_yield,
    plot_fig3_regional_yield,
    plot_fig4_knn_validation_curve,
    plot_fig5_scalability,
    plot_fig6_model_comparison
)

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'results', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'pipeline_execution.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AgroPipelineOrchestrator')


def run_full_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING CLIMATE-RESILIENT CROP YIELD FORECASTING PIPELINE (ITA0613)")
    logger.info("Aligned to CO2 (Version Spaces), CO6 (Instance-Based), CO7 (NumPy/Pandas)")
    logger.info("=" * 80)
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'crop_yield_dataset.csv')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    table_dir = os.path.join(base_dir, 'results', 'tables')
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Data Pipeline & Feature Engineering (CO7)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 1: Running Data Pipeline, Cleaning, Imputation & Feature Engineering")
    data_dict, pipeline = run_data_pipeline(data_path, split_year=2021)
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save Data Audit Log
    with open(os.path.join(table_dir, 'data_audit_log.json'), 'w') as f:
        json.dump(pipeline.audit_log, f, indent=4)
        
    # -------------------------------------------------------------
    # 2. Exploratory Data Analysis & Visualizations (Figures 1, 2, 3)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 2: Generating Exploratory Visualizations")
    fig1_path = os.path.join(fig_dir, 'fig1_yield_distribution.png')
    fig2_path = os.path.join(fig_dir, 'fig2_rainfall_vs_yield.png')
    fig3_path = os.path.join(fig_dir, 'fig3_regional_yield_variation.png')
    
    plot_fig1_yield_distribution(full_df, fig1_path)
    plot_fig2_rainfall_vs_yield(full_df, fig2_path)
    plot_fig3_regional_yield(full_df, fig3_path)
    
    # -------------------------------------------------------------
    # 3. From-Scratch k-NN Regressor & Manual k Selection (CO6, CO7)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 3: Evaluating From-Scratch k-NN Regressor (Euclidean & Mahalanobis)")
    
    # Split a validation set from training for manual k tuning (20% of train)
    val_size = int(0.2 * len(X_train))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
    
    k_candidates = [3, 5, 7, 9, 11, 15, 21]
    
    # 3.1 Euclidean distance curve
    euclid_results, best_k_euclid = manual_k_validation_curve(
        X_tr, y_tr, X_val, y_val, k_values=k_candidates, metric='euclidean'
    )
    
    # 3.2 Mahalanobis distance curve
    mahal_results, best_k_mahal = manual_k_validation_curve(
        X_tr, y_tr, X_val, y_val, k_values=k_candidates, metric='mahalanobis'
    )
    
    df_knn_val = pd.DataFrame(euclid_results + mahal_results)
    df_knn_val.to_csv(os.path.join(table_dir, 'knn_validation_curve.csv'), index=False)
    
    logger.info(f"Optimal k for Euclidean: {best_k_euclid}")
    logger.info(f"Optimal k for Mahalanobis: {best_k_mahal}")
    
    # Figure 4: k Validation Curve
    fig4_path = os.path.join(fig_dir, 'fig4_knn_k_validation_curve.png')
    plot_fig4_knn_validation_curve(euclid_results, mahal_results, fig4_path)
    
    # Fit final k-NN model on full train set and evaluate on unseen test set
    final_knn = FromScratchKNNRegressor(k=best_k_euclid, metric='euclidean')
    final_knn.fit(X_train, y_train)
    knn_test_preds = final_knn.predict(X_test)
    knn_test_metrics = evaluate_regression(y_test, knn_test_preds)
    
    logger.info(f"k-NN (k={best_k_euclid}) Hold-Out Test Metrics: RMSE={knn_test_metrics['rmse']:.4f}, MAE={knn_test_metrics['mae']:.4f}, R2={knn_test_metrics['r2']:.4f}")
    
    # -------------------------------------------------------------
    # 4. From-Scratch Locally Weighted Regression (LWR) (CO6, CO7)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 4: Evaluating From-Scratch Locally Weighted Regression (LWR)")
    tau_candidates = [0.6, 0.9, 1.2, 1.5, 2.0]
    
    lwr_results, best_tau = manual_tau_validation_curve(
        X_tr, y_tr, X_val, y_val, tau_values=tau_candidates, max_eval_samples=400
    )
    
    df_lwr_val = pd.DataFrame(lwr_results)
    df_lwr_val.to_csv(os.path.join(table_dir, 'lwr_validation_curve.csv'), index=False)
    logger.info(f"Optimal Bandwidth tau for LWR: {best_tau}")
    
    # Fit final LWR model and test on hold-out set
    final_lwr = FromScratchLWR(tau=best_tau, reg_lambda=1e-4)
    final_lwr.fit(X_train, y_train)
    
    # Predict on test set (subsampled if large for fast execution)
    test_eval_size = min(600, len(X_test))
    lwr_test_preds = final_lwr.predict(X_test[:test_eval_size])
    lwr_test_metrics = evaluate_regression(y_test[:test_eval_size], lwr_test_preds)
    
    logger.info(f"LWR (tau={best_tau}) Hold-Out Test Metrics: RMSE={lwr_test_metrics['rmse']:.4f}, MAE={lwr_test_metrics['mae']:.4f}, R2={lwr_test_metrics['r2']:.4f}")
    
    # Figure 6: Model Comparison
    fig6_path = os.path.join(fig_dir, 'fig6_knn_vs_lwr_comparison.png')
    plot_fig6_model_comparison(knn_test_metrics, lwr_test_metrics, fig6_path)
    
    # -------------------------------------------------------------
    # 5. Candidate-Elimination / Version Space Analysis (CO2)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 5: Running Candidate-Elimination / Version Space Learning")
    vs_df = pipeline.prepare_version_space_data(full_df, target_crop='Wheat')
    vs_results = run_version_space_analysis(vs_df, sample_size=40)
    
    with open(os.path.join(table_dir, 'version_space_boundaries.json'), 'w') as f:
        json.dump(vs_results, f, indent=4)
        
    logger.info(f"Version Space Final Specific Boundary (S): {vs_results['specific_boundary']}")
    logger.info(f"Version Space Final General Boundary (G): {vs_results['general_boundary']}")
    
    # -------------------------------------------------------------
    # 6. Scalability Benchmarking (10^3 to 10^5 records) (CO6, CO7)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 6: Running Scalability & Complexity Benchmarks")
    scalability_df = benchmark_scalability(
        scale_sizes=[1000, 5000, 10000, 50000, 100000],
        num_features=X_train.shape[1],
        num_query_repeats=15
    )
    scalability_df.to_csv(os.path.join(table_dir, 'scalability_benchmark.csv'), index=False)
    
    # Figure 5: Scalability plot
    fig5_path = os.path.join(fig_dir, 'fig5_scalability_runtime_memory.png')
    plot_fig5_scalability(scalability_df, fig5_path)
    
    # -------------------------------------------------------------
    # 7. Final Model Comparison Table
    # -------------------------------------------------------------
    comparison_data = [
        {
            'Model': 'k-NN Regressor (Euclidean)',
            'Hyperparameter': f'k = {best_k_euclid}',
            'Hold_Out_RMSE': knn_test_metrics['rmse'],
            'Hold_Out_MAE': knn_test_metrics['mae'],
            'Hold_Out_R2': knn_test_metrics['r2']
        },
        {
            'Model': 'Locally Weighted Regression (LWR)',
            'Hyperparameter': f'tau = {best_tau}',
            'Hold_Out_RMSE': lwr_test_metrics['rmse'],
            'Hold_Out_MAE': lwr_test_metrics['mae'],
            'Hold_Out_R2': lwr_test_metrics['r2']
        }
    ]
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(os.path.join(table_dir, 'model_comparison.csv'), index=False)
    
    # -------------------------------------------------------------
    # Summary Output
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE - SUMMARY OF RESULTS")
    logger.info("=" * 80)
    logger.info("\n[1] DATA PIPELINE AUDIT:")
    logger.info(f"    - Total Records Processed: {pipeline.audit_log['raw_row_count']}")
    logger.info(f"    - Training Split (<= 2021): {pipeline.audit_log['train_size']} records")
    logger.info(f"    - Test Split (>= 2022):     {pipeline.audit_log['test_size']} records")
    logger.info(f"    - Engineered Features:     Rainfall_Anomaly, GDD_Proxy, Nutrient_Index, Climate_Stress_Index")
    
    logger.info("\n[2] MODEL PERFORMANCE (HOLD-OUT TEST):")
    logger.info(df_comparison.to_string(index=False))
    
    logger.info("\n[3] VERSION SPACE BOUNDARIES (CO2):")
    logger.info(f"    - S Boundary: {vs_results['specific_boundary']}")
    logger.info(f"    - G Boundary: {vs_results['general_boundary']}")
    logger.info("    - Risk Band Boundary Profiles:")
    for band, profile in vs_results['risk_band_boundaries'].items():
        logger.info(f"      * {band}: {profile}")
    
    logger.info("\n[4] SCALABILITY BENCHMARK (CO6/CO7):")
    logger.info(scalability_df[['Records_N', 'Memory_MB', 'Query_Latency_ms']].to_string(index=False))
    
    logger.info("\n[5] ARTIFACTS CREATED:")
    logger.info(f"    - Figures in: {fig_dir}")
    logger.info(f"    - Tables in:  {table_dir}")
    logger.info(f"    - Logs in:    {log_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_full_pipeline()
