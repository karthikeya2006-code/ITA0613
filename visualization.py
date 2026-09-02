"""
Publication-Quality Visualizations for Climate-Resilient Crop Yield Forecasting
Generates 6 figures aligned with technical report figures:
1. Yield Distribution
2. Rainfall vs. Yield Relationship
3. Regional Yield Variation
4. Manual k-NN Validation Curve
5. Scalability & Memory Growth
6. Model Hold-Out Error Comparison (k-NN vs LWR)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Set modern publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'


def plot_fig1_yield_distribution(df: pd.DataFrame, output_path: str):
    """Figure 1: Yield distribution across major crops."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # Filter out extreme outliers for clean visualization
    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Pulses', 'Millets', 'Groundnut']
    sub_df = df[df['Crop'].isin(crops)]
    
    palette = sns.color_palette("mako", n_colors=len(crops))
    sns.boxplot(
        data=sub_df, 
        x='Crop', 
        y='Yield_tonnes_per_ha', 
        hue='Crop',
        palette=palette, 
        ax=ax, 
        width=0.6,
        fliersize=2,
        linewidth=1.2,
        legend=False
    )
    
    ax.set_title('Figure 1: Distribution of Crop Yield Across Major Crop Types (tonnes/ha)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Crop Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Yield (Tonnes per Hectare)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig2_rainfall_vs_yield(df: pd.DataFrame, output_path: str):
    """Figure 2: Relationship between annual rainfall and crop yield."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    # Filter for representative cereal crops
    sub_df = df[df['Crop'].isin(['Wheat', 'Rice', 'Maize'])].sample(n=min(2000, len(df)), random_state=42)
    
    scatter = ax.scatter(
        sub_df['Annual_Rainfall_mm'],
        sub_df['Yield_tonnes_per_ha'],
        c=sub_df['Climate_Stress_Index'],
        cmap='viridis',
        alpha=0.65,
        edgecolors='none',
        s=30
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Climate Stress Index', fontsize=10, fontweight='bold')
    
    ax.set_title('Figure 2: Annual Rainfall vs. Crop Yield under Climate Stress Gradient', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Annual Rainfall (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Crop Yield (Tonnes / ha)', fontsize=11, fontweight='bold')
    ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig3_regional_yield(df: pd.DataFrame, output_path: str):
    """Figure 3: Regional variation in mean crop yield."""
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    
    state_means = df.groupby('State')['Yield_tonnes_per_ha'].agg(['mean', 'std']).reset_index()
    state_means = state_means.sort_values(by='mean', ascending=False)
    
    bars = ax.bar(
        state_means['State'], 
        state_means['mean'], 
        yerr=state_means['std'], 
        capsize=5, 
        color='#2b5c8f', 
        edgecolor='#1a365d', 
        alpha=0.85
    )
    
    ax.set_title('Figure 3: Regional Variation in Mean Crop Yield by State', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('State', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Yield (Tonnes / ha)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig4_knn_validation_curve(
    euclid_results: List[Dict[str, Any]], 
    mahal_results: List[Dict[str, Any]], 
    output_path: str
):
    """Figure 4: Manual validation curve for neighbourhood parameter k."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    k_vals = [r['k'] for r in euclid_results]
    euclid_rmse = [r['val_rmse'] for r in euclid_results]
    mahal_rmse = [r['val_rmse'] for r in mahal_results]
    
    ax.plot(k_vals, euclid_rmse, marker='o', linewidth=2.2, color='#1f77b4', label='k-NN (Euclidean Distance)')
    ax.plot(k_vals, mahal_rmse, marker='s', linewidth=2.2, linestyle='--', color='#ff7f0e', label='k-NN (Mahalanobis Distance)')
    
    # Highlight optimal k
    min_idx = np.argmin(euclid_rmse)
    ax.scatter(k_vals[min_idx], euclid_rmse[min_idx], color='red', s=90, zorder=5, label=f'Optimal k={k_vals[min_idx]}')
    
    ax.set_title('Figure 4: Manual k-NN Validation Curve across Neighbourhood Sizes (k)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Neighbourhood Size (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation RMSE (Tonnes / ha)', fontsize=11, fontweight='bold')
    ax.set_xticks(k_vals)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='#f8f9fa', framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig5_scalability(scalability_df: pd.DataFrame, output_path: str):
    """Figure 5: Observed computational scaling of single distance query and memory."""
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(
        scalability_df['Records_N'], 
        scalability_df['Query_Latency_ms'], 
        marker='o', 
        color='#d62728', 
        linewidth=2.2, 
        label='Query Latency (ms)'
    )
    
    line2 = ax2.plot(
        scalability_df['Records_N'], 
        scalability_df['Memory_MB'], 
        marker='^', 
        color='#2ca02c', 
        linewidth=2.2, 
        linestyle=':', 
        label='Feature Matrix Memory (MB)'
    )
    
    ax1.set_title('Figure 5: Computational Scaling and Memory Footprint vs. Dataset Size (N)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Number of Training Records (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Single Query Distance Time (ms)', color='#d62728', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Feature Matrix Memory (MB)', color='#2ca02c', fontsize=11, fontweight='bold')
    
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.grid(True)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='#f8f9fa')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig6_model_comparison(knn_metrics: Dict[str, float], lwr_metrics: Dict[str, float], output_path: str):
    """Figure 6: Demonstration hold-out error comparison between k-NN and LWR."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    metrics = ['RMSE', 'MAE', 'R2']
    knn_vals = [knn_metrics['rmse'], knn_metrics['mae'], knn_metrics['r2']]
    lwr_vals = [lwr_metrics['rmse'], lwr_metrics['mae'], lwr_metrics['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, knn_vals, width, label='k-NN (Optimal k)', color='#1f77b4', alpha=0.85)
    rects2 = ax.bar(x + width/2, lwr_vals, width, label='LWR (Optimal tau)', color='#2ca02c', alpha=0.85)
    
    ax.set_title('Figure 6: Hold-Out Test Error Comparison (k-NN vs. Locally Weighted Regression)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE (tonnes/ha)', 'MAE (tonnes/ha)', 'R² Score'], fontsize=10, fontweight='bold')
    ax.grid(True, axis='y')
    ax.legend(frameon=True, facecolor='#f8f9fa')
    
    # Add numerical value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
                    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
