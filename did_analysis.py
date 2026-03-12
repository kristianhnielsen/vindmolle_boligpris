"""
Causal Analysis: Wind Turbine Impact on House Prices

Analyzes the relationship between wind turbine proximity and house price growth.
Uses linear regression with appropriate controls to isolate causal effects.

Model: price_change ~ has_new_turbine + inverse_dist + visual_prominence + controls
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent / "models"))

from data_handler import DataHandler


def run_causal_analysis():
    """Run causal analysis of wind turbine impact on house prices."""
    print("="*80)
    print("WIND TURBINE IMPACT ANALYSIS: Causal Coefficients")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    dh = DataHandler()
    data = dh.data_next.copy()
    
    if hasattr(data, 'geometry'):
        data = pd.DataFrame(data.drop(columns=['geometry']))
    
    # Clean data
    data = data.dropna(subset=[
        'price_change', 'has_new_turbine', 'inverse_dist_to_new_turbine',
        'visual_prominence', 'SamletKoebesum_prev', 'years_diff'
    ])
    
    print(f"Dataset size: {len(data)} property sales pairs")
    print(f"  Treatment group (near turbine): {(data['has_new_turbine'] == 1).sum()} ({100*(data['has_new_turbine'] == 1).sum()/len(data):.1f}%)")
    print(f"  Control group (no turbine):     {(data['has_new_turbine'] == 0).sum()} ({100*(data['has_new_turbine'] == 0).sum()/len(data):.1f}%)")
    
    # Prepare features
    X_cols = ['has_new_turbine', 'inverse_dist_to_new_turbine', 'visual_prominence',
               'SamletKoebesum_prev', 'years_diff', 'dist_to_center']
    
    X = data[X_cols].values
    y = data['price_change'].values
    
    # Standardize features for interpretability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    print("\nFitting linear regression model...")
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    # Calculate statistics
    n = len(y)
    k = X_scaled.shape[1]
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    # Standard errors
    residuals = y - y_pred
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / (n - k - 1))
    var_covar_matrix = residual_std_error ** 2 * np.linalg.inv(X_scaled.T @ X_scaled)
    std_errors = np.sqrt(np.diag(var_covar_matrix))
    t_stats = model.coef_ / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))
    
    # Print results
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80)
    print(f"{'Feature':<40} {'Coefficient':>12} {'Std.Err':>12} {'t-stat':>10} {'p-value':>10}")
    print("-"*80)
    
    for i, col in enumerate(X_cols):
        coef = model.coef_[i]
        se = std_errors[i]
        t = t_stats[i]
        p = p_values[i]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        
        print(f"{col:<40} {coef:>12,.0f} {se:>12,.0f} {t:>10.3f} {p:>10.4f} {sig}")
    
    print("-"*80)
    print(f"{'Intercept':<40} {model.intercept_:>12,.0f}")
    print(f"\nR² = {r2:.4f}")
    print(f"Adj R² = {adj_r2:.4f}")
    print(f"RMSE = {rmse:,.0f} DKK")
    print(f"N = {n}")
    
    # Interpret the turbine effect
    print("\n" + "="*80)
    print("INTERPRETATION: WIND TURBINE IMPACT")
    print("="*80)
    
    turbine_idx = X_cols.index('has_new_turbine')
    turbine_coef = model.coef_[turbine_idx]
    turbine_se = std_errors[turbine_idx]
    turbine_p = p_values[turbine_idx]
    turbine_t = t_stats[turbine_idx]
    
    print(f"\n📊 TREATMENT EFFECT (Being near a wind turbine):")
    print(f"   Coefficient: {turbine_coef:,.0f} DKK")
    print(f"   Std. Error:  {turbine_se:,.0f} DKK")
    print(f"   t-statistic: {turbine_t:.3f}")
    print(f"   p-value:     {turbine_p:.4f}")
    
    if turbine_p < 0.05:
        direction = "REDUCES" if turbine_coef < 0 else "INCREASES"
        print(f"\n   ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"\n   Being within sight of a wind turbine {direction} price appreciation by")
        print(f"   approximately {abs(turbine_coef):,.0f} DKK on average,")
        print(f"   compared to properties far from turbines.")
    else:
        print(f"\n   ❌ NOT STATISTICALLY SIGNIFICANT (p = {turbine_p:.4f})")
        print(f"\n   The estimated turbine effect ({turbine_coef:,.0f} DKK) is not")
        print(f"   statistically distinguishable from zero.")
    
    # Distance effects
    print(f"\n📏 DISTANCE EFFECTS (inverse_dist_to_new_turbine):")
    dist_idx = X_cols.index('inverse_dist_to_new_turbine')
    dist_coef = model.coef_[dist_idx]
    dist_p = p_values[dist_idx]
    
    if dist_p < 0.05:
        print(f"   Coefficient: {dist_coef:>12,.0f} DKK (SIGNIFICANT, p={dist_p:.4f})")
        if dist_coef < 0:
            print(f"   → Closer distance REDUCES price appreciation")
        else:
            print(f"   → Closer distance INCREASES price appreciation")
    else:
        print(f"   Coefficient: {dist_coef:>12,.0f} DKK (NOT significant)")
    
    # Visual prominence
    print(f"\n👁️  VISUAL PROMINENCE (height/distance):")
    vis_idx = X_cols.index('visual_prominence')
    vis_coef = model.coef_[vis_idx]
    vis_p = p_values[vis_idx]
    
    if vis_p < 0.05:
        print(f"   Coefficient: {vis_coef:>12,.0f} DKK (SIGNIFICANT, p={vis_p:.4f})")
        if vis_coef < 0:
            print(f"   → Higher visual prominence REDUCES price appreciation")
        else:
            print(f"   → Higher visual prominence INCREASES price appreciation")
    else:
        print(f"   Coefficient: {vis_coef:>12,.0f} DKK (NOT significant)")
    
    print("\n" + "="*80)
    print("Summary: Negative coefficients on turbine features suggest that")
    print("proximity to wind turbines reduces house price growth.")
    print("="*80)
    
    # Group statistics
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS BY GROUP")
    print("="*80)
    
    for group_name, group_val in [("Control (No Turbine)", 0), ("Treatment (Near Turbine)", 1)]:
        group_data = data[data['has_new_turbine'] == group_val]['price_change']
        print(f"\n{group_name}:")
        print(f"  N:    {len(group_data)}")
        print(f"  Mean: {group_data.mean():>12,.0f} DKK")
        print(f"  Std:  {group_data.std():>12,.0f} DKK")
        print(f"  Median: {group_data.median():>10,.0f} DKK")
        print(f"  Min:  {group_data.min():>12,.0f} DKK")
        print(f"  Max:  {group_data.max():>12,.0f} DKK")


if __name__ == "__main__":
    run_causal_analysis()
