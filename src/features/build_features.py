import pandas as pd
import numpy as np

def extract_rolling_features(df: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
    """
    Extracts rolling statistical features (mean, std, skew, kurtosis) 
    over a sliding window across 3-phase current telemetry.
    
    :param df: Input DataFrame with telemetry (current_phase_a, b, c)
    :param window_size: Number of time samples per sliding window (default: 50)
    :return: DataFrame enriched with rolling features (NaN rows dropped)
    """
    df_feat = df.copy()
    phase_cols = ['current_phase_a', 'current_phase_b', 'current_phase_c']
    
    for col in phase_cols:
        # Calculate rolling statistics across the sliding time window
        df_feat[f'{col}_roll_mean'] = df_feat[col].rolling(window=window_size).mean()
        df_feat[f'{col}_roll_std'] = df_feat[col].rolling(window=window_size).std()
        df_feat[f'{col}_roll_skew'] = df_feat[col].rolling(window=window_size).skew()
        df_feat[f'{col}_roll_kurt'] = df_feat[col].rolling(window=window_size).kurt()
    
    # Drop initial rows containing NaN values during window warm-up
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    return df_feat

if __name__ == "__main__":
    from src.data.synthetic_generator import generate_base_telemetry, inject_anomalies
    
    raw_df = generate_base_telemetry()
    fault_df = inject_anomalies(raw_df)
    featured_df = extract_rolling_features(fault_df)
    
    print("Raw Data Shape:", fault_df.shape)
    print("Featured Data Shape:", featured_df.shape)
    print("\nExtracted Feature Columns:")
    print([c for c in featured_df.columns if 'roll' in c])