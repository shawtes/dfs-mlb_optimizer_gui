import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class EnhancedMLBFinancialStyleEngine:
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28, 45]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        
        # --- Preprocessing ---
        # Ensure date is datetime and sort
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(['Name', date_col])

        # Standardize opportunity columns
        if 'PA' not in df.columns and 'PA.1' in df.columns:
            df['PA'] = df['PA.1']
        if 'AB' not in df.columns and 'AB.1' in df.columns:
            df['AB'] = df['AB.1']
            
        # Ensure base columns exist
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                print(f"Warning: Column '{col}' not found. Initialized with 0.")

        # Group by player
        all_players_data = []
        for name, group in df.groupby('Name'):
            group = group.copy()
            
            # --- Momentum Features (like RSI, MACD) ---
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    # Rolling means (SMA)
                    group[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    # Exponential rolling means (EMA)
                    group[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    # Rate of Change (Momentum)
                    group[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                # Performance vs moving average
                group[f'{col}_vs_sma_28'] = (group[col] / group[f'{col}_sma_28']) - 1
            
            # --- Volatility Features (like Bollinger Bands) ---
            for window in self.rolling_windows:
                mean = group['dk_fpts'].rolling(window).mean()
                std = group['dk_fpts'].rolling(window).std()
                group[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                group[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                group[f'dk_fpts_band_width_{window}'] = (group[f'dk_fpts_upper_band_{window}'] - group[f'dk_fpts_lower_band_{window}']) / mean
                group[f'dk_fpts_band_position_{window}'] = (group['dk_fpts'] - group[f'dk_fpts_lower_band_{window}']) / (group[f'dk_fpts_upper_band_{window}'] - group[f'dk_fpts_lower_band_{window}'])

            # --- "Volume" (PA/AB) based Features ---
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    group[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    group[f'{vol_col}_ratio'] = group[vol_col] / group[f'{vol_col}_roll_mean_28']
                    group[f'dk_fpts_{vol_col}_corr_28'] = group['dk_fpts'].rolling(28).corr(group[vol_col])

            # --- Interaction / Ratio Features ---
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns:
                    group[f'{col}_per_pa'] = group[col] / group['PA']
            
            # --- Temporal Features ---
            group['day_of_week'] = group[date_col].dt.dayofweek
            group['month'] = group[date_col].dt.month
            group['is_weekend'] = (group['day_of_week'] >= 5).astype(int)
            group['day_of_week_sin'] = np.sin(2 * np.pi * group['day_of_week'] / 7)
            group['day_of_week_cos'] = np.cos(2 * np.pi * group['day_of_week'] / 7)

            all_players_data.append(group)
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        # Final cleanup
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.fillna(method='ffill')
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df

def main():
    input_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/merged_fangraphs_logs_with_fpts.csv'
    output_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/mlb_financial_style_features.csv'
    
    print("Loading data...")
    df = pd.read_csv(input_csv, low_memory=False)
    
    print("Starting feature engineering...")
    engine = EnhancedMLBFinancialStyleEngine()
    enhanced_df = engine.calculate_features(df)
    
    print(f"Saving file with {enhanced_df.shape[1]} columns...")
    enhanced_df.to_csv(output_csv, index=False)
    print(f"Enhanced MLB features with financial-style patterns saved to {output_csv}")

if __name__ == "__main__":
    main() 