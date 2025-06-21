import pandas as pd
import numpy as np

class EnhancedMLBFeatureEngine:
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        # Ensure date is datetime
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values(['Name', 'game_date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['Name', 'date'])
        else:
            raise ValueError('No date column found!')

        # Fill missing stat columns with 0
        for col in self.stat_cols:
            if col not in df.columns:
                df[col] = 0

        # Rolling/statistical features per player
        feature_frames = []
        for name, group in df.groupby('Name'):
            group = group.copy()
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    group[f'{col}_roll_mean_{window}'] = group[col].rolling(window).mean()
                    group[f'{col}_roll_std_{window}'] = group[col].rolling(window).std()
                    group[f'{col}_roll_min_{window}'] = group[col].rolling(window).min()
                    group[f'{col}_roll_max_{window}'] = group[col].rolling(window).max()
                    group[f'{col}_roll_sum_{window}'] = group[col].rolling(window).sum()
                    group[f'{col}_roll_median_{window}'] = group[col].rolling(window).median()
                    group[f'{col}_roll_skew_{window}'] = group[col].rolling(window).skew()
                    group[f'{col}_roll_kurt_{window}'] = group[col].rolling(window).kurt()
                    group[f'{col}_zscore_{window}'] = (group[col] - group[col].rolling(window).mean()) / group[col].rolling(window).std()
                # Lag features
                for lag in [1, 2, 3, 7]:
                    group[f'{col}_lag_{lag}'] = group[col].shift(lag)
                # Cumulative
                group[f'{col}_cumsum'] = group[col].cumsum()
            feature_frames.append(group)
        enhanced_df = pd.concat(feature_frames, ignore_index=True)
        return enhanced_df

def main():
    input_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/merged_fangraphs_logs_with_fpts.csv'
    output_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/merged_fangraphs_logs_with_enhanced_features.csv'
    df = pd.read_csv(input_csv)
    engine = EnhancedMLBFeatureEngine()
    enhanced_df = engine.calculate_features(df)
    enhanced_df.to_csv(output_csv, index=False)
    print(f"Enhanced MLB features saved to {output_csv}")

if __name__ == "__main__":
    main() 