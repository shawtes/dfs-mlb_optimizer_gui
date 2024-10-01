import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load pre-trained model
base_path = '/Users/sineshawmesfintesfaye/newenv/'
model_path = os.path.join(base_path, 'dk_fpts_model.keras')
scaler_path = os.path.join(base_path, 'scaler.pkl')
feature_list_path = os.path.join(base_path, 'selected_features.pkl')

model = load_model(model_path)
scaler = joblib.load(scaler_path)
selected_features = joblib.load(feature_list_path)

print(f"Loaded selected features: {selected_features}")

# Load encoders
name_encoder_path = os.path.join(base_path, 'label_encoder_name.pkl')
team_encoder_path = os.path.join(base_path, 'label_encoder_team.pkl')

le_name = joblib.load(name_encoder_path)
le_team = joblib.load(team_encoder_path)

# League averages and weights (unchanged)
league_avg_wOBA = {
    2020: 0.320, 2021: 0.318, 2022: 0.317, 2023: 0.316, 2024: 0.315
}
league_avg_HR_FlyBall = {
    2020: 0.145, 2021: 0.144, 2022: 0.143, 2023: 0.142, 2024: 0.141
}
wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

def extend_label_encoder(le, new_labels):
    current_classes = set(le.classes_)
    new_classes = set(new_labels) - current_classes
    if new_classes:
        le.classes_ = np.concatenate([le.classes_, list(new_classes)])
# Add teammate interaction terms
def add_teammate_interactions(df):
    df = df.sort_values(by=['Team', 'Name', 'date'])
    team_stats = df.groupby(['Team', 'date']).agg({
        'calculated_dk_fpts': 'sum', 'AB': 'sum', 'HR': 'sum', 'H': 'sum'
    }).reset_index()
    df = pd.merge(df, team_stats, on=['Team', 'date'], suffixes=('', '_team'))
    df['teammate_dk_fpts'] = df['calculated_dk_fpts_team'] - df['calculated_dk_fpts']
    return df
# Load encoders
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_name_1_sep5.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_team_1_sep5.pkl'

le_name = joblib.load(name_encoder_path)
le_team = joblib.load(team_encoder_path)

def engineer_features(df):
    # Extend label encoders
    extend_label_encoder(le_name, df['Name'].unique())
    extend_label_encoder(le_team, df['Team'].unique())
    extend_label_encoder(le_team, df['Opponent'].unique())

    # Now encode
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    df['Opponent_encoded'] = le_team.transform(df['Opponent'])

    print("Columns after encoding:", df.columns.tolist())

    # Your existing feature engineering code...
    df['year'] = df['date'].dt.year
    df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df['5_game_avg'] = df['5_game_avg'].replace(0, np.nan).fillna(df['calculated_dk_fpts'].mean())
    
    # Calculate key statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA[x['year']]) / 1.15) * x['AB'] if x['AB'] > 0 else 0, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA[x['year']] * 100) if x['AB'] > 0 and league_avg_wOBA[x['year']] > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall[x['year']] if league_avg_HR_FlyBall[x['year']] > 0 else 0, axis=1)

    # Calculate singles
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    # Calculate wOBA using year-specific weights
    df['wOBA_Statcast'] = df.apply(lambda x: (
        wOBA_weights[x['year']]['BB'] * x['BB'] +
        wOBA_weights[x['year']]['HBP'] * x['HBP'] +
        wOBA_weights[x['year']]['1B'] * x['1B'] +
        wOBA_weights[x['year']]['2B'] * x['2B'] +
        wOBA_weights[x['year']]['3B'] * x['3B'] +
        wOBA_weights[x['year']]['HR'] * x['HR']
    ) / (x['AB'] + x['BB'] - x['IBB'] + x['SF'] + x['HBP']) if (x['AB'] + x['BB'] - x['IBB'] + x['SF'] + x['HBP']) > 0 else 0, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = df.apply(lambda x: (
        x['1B'] + (2 * x['2B']) + (3 * x['3B']) + (4 * x['HR'])
    ) / x['AB'] if x['AB'] > 0 else 0, axis=1)

    # Calculate RAR_Statcast (Runs Above Replacement)
    df['RAR_Statcast'] = df['WAR'] * 10 if 'WAR' in df.columns else 0

    # Calculate Offense_Statcast
    df['Offense_Statcast'] = df['wRAA'] + df['BsR'] if 'BsR' in df.columns else df['wRAA']

    # Calculate Dollars_Statcast
    WAR_conversion_factor = 8.0  # Example conversion factor, can be adjusted
    df['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor if 'WAR' in df.columns else 0

    # Calculate WPA/LI_Statcast
    df['WPA/LI_Statcast'] = df['WPA/LI'] if 'WPA/LI' in df.columns else 0

    # Calculate rolling statistics if 'calculated_dk_fpts' is present
    if 'calculated_dk_fpts' in df.columns:
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Calculate the 5-game average and difference from prediction
    df = df.sort_values(['Name', 'date'])
    df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['5_game_diff'] = df['calculated_dk_fpts'] - df['5_game_avg']
    
    # Calculate positive and negative differences
    df['5_game_pos_diff'] = df['5_game_diff'].clip(lower=0)
    df['5_game_neg_diff'] = df['5_game_diff'].clip(upper=0)
    
    # Calculate rolling average of positive and negative differences
    df['5_game_pos_diff_avg'] = df.groupby('Name')['5_game_pos_diff'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['5_game_neg_diff_avg'] = df.groupby('Name')['5_game_neg_diff'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Debug: Print 5-game average calculation
    print("5-game average calculation:", df[['Name', 'date', 'calculated_dk_fpts', '5_game_avg']].head(10))
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Player consistency features
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    # Create interaction terms after encoding
    df['Name_Opponent_Interaction'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    df['Name_Team_Interaction'] = df['Name_encoded'] * df['Team_encoded']
    df['Team_Opponent_Interaction'] = df['Team_encoded'] * df['Opponent_encoded']
    df['Date_Team_Interaction'] = df['year'].astype(str) + '_' + df['Team_encoded'].astype(str)
    
    # Convert date to a numeric format for interaction
    df['date_numeric'] = pd.to_datetime(df['date']).astype(int) // 10**9  # Convert to Unix timestamp
    df['interaction_name_date'] = df['Name_encoded'].astype(str) + '_' + df['date_numeric'].astype(str)
    df['interaction_name_team'] = df['Name_encoded'].astype(str) + '_' + df['Team_encoded'].astype(str)
    df['interaction_name_opponent'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    
    # Add teammate interactions
    df = add_teammate_interactions(df)
    
    df.fillna(0, inplace=True)
    
    print("Columns at the end of engineer_features:", df.columns.tolist())
    return df

def predict_for_all_players(df, prediction_date):
    df_all_players = df.drop_duplicates(subset=['Name', 'Team']).copy()
    df_all_players['date'] = pd.to_datetime(prediction_date)
    
    df_all_players = engineer_features(df_all_players)

    X_future = df_all_players[selected_features]

    print(f"Shape of X_future: {X_future.shape}")
    print(f"Features used: {X_future.columns.tolist()}")

    if X_future.shape[1] != len(selected_features):
        raise ValueError(f"Expected {len(selected_features)} features, but got {X_future.shape[1]}. Please check the 'selected_features' list.")
    
    X_future_scaled = scaler.transform(X_future)

    # Load ensemble models
    rf_model = joblib.load('/Users/sineshawmesfintesfaye/newenv/rf_model.pkl')
    gb_model = joblib.load('/Users/sineshawmesfintesfaye/newenv/gb_model.pkl')

    # Make predictions with all models
    y_pred_nn = model.predict(X_future_scaled)
    y_pred_rf = rf_model.predict(X_future_scaled)
    y_pred_gb = gb_model.predict(X_future_scaled)

    # Ensemble predictions
    y_pred_future = (y_pred_nn.flatten() + y_pred_rf + y_pred_gb) / 3

    predicted_df = df_all_players[['Name', 'Team', 'date', 'Pos']].copy()
    predicted_df['predicted_dk_fpts'] = y_pred_future

    return predicted_df

# Load dataset for prediction
input_file = '/users/sineshawmesfintesfaye/newenv/merged_output_29_sep_01.csv'
df = pd.read_csv(input_file, low_memory=False)
df['date'] = pd.to_datetime(df['date'])

# Set the date for prediction (e.g., 2024-09-29)
date = '2024-09-29'

try:
    predicted_df = predict_for_all_players(df, date)
    
    output_file = '/Users/sineshawmesfintesfaye/newenv/fan_graphs_sep_18_merged.csv'
    predicted_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(predicted_df)}")
    print(predicted_df.head())
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    print(traceback.format_exc())
