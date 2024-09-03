import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.model_selection import TimeSeriesSplit
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Define constants for calculations
# League averages for 2020 to 2024
league_avg_wOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315
}

league_avg_HR_FlyBall = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141
}

# wOBA weights for 2020 to 2024
wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

selected_features = [
     'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+', 
    'flyBalls', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
    'singles', 'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI','AB', 'WAR'  
]

engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast', 'Name_encoded', 'team_encoded','wRC+', 'wRAA', 'wOBA',   
]
selected_features += engineered_features

def calculate_dk_fpts(row):
    return (row['1B'] * 3 + row['2B'] * 5 + row['3B'] * 8 + row['HR'] * 10 +
            row['RBI'] * 2 + row['R'] * 2 + row['BB'] * 2 + row['HBP'] * 2 + row['SB'] * 5)

def engineer_features(df, date_series=None):
    if date_series is None:
        date_series = df['date']
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    # Extract date features
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days

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

    # Calculate 5-game average
    df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Handle zero values in 5-game average
    df['5_game_avg'] = df['5_game_avg'].replace(0, np.nan).fillna(df['calculated_dk_fpts'].mean())
    
    # Debug: Print 5-game average calculation
    print("5-game average calculation:", df[['Name', 'date', 'calculated_dk_fpts', '5_game_avg']].head(10))
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        processed_chunks = list(executor.map(engineer_features, chunks, [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using randomized default values.")
            default_row = pd.DataFrame([{col: np.random.uniform(0, 1) for col in df.columns}])
            default_row['date'] = prediction_date
            default_row['Name'] = player
            default_row['calculated_dk_fpts'] = np.random.uniform(0, 5)  # Random value between 0 and 5
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use all available data, up to 45 most recent games
            player_df = player_df.head(20)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

def process_predictions(chunk, pipeline, is_current_day=False):
    if is_current_day:
        features = chunk.drop(columns=['calculated_dk_fpts'])
    else:
        features = chunk
    
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    
    # Adjust predictions to be within the range of 5-game average ± 4
    chunk['predicted_dk_fpts'] = chunk.apply(
        lambda row: max(row['5_game_avg'] - 2, min(row['predicted_dk_fpts'], row['5_game_avg'] + 2)), axis=1
    )
    
    # Debug: Print after adjustment
    print("After adjustment:", chunk[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    return chunk

def process_chunk_predictions(chunk, pipeline, is_current_day=False):
    return process_predictions(chunk, pipeline, is_current_day)

def predict_unseen_data(input_file, model_file, prediction_date):
    print("Loading dataset...")
    df = pd.read_csv(input_file,
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    prediction_date = pd.to_datetime(prediction_date)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range in dataset: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique players: {df['Name'].nunique()}")
    
    # Get all unique players from the entire dataset
    all_players = df['Name'].unique()
    
    # No need to filter data up to the prediction date
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Load or create LabelEncoders
    name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_name_sep2.pkl'
    team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_team_sep2.pkl'

    if os.path.exists(name_encoder_path):
        le_name = joblib.load(name_encoder_path)
    else:
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)

    if os.path.exists(team_encoder_path):
        le_team = joblib.load(team_encoder_path)
    else:
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    # Update LabelEncoders with new players/teams
    new_names = set(df['Name']) - set(le_name.classes_)
    if new_names:
        le_name.classes_ = np.append(le_name.classes_, list(new_names))
        joblib.dump(le_name, name_encoder_path)  # Save updated encoder

    new_teams = set(df['Team']) - set(le_team.classes_)
    if new_teams:
        le_team.classes_ = np.append(le_team.classes_, list(new_teams))
        joblib.dump(le_team, team_encoder_path)  # Save updated encoder

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    chunksize = 10000
    df = concurrent_feature_engineering(df, chunksize)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    df['team_encoded'] = df['Team']
    df['Name_encoded'] = df['Name']

    if df.empty:
        raise ValueError(f"No data available up to {prediction_date}")
    
    print(f"Data available up to {df['date'].max()}")

    print("Loading model...")
    pipeline = joblib.load(model_file)
    
    print("Model pipeline steps:")
    for step_name, step in pipeline.named_steps.items():
        print(f"- {step_name}: {type(step).__name__}")
    
    print(f"Processing date: {prediction_date}")
    
    # Create synthetic rows for all players for the prediction date
    current_df = create_synthetic_rows_for_all_players(df, all_players, prediction_date)
    
    if current_df.empty:
        print(f"No data available for date: {prediction_date}")
        return
    
    # Process predictions in chunks
    chunks = [current_df[i:i+chunksize] for i in range(0, current_df.shape[0], chunksize)]
    chunk_predictions = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_pred = process_chunk_predictions(chunk, pipeline, is_current_day=True)
        chunk_predictions.append(chunk_pred)
    
    # Combine chunk predictions
    predictions = pd.concat(chunk_predictions)
    
    print("Prediction statistics:")
    if 'predicted_dk_fpts' in predictions.columns:
        print(predictions['predicted_dk_fpts'].describe())
    else:
        print("Error: 'predicted_dk_fpts' column not found in predictions.")
        print("Available columns:", predictions.columns.tolist())
    
    # Save predictions
    output_file = f'/Users/sineshawmesfintesfaye/newenv/batters_predictions_{prediction_date.strftime("%Y%m%d")}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Print a sample of predictions
    print("\nSample predictions:")
    print(predictions.head())

    return predictions

if __name__ == "__main__":
    input_file = '/Users/sineshawmesfintesfaye/newenv/fangraphs-leaderboards_full_sep1_batters.csv'
    model_file = '/Users/sineshawmesfintesfaye/newenv/batters_final_ensemble_model_pipeline.pkl'
    prediction_date = '2024-09-01'
    
    predict_unseen_data(input_file, model_file, prediction_date)