import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import warnings
import multiprocessing
import os
import logging
from sklearn.neural_network import MLPRegressor
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import ta
import sys
import json

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust chunksize based on your system's memory
chunksize = 5000  # Adjust this value as needed

# Define paths for saving the label encoders and scaler
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/name_encoder_1_nba4.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/team_encoder_1_nba.pkl'
opponent_encoder_path = '/Users/sineshawmesfintesfaye/newenv/opponent_encoder_nba.pkl'
scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler_nba.pkl'  # Use this path consistently

selected_features = [
    'Player', 'Date', 'Team', 'Opp', 'Result',  'MP',  'calculated_dk_fpts',
    'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
    'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
    'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', 'Name_encoded',
    'Team_encoded','Opponent_encoded','Salary',
    'interaction_name_date','interaction_name_team','interaction_name_opponent',
    'calculated_dk_fpts','5_game_avg','rolling_mean_fpts','rolling_mean_fpts_opponent',
    'rolling_min_fpts_7','rolling_max_fpts_7','rolling_mean_fpts_7','lag_mean_fpts_7',
    'lag_max_fpts_7','lag_min_fpts_7','rolling_min_fpts_49','rolling_max_fpts_49',
    'rolling_mean_fpts_49','lag_mean_fpts_49','lag_max_fpts_49','lag_min_fpts_49',
    'year','month','day','day_of_week','day_of_season',
    'week_of_season','day_of_year','rolling_dk_fpts_3','rolling_dk_fpts_7',
    'rolling_dk_fpts_14','rolling_dk_fpts_28','dk_fpts_std','dk_fpts_consistency'
    
]
# Update selected_features
selected_features.extend([
    'eFG%', 'rolling_dk_fpts_3', 'rolling_dk_fpts_21', 'rolling_fouls_5',
    'rolling_turnovers_5', 'TOV%', 'fantasy_points_per_minute',
    'opp_points_allowed', 'hot_streak', 'usage_proxy', 'OREB%', 'DREB%',
    'days_since_last_game', 'back_to_back'
])
selected_features.extend(['is_home', 'opp_recent_avg_pts', 'opp_defensive_strength'])

#  df['Name_Opponent_Interaction'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
#     df['Name_Team_Interaction'] = df['Name_encoded'] * df['Team_encoded']
#     df['Team_Opponent_Interaction'] = df['Team_encoded'] * df['Opponent_encoded']
#     df['Date_Team_Interaction'] = df['year'].astype(str) + '_' + df['Team_encoded'].astype(str)
    

# Add this function near the top of your script, before engineer_features
def calculate_dk_fpts(row):
    points = row['PTS']
    rebounds = row['TRB']
    assists = row['AST']
    steals = row['STL']
    blocks = row['BLK']
    turnovers = row['TOV']
    
    dk_fpts = (points * 1 + 
               rebounds * 1.25 + 
               assists * 1.5 + 
               steals * 2 + 
               blocks * 2 - 
               turnovers * 0.5)
    
    # Bonus points
    if points >= 10 and rebounds >= 10:
        dk_fpts += 1.5
    if points >= 10 and assists >= 10:
        dk_fpts += 1.5
    if rebounds >= 10 and assists >= 10:
        dk_fpts += 1.5
    if points >= 10 and rebounds >= 10 and assists >= 10:
        dk_fpts += 3
    
    return dk_fpts

def load_or_create_label_encoders(df):
    le_name = LabelEncoder()
    le_team = LabelEncoder()
    le_opponent = None
    
    # Fill NaN values for Name and Team
    df['Player'] = df['Player'].fillna('Unknown')
    df['Team'] = df['Team'].fillna('Unknown')

    # Handle opponent encoding
    if 'Opp' in df.columns:
        le_opponent = LabelEncoder()
        df['Opp'] = df['Opp'].fillna('Unknown')
        le_opponent.fit(df['Opp'].unique())
        joblib.dump(le_opponent, opponent_encoder_path)
    else:
        print("Warning: 'Opp' column not found. Creating dummy opponent encoding.")
        # Create a dummy opponent encoder with a single class
        le_opponent = LabelEncoder()
        le_opponent.fit(['Unknown'])
        df['Opp'] = 'Unknown'  # Add dummy Opp column
        joblib.dump(le_opponent, opponent_encoder_path)

    le_name.fit(df['Player'].unique())
    le_team.fit(df['Team'].unique())
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    
    return le_name, le_team, le_opponent

# Add this function to identify numeric columns
def identify_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    return list(numeric_columns)

# Modify the load_or_create_scaler function
def load_or_create_scaler(df, numeric_features):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        # Filter out non-numeric columns
        valid_numeric_features = [col for col in numeric_features if col in identify_numeric_columns(df)]
        scaler.fit(df[valid_numeric_features])
        joblib.dump(scaler, scaler_path)
    return scaler

def add_teammate_interactions(df):
    try:
        # Ensure 'calculated_dk_fpts' and  are numeric
        df['calculated_dk_fpts'] = pd.to_numeric(df['calculated_dk_fpts'], errors='coerce')
        # df[] = pd.to_numeric(df[], errors='coerce')

        # Log info about non-numeric values
        non_numeric_dk_fpts = df[pd.isna(df['calculated_dk_fpts'])]['calculated_dk_fpts'].unique()
        # non_numeric_dk_fpts_team = df[pd.isna(df[])][].unique()
        
        if len(non_numeric_dk_fpts) > 0:
            logger.warning(f"Non-numeric values found in 'calculated_dk_fpts': {non_numeric_dk_fpts}")
        # if len(non_numeric_dk_fpts_team) > 0:
        #     logger.warning(f"Non-numeric values found in : {non_numeric_dk_fpts_team}")

        # Calculate teammate_dk_fpts
        # df['teammate_dk_fpts'] = df[] - df['calculated_dk_fpts']

        # Handle any potential NaN values resulting from the subtraction
        # df['teammate_dk_fpts'] = df['teammate_dk_fpts'].fillna(0)

        logger.info("Teammate interactions added successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in add_teammate_interactions: {str(e)}")
        raise

def engineer_features(df, date_series=None):
    print("Starting feature engineering...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Columns before engineering: {df.columns.tolist()}")
    
    # Create a list of required columns with default values
    required_columns = {
        'FG': 0, 'FGA': 0, 'FG%': 0, '2P': 0, '2PA': 0, '2P%': 0,
        '3P': 0, '3PA': 0, '3P%': 0, 'FT': 0, 'FTA': 0, 'FT%': 0,
        'TS%': 0, 'ORB': 0, 'DRB': 0, 'TRB': 0, 'AST': 0, 'STL': 0,
        'BLK': 0, 'TOV': 0, 'PF': 0, 'PTS': 0, 'GmSc': 0, 'BPM': 0,
        'MP': 0
    }
    
    # Add missing columns with default values
    for col, default_value in required_columns.items():
        if col not in df.columns:
            print(f"Adding missing column {col} with default value {default_value}")
            df[col] = default_value
    
    if date_series is None:
        if 'Date' not in df.columns:
            print("Warning: No Date column found. Using today's date.")
            df['Date'] = pd.Timestamp.now()
        date_series = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    # Ensure calculated_dk_fpts is not dropped
    if 'calculated_dk_fpts' in df.columns:
        calculated_dk_fpts = df['calculated_dk_fpts'].copy()
    else:
        print("Warning: 'calculated_dk_fpts' not found in columns. Calculating it now.")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
    
    # Extract date features
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days
    df['week_of_season'] = (date_series - date_series.min()).dt.days // 7
    df['day_of_year'] = date_series.dt.dayofyear

    # Calculate DK fantasy points if not already present
    if 'calculated_dk_fpts' not in df.columns:
        required_columns = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
        if all(col in df.columns for col in required_columns):
            df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
        else:
            print("Warning: Some required columns for DK points calculation are missing. Setting calculated_dk_fpts to 0.")
            df['calculated_dk_fpts'] = 0

    # Create Name_encoded column if it doesn't exist
    if 'Name_encoded' not in df.columns:
        if 'Player' in df.columns:
            le_name = LabelEncoder()
            df['Name_encoded'] = le_name.fit_transform(df['Player'])
        else:
            print("Warning: 'Player' column not found. Cannot create 'Name_encoded'.")
            df['Name_encoded'] = 0

    # Now we can use 'calculated_dk_fpts' for other calculations
    df['5_game_avg'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['rolling_mean_fpts'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(20, min_periods=1).mean())
    
    # Check if 'Opp' column exists before using it
    if 'Opp' in df.columns:
        df['rolling_mean_fpts_opponent'] = df.groupby(['Player', 'Opp'])['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(20, min_periods=1).mean())
    else:
        print("Warning: 'Opp' column not found. Skipping opponent-related calculations.")
    
    # Add the missing lag calculations
    for window in [7, 49]:
        df[f'lag_mean_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        df[f'lag_max_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(window, min_periods=1).max())
        df[f'lag_min_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(window, min_periods=1).min())

    # Add player_team_date_interaction
    df['player_team_date_interaction'] = df['Player'].astype(str) + '_' + df['Team'].astype(str) + '_' + df['Date'].astype(str)

    # Create interaction features
    df['interaction_name_date'] = df['Name_encoded'].astype(str) + '_' + df['Date'].astype(str)
    df['interaction_name_team'] = df['Name_encoded'].astype(str) + '_' + df['Team'].astype(str)
    
    if 'Opp' in df.columns:
        df['interaction_name_opponent'] = df['Name_encoded'].astype(str) + '_' + df['Opp'].astype(str)
    else:
        print("Warning: 'Opp' column not found. Skipping interaction_name_opponent calculation.")

    # Add back the previously removed features
    df['rolling_min_fpts_7'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).min())
    df['rolling_max_fpts_7'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).max())
    df['rolling_mean_fpts_7'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_min_fpts_49'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).min())
    df['rolling_max_fpts_49'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).max())
    df['rolling_mean_fpts_49'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).mean())

    # Add teammate interactions
    df = add_teammate_interactions(df)

    # Convert relevant columns to numeric, coercing errors to NaN
    df['FG'] = pd.to_numeric(df['FG'], errors='coerce')
    df['3P'] = pd.to_numeric(df['3P'], errors='coerce')
    df['FGA'] = pd.to_numeric(df['FGA'], errors='coerce')
    df['MP'] = pd.to_numeric(df['MP'], errors='coerce')  # Ensure MP is numeric
    
    # Debugging: Identify problematic rows for eFG% calculation
    try:
        df['eFG%'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
    except Exception as e:
        problematic_rows = df[df[['FG', '3P', 'FGA']].isnull().any(axis=1)]
        print("Error calculating eFG% for the following rows:")
        print(problematic_rows)
        raise e  # Re-raise the exception after logging

    # Add consistency metrics
    df['dk_fpts_std'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['dk_fpts_consistency'] = df['rolling_mean_fpts_7'] / df['dk_fpts_std'].replace(0, 1)  # Avoid division by zero

    # Add rolling averages for DK points
    for window in [3, 7, 14, 28]:
        df[f'rolling_dk_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # Add new features
    
    # 1. True Shooting Percentage (TS%) is already in the dataset
    
    # 2. Effective Field Goal Percentage (eFG%)
    df['eFG%'] = (df['FG'] + 0.5 * df['3P']) / df['FGA']
    
    # 3. Advanced Rolling Averages and Moving Windows
    for window in [3, 21]:  # Adding 3-game and 21-game windows
        df[f'rolling_dk_fpts_{window}'] = df.groupby('Player')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # 4. Foul Trouble and Turnover Analysis
    df['rolling_fouls_5'] = df.groupby('Player')['PF'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['rolling_turnovers_5'] = df.groupby('Player')['TOV'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['TOV%'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    
    # 5. Fantasy Points Per Minute
    try:
        df['fantasy_points_per_minute'] = df['calculated_dk_fpts'] / df['MP']
    except Exception as e:
        problematic_rows = df[df['MP'].isnull()]
        print("Error calculating fantasy_points_per_minute for the following rows:")
        print(problematic_rows)
        raise e  # Re-raise the exception after logging
    
    # 6. Opponent's Defensive Rating (proxy using points allowed)
    df['opp_points_allowed'] = df.groupby('Opp')['PTS'].transform('mean')
    
    # 7. Player Hot/Cold Streaks
    df['rolling_pts_5'] = df.groupby('Player')['PTS'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['player_avg_pts'] = df.groupby('Player')['PTS'].transform('mean')
    df['hot_streak'] = (df['rolling_pts_5'] > 1.2 * df['player_avg_pts']).astype(int)
    
    # 8. Usage Rate Proxy
    df['usage_proxy'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MP']
    
    # 9. Rebound Rates
    # Note: We don't have opponent ORB and DRB, so we'll use a simplified version
    df['OREB%'] = df['ORB'] / (df['ORB'] + df['DRB'])
    df['DREB%'] = df['DRB'] / (df['ORB'] + df['DRB'])
    
    # 10. Player Fatigue/Back-to-Back Games
    df['days_since_last_game'] = df.groupby('Player')['Date'].diff().dt.days
    df['back_to_back'] = (df['days_since_last_game'] == 1).astype(int)

    # Add opponent defensive strength
    # Calculate the average points allowed by each opponent
    df['opp_defensive_strength'] = df.groupby('Opp')['PTS'].transform('mean')

    print(f"Columns after engineering: {df.columns.tolist()}")
    print(f"Output DataFrame shape: {df.shape}")
    return df





# Update selected_features and numeric_features


def engineer_chunk(chunk):
    if chunk.empty:
        print("Warning: Empty chunk encountered.")
        return pd.DataFrame()  # Return an empty DataFrame for empty chunks
    return engineer_features(chunk)

def concurrent_feature_engineering(df, chunksize):
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping feature engineering.")
        return df

    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        processed_chunks = list(executor.map(engineer_chunk, chunks))
    
    # Filter out empty chunks
    processed_chunks = [chunk for chunk in processed_chunks if not chunk.empty]
    
    if not processed_chunks:
        print("Warning: All processed chunks are empty. Applying feature engineering to the entire dataframe.")
        return engineer_features(df)
    
    result = pd.concat(processed_chunks)
    print(f"Shape after feature engineering: {result.shape}")
    return result

def calculate_player_performance_against_teams(df):
    # Calculate average DK points against each opponent
    df['avg_dk_fpts_vs_opponent'] = df.groupby(['Player', 'opponent'])['calculated_dk_fpts'].transform('mean')
    return df

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    
    # Ensure the Date column is of type datetime64
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print("Converted Date column to datetime.")

    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Player'] == player].sort_values('Date', ascending=False)
        
        # Debugging: Print the player_df to check the order and content
        print(f"\nPlayer: {player}")
        print(player_df[['Date', 'calculated_dk_fpts']].head(10))  # Print the first 10 rows for inspection
        
        if player_df.empty:
            print(f"No historical data found for player {player}. Using randomized default values.")
            default_row = pd.DataFrame([{col: np.random.uniform(0, 1) for col in df.columns}])
            default_row['Date'] = prediction_date
            default_row['Player'] = player
            default_row['calculated_dk_fpts'] = np.random.uniform(0, 5)  # Random value between 0 and 5
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['Date'].min()} to {player_df['Date'].max()}")
            
            # Use all available data, up to 5 most recent games
            player_df = player_df.head(5)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['Date'] = prediction_date
            synthetic_row['Player'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and calculated correctly
            if 'calculated_dk_fpts' in player_df.columns:
                synthetic_row['calculated_dk_fpts'] = player_df['calculated_dk_fpts'].mean()
            else:
                synthetic_row['calculated_dk_fpts'] = np.nan  # Placeholder, replace with actual calculation if needed
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['Date', 'Player']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            # Add matchup data
            synthetic_row['opponent'] = player_df['Opp'].iloc[0]
            synthetic_row['is_home'] = player_df['is_home'].iloc[0]
            synthetic_row['opp_recent_avg_pts'] = player_df['opp_recent_avg_pts'].iloc[0]
            
            # Add player-specific performance against the opponent
            synthetic_row['avg_dk_fpts_vs_opponent'] = player_df['avg_dk_fpts_vs_opponent'].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    if synthetic_rows:
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
        print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
        return synthetic_df
    else:
        print("Warning: No synthetic rows created.")
        return pd.DataFrame()

def analyze_prediction_differences(df):
    # Sort the DataFrame by 'Player' and 'Date' to ensure the last 5 games are correctly identified
    df = df.sort_values(by=['Player', 'Date'])
    
    # Create a new column to identify the last 5 games for each player
    df['last_5_games'] = df.groupby('Player')['Date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=10)))

    # Filter the DataFrame to include only the last 5 games
    last_5_games_df = df[df['last_5_games']]

    # Calculate the difference
    last_5_games_df['difference'] = last_5_games_df['predicted_dk_fpts'] - last_5_games_df['calculated_dk_fpts']
    
    player_adjustments = last_5_games_df.groupby('Player').agg({
        'difference': [
            ('avg_positive_diff', lambda x: x[x > 0].mean()),
            ('avg_negative_diff', lambda x: x[x < 0].mean())
        ]
    })
    
    player_adjustments.columns = player_adjustments.columns.droplevel()  # Flatten multi-level columns
    player_adjustments['avg_positive_diff'] = player_adjustments['avg_positive_diff'].fillna(0)
    player_adjustments['avg_negative_diff'] = player_adjustments['avg_negative_diff'].fillna(0)
    
    print("Player-specific adjustments calculated for the last 5 games.")
    return player_adjustments

# Modify the process_predictions function
def process_predictions(chunk, pipeline, player_adjustments):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    
    # Apply the new player-specific adjustment
    chunk['predicted_dk_fpts'] = chunk.apply(lambda row: adjust_predictions(row, player_adjustments), axis=1)
    
    return chunk

# Modify the rolling_predictions function
def rolling_predictions(train_data, model_pipeline, test_dates, chunksize, player_adjustments):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        train_data_filtered = train_data[train_data['Date'] < current_date]  # Exclude data on or after the current date
        synthetic_rows = create_synthetic_rows_for_all_players(train_data_filtered, train_data_filtered['Player'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, 
                                                 [model_pipeline]*len(chunks), 
                                                 [player_adjustments]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred, dates):
    """Evaluate model performance with time consideration"""
    print("Evaluating model...")
    
    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Create evaluation DataFrame
    eval_df = pd.DataFrame({
        'date': dates,
        'actual': y_true,
        'predicted': y_pred,
        'error': y_pred - y_true
    })
    
    # Calculate metrics by time period
    time_based_metrics = eval_df.set_index('date').resample('W').agg({
        'error': ['mean', 'std', 'count'],
        'actual': 'mean',
        'predicted': 'mean'
    })
    
    # Save time-based evaluation
    time_based_metrics.to_csv('/Users/sineshawmesfintesfaye/newenv/time_based_evaluation.csv')
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    
    return mae, mse, r2

# Define the polynomial regression model
print("Defining poly_model...")
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
print("poly_model defined.")

# Update the base models for stacking
print("Defining base_models...")
base_models = [
    ('poly', poly_model),  # Replacing linear regression with polynomial regression
    ('lasso', Lasso()),
    ('dt', DecisionTreeRegressor()),
    ('svr', SVR()),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1)),
    ('bagging', BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)),
    ('nnw', MLPRegressor(
        hidden_layer_sizes=(50, 25),  # Smaller network
        max_iter=1000,  # Increased iterations
        random_state=42,
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        learning_rate_init=0.001,
        alpha=0.0001
    ))
]
print("base_models defined.")

# Update the stacking model
print("Defining stacking_model...")
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)
print("stacking_model defined.")

# The final model remains the same
final_model = StackingRegressor(
    estimators=[('stacking', stacking_model)],
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)

def clean_infinite_values(df):
    print("Starting to clean infinite values...")
    
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Replaced inf and -inf with NaN")
    
    # For numeric columns, replace NaN with the mean of the column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    total_columns = len(numeric_columns)
    for i, col in enumerate(numeric_columns, 1):
        if i % 10 == 0:  # Print progress every 10 columns
            print(f"Processing numeric column {i}/{total_columns}: {col}")
        df[col] = df[col].fillna(df[col].mean())
    
    # For non-numeric columns, replace NaN with a placeholder value (e.g., 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def rolling_window_prediction(df, pipeline, window_size=30, step_size=1):
    results = []
    dates = df['Date'].sort_values().unique()
    
    print(f"Total unique dates: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Adjust window_size if we don't have enough dates
    if len(dates) <= window_size:
        print(f"Warning: Not enough dates for specified window size. Adjusting window size to {len(dates) - 1}")
        window_size = len(dates) - 1
    
    for i in range(window_size, len(dates), step_size):
        train_dates = dates[max(0, i-window_size):i]
        test_date = dates[i]
        
        print(f"Processing window: Train dates from {train_dates[0]} to {train_dates[-1]}, Test date: {test_date}")
        
        train_data = df[df['Date'].isin(train_dates)]
        test_data = df[df['Date'] == test_date]
        
        if train_data.empty or test_data.empty:
            print(f"Warning: Empty train or test data for window ending on {test_date}")
            continue
        
        train_features = train_data[all_features]
        test_features = test_data[all_features]
        
        print(f"Train data shape: {train_features.shape}, Test data shape: {test_features.shape}")
        
        pipeline.fit(train_features, train_data['calculated_dk_fpts'])
        predictions = pipeline.predict(test_features)
        
        test_data['predicted_dk_fpts'] = predictions
        results.append(test_data)
        
        print(f"Processed window ending on {test_date}, predictions shape: {predictions.shape}")
    
    if not results:
        print("Warning: No predictions were generated.")
        return df  # Return original dataframe if no predictions
    
    return pd.concat(results)

def parse_date(date_str):
    try:
        # First, try parsing as a regular date
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        try:
            # If that fails, try parsing as a timestamp
            return pd.to_datetime(date_str, unit='s')
        except ValueError:
            # If all else fails, return NaT
            return pd.NaT

# Add this definition near the top of your script, after the imports and before the main execution

all_features = [
    'MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
    'FT', 'FTA', 'FT%', 'TS%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 
    'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', 'Name_encoded',
    'Team_encoded', 'Opponent_encoded', 'calculated_dk_fpts', '5_game_avg',
    'rolling_mean_fpts', 'rolling_mean_fpts_opponent', 'rolling_min_fpts_7',
    'rolling_max_fpts_7', 'rolling_mean_fpts_7', 'lag_mean_fpts_7',
    'lag_max_fpts_7', 'lag_min_fpts_7', 'rolling_min_fpts_49', 'rolling_max_fpts_49',
    'rolling_mean_fpts_49', 'lag_mean_fpts_49', 'lag_max_fpts_49', 'lag_min_fpts_49',
    'year', 'month', 'day', 'day_of_week', 'day_of_season',
    'week_of_season', 'day_of_year', 'rolling_dk_fpts_3', 'rolling_dk_fpts_7',
    'rolling_dk_fpts_14', 'rolling_dk_fpts_28', 'dk_fpts_std', 'dk_fpts_consistency',
    'Player', 'Date', 'Team', 'Opp', 'Result', 'interaction_name_opponent',
    'interaction_name_team', 'interaction_name_date', 'player_team_date_interaction',
    'eFG%', 'rolling_dk_fpts_3', 'rolling_dk_fpts_21', 'rolling_fouls_5',
    'rolling_turnovers_5', 'TOV%', 'fantasy_points_per_minute',
    'opp_points_allowed', 'hot_streak', 'usage_proxy', 'OREB%', 'DREB%',
    'days_since_last_game', 'back_to_back','eFG%', 'rolling_dk_fpts_3',
    'rolling_dk_fpts_21', 'rolling_fouls_5',
    'rolling_turnovers_5', 'TOV%', 'fantasy_points_per_minute',
    'opp_points_allowed', 'hot_streak', 'usage_proxy', 'OREB%', 'DREB%',
    'days_since_last_game', 'back_to_back','Salary'
]

def calculate_and_save_adjustments(df, output_path):
    # Ensure predictions are made before calculating adjustments
    if 'predicted_dk_fpts' not in df.columns:
        print("Error: 'predicted_dk_fpts' column not found. Ensure predictions are made before calculating adjustments.")
        return

    # Sort the DataFrame by 'Player' and 'Date'
    df = df.sort_values(by=['Player', 'Date'])
    
    # Create a new column to identify the last 3 games for each player
    df['last_3_games'] = df.groupby('Player')['Date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=3)))

    # Filter the DataFrame to include only the last 3 games
    last_3_games_df = df[df['last_3_games']]

    # Calculate the difference
    last_3_games_df['difference'] = last_3_games_df['predicted_dk_fpts'] - last_3_games_df['calculated_dk_fpts']
    
    player_adjustments = last_3_games_df.groupby('Player').agg({
        'difference': [
            ('avg_positive_diff', lambda x: x[x > 0].mean()),
            ('avg_negative_diff', lambda x: x[x < 0].mean())
        ]
    })
    
    player_adjustments.columns = player_adjustments.columns.droplevel()  # Flatten multi-level columns
    player_adjustments['avg_positive_diff'] = player_adjustments['avg_positive_diff'].fillna(0)
    player_adjustments['avg_negative_diff'] = player_adjustments['avg_negative_diff'].fillna(0)
    
    print("Player-specific adjustments calculated for the last 3 games.")
    
    # Save the adjustments to a file
    joblib.dump(player_adjustments, output_path)
    print(f"Player adjustments saved to {output_path}")

def analyze_predictions_over_time(df):
    """Analyze prediction errors chronologically"""
    df = df.sort_values('Date')
    
    # Calculate daily average errors
    daily_errors = df.groupby('Date').agg({
        'prediction_error': ['mean', 'std', 'count'],
        'prediction_error_percentage': ['mean', 'std']
    }).round(2)
    
    # Save detailed analysis
    error_analysis_path = '/Users/sineshawmesfintesfaye/newenv/prediction_error_analysis.csv'
    daily_errors.to_csv(error_analysis_path)
    print(f"Error analysis saved to {error_analysis_path}")
    
    # Plot error trends
    plt.figure(figsize=(12, 6))
    plt.plot(daily_errors.index, daily_errors[('prediction_error', 'mean')])
    plt.title('Average Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/error_trend.png')
    plt.close()

def check_data_leakage(df):
    """Check for potential data leakage issues"""
    df = df.sort_values(['Player', 'Date'])
    
    # Check for future data in rolling calculations
    for col in df.columns:
        if 'rolling' in col.lower() or 'lag' in col.lower():
            current_values = df.groupby('Player')[col].shift(0)
            future_values = df.groupby('Player')[col].shift(-1)
            
            # Check if future values influence current values
            correlation = current_values.corr(future_values)
            if correlation > 0.9:  # High correlation might indicate leakage
                print(f"Warning: Possible data leakage in column {col}")
                print(f"Correlation with future values: {correlation:.2f}")

# Add this function to calculate technical indicators
def calculate_technical_indicators(df, column='calculated_dk_fpts'):
    """Calculate technical analysis indicators for the specified column."""
    # Sort by date to ensure proper calculation
    df = df.sort_values(['Player', 'Date'])
    
    # Simple Moving Averages
    df[f'{column}_sma7'] = df.groupby('Player')[column].transform(
        lambda x: x.rolling(window=7).mean())
    df[f'{column}_sma14'] = df.groupby('Player')[column].transform(
        lambda x: x.rolling(window=14).mean())
    
    # Exponential Moving Averages
    df[f'{column}_ema7'] = df.groupby('Player')[column].transform(
        lambda x: x.ewm(span=7, adjust=False).mean())
    df[f'{column}_ema14'] = df.groupby('Player')[column].transform(
        lambda x: x.ewm(span=14, adjust=False).mean())
    
    # Calculate indicators for each player separately
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player].copy()
        
        # MACD
        macd = MACD(close=player_data[column])
        df.loc[player_data.index, f'{column}_macd'] = macd.macd()
        df.loc[player_data.index, f'{column}_macd_signal'] = macd.macd_signal()
        
        # RSI
        rsi = RSIIndicator(close=player_data[column])
        df.loc[player_data.index, f'{column}_rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=player_data[column])
        df.loc[player_data.index, f'{column}_bb_high'] = bb.bollinger_hband()
        df.loc[player_data.index, f'{column}_bb_low'] = bb.bollinger_lband()
        df.loc[player_data.index, f'{column}_bb_mid'] = bb.bollinger_mavg()
    
    return df

# Add this class for the prediction feedback loop
class PredictionFeedbackLoop:
    def __init__(self, model_path, adjustment_window=10):
        self.model_path = model_path
        self.adjustment_window = adjustment_window
        self.prediction_history = pd.DataFrame()
        self.adjustment_factors = {}
        
    def update_prediction_history(self, new_predictions, actual_values):
        """Update prediction history with new predictions and actual values."""
        new_data = pd.DataFrame({
            'predicted': new_predictions,
            'actual': actual_values,
            'date': pd.Timestamp.now()
        })
        self.prediction_history = pd.concat([self.prediction_history, new_data])
        self.prediction_history = self.prediction_history.tail(1000)  # Keep last 1000 predictions
        
    def calculate_adjustment_factors(self):
        """Calculate adjustment factors based on recent prediction accuracy."""
        recent_predictions = self.prediction_history.tail(self.adjustment_window)
        if not recent_predictions.empty:
            mape = mean_absolute_percentage_error(
                recent_predictions['actual'], 
                recent_predictions['predicted']
            )
            bias = (recent_predictions['predicted'] - recent_predictions['actual']).mean()
            
            # Calculate adjustment factor
            if bias > 0:  # Consistently overpredicting
                adjustment = 1 - (mape * 0.1)  # Reduce predictions
            else:  # Consistently underpredicting
                adjustment = 1 + (mape * 0.1)  # Increase predictions
                
            return adjustment
        return 1.0
        
    def adjust_predictions(self, predictions):
        """Adjust new predictions based on historical accuracy."""
        adjustment = self.calculate_adjustment_factors()
        return predictions * adjustment

# Add this near the top of your script
def verify_required_columns(df):
    minimum_required = ['Player', 'Team', 'Date']
    missing = [col for col in minimum_required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

# Modify the main execution section
if __name__ == "__main__":
    start_time = time.time()
    
    # Loading dataset
    file_path = '/Users/sineshawmesfintesfaye/newenv/nba_oct_4name_display_csk/fuckoifyou.csv'
    print(f"Loading data from {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Initial DataFrame shape: {df.shape}")
        print(f"Initial columns: {df.columns.tolist()}")
        
        # Use calculated_salary if available, otherwise 0
        if 'calculated_salary' in df.columns:
            df['Salary'] = df['calculated_salary']
        else:
            df['Salary'] = 0
            
        # Convert Date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Get unique dates for splitting
        unique_dates = df['Date'].unique()
        n_dates = len(unique_dates)
        split_idx = int(n_dates * 0.8)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]
        
        # Split data
        train_data = df[df['Date'].isin(train_dates)].copy()
        test_data = df[df['Date'].isin(test_dates)].copy()
        
        print(f"Train data: {len(train_data)} rows, date range: {train_data['Date'].min()} to {train_data['Date'].max()}")
        print(f"Test data: {len(test_data)} rows, date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
        
        if len(train_data) == 0:
            print("Warning: No training data. Using 80-20 split of all data instead.")
            train_data = df.sample(frac=0.8, random_state=42)
            test_data = df.drop(train_data.index)
        
        # Feature engineering
        train_data = engineer_features(train_data)
        test_data = engineer_features(test_data)
        
        # Clean infinite values
        train_data = clean_infinite_values(train_data)
        test_data = clean_infinite_values(test_data)
        
        # Define features for model - use the actual columns from train_data
        print("Available columns:", train_data.columns.tolist())
        
        # Define features to exclude
        exclude_columns = ['Date', 'Player', 'Team', 'Opp', 'calculated_dk_fpts', 
                          'predicted_dk_fpts', 'prediction_error', 'prediction_error_percentage',
                          'player_team_date_interaction', 'interaction_name_date', 
                          'interaction_name_team', 'interaction_name_opponent']
        
        # Get feature columns that exist in the data
        feature_cols = [col for col in train_data.columns 
                       if col not in exclude_columns]
        
        print(f"Selected features: {len(feature_cols)} columns")
        print("First 10 features:", feature_cols[:10])
        
        # Split features into numeric and categorical based on actual data types
        numeric_features = []
        categorical_features = []
        
        for col in feature_cols:
            if col in train_data.columns:  # Verify column exists
                if train_data[col].dtype in ['int64', 'float64']:
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Prepare features
        X_train = train_data[feature_cols]
        y_train = train_data['calculated_dk_fpts']
        X_test = test_data[feature_cols]
        y_test = test_data['calculated_dk_fpts']
        
        # Define the pipeline components
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Feature selection
        selector = SelectKBest(f_regression, k=min(40, len(feature_cols)))

        # Create the full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('model', stacking_model)  # stacking_model was defined earlier
        ])

        # Now fit the pipeline
        print("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        
        # Add predictions to dataframes
        train_data['predicted_dk_fpts'] = train_predictions
        test_data['predicted_dk_fpts'] = test_predictions
        
        # Combine results
        full_results = pd.concat([train_data, test_data])
        
        # Safe sort
        sort_columns = ['Date', 'Player']
        if 'Salary' in full_results.columns:
            sort_columns.append('Salary')
        
        full_results = full_results.sort_values(sort_columns)
        
        # Save results
        output_path = '/Users/sineshawmesfintesfaye/newenv/nba_predictions_results.csv'
        full_results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Calculate and print metrics
        mae = mean_absolute_error(y_test, test_predictions)
        mse = mean_squared_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        print(f"\nModel Performance Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        print(f"R2 Score: {r2:.2f}")
        
        # Define model save paths
        model_dir = '/Users/sineshawmesfintesfaye/newenv/models'
        os.makedirs(model_dir, exist_ok=True)  # Create models directory if it doesn't exist
        
        model_file_path = os.path.join(model_dir, 'nba_prediction_model.pkl')
        feature_list_path = os.path.join(model_dir, 'feature_list.pkl')
        model_metrics_path = os.path.join(model_dir, 'model_metrics.json')
        
        # Save the trained model
        print(f"Saving model to {model_file_path}")
        joblib.dump(pipeline, model_file_path)
        
        # Save the feature list
        print(f"Saving feature list to {feature_list_path}")
        joblib.dump(feature_cols, feature_list_path)
        
        # Save model metrics
        metrics = {
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'training_date': str(pd.Timestamp.now()),
            'n_train_samples': len(train_data),
            'n_test_samples': len(test_data),
            'n_features': len(feature_cols)
        }
        
        print(f"Saving model metrics to {model_metrics_path}")
        with open(model_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nModel files saved:")
        print(f"- Model: {model_file_path}")
        print(f"- Features: {feature_list_path}")
        print(f"- Metrics: {model_metrics_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
