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

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this line near the beginning of your main script, after importing libraries
chunksize = 10000  # You can adjust this value based on your system's memory and processing power

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
    'Team_encoded','Opponent_encoded',
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
    
    if 'Opp' in df.columns:
        if os.path.exists(opponent_encoder_path):
            le_opponent = joblib.load(opponent_encoder_path)
            # Add unknown class if not present
            if 'Unknown' not in le_opponent.classes_:
                le_opponent.classes_ = np.append(le_opponent.classes_, 'Unknown')
        else:
            le_opponent = LabelEncoder()
        # Fill NaN values with 'Unknown' before fitting
        df['Opp'] = df['Opp'].fillna('Unknown')
        # Fit with all unique values including 'Unknown'
        le_opponent.fit(df['Opp'].unique())
        joblib.dump(le_opponent, opponent_encoder_path)
    else:
        print("Warning: 'Opp' column not found. Skipping opponent encoding.")
        le_opponent = None

    # Fill NaN values for Name and Team as well
    df['Player'] = df['Player'].fillna('Unknown')
    df['Team'] = df['Team'].fillna('Unknown')

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
    
    if date_series is None:
        date_series = df['Date']
    
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
    df['fantasy_points_per_minute'] = df['calculated_dk_fpts'] / df['MP']
    
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
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Player'] == player].sort_values('Date', ascending=False)
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
            
            # Use all available data, up to 20 most recent games
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
            synthetic_row['opponent'] = player_df['opponent'].iloc[0]
            synthetic_row['is_home'] = player_df['is_home'].iloc[0]
            synthetic_row['opp_recent_avg_pts'] = player_df['opp_recent_avg_pts'].iloc[0]
            
            # Add player-specific performance against the opponent
            synthetic_row['avg_dk_fpts_vs_opponent'] = player_df['avg_dk_fpts_vs_opponent'].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    if synthetic_rows:  # Check if synthetic_rows is not empty
        synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
        print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
        print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
        return synthetic_df
    else:
        print("Warning: No synthetic rows created.")
        return pd.DataFrame()  # Return an empty DataFrame if no synthetic rows were created

def analyze_prediction_differences(df):
    # Sort the DataFrame by 'Player' and 'Date' to ensure the last 5 games are correctly identified
    df = df.sort_values(by=['Player', 'Date'])
    
    # Create a new column to identify the last 5 games for each player
    df['last_5_games'] = df.groupby('Player')['Date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=5)))

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

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Add a small constant to avoid division by zero or very small values
    epsilon = 1e-10
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted DK Points")
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/actual_vs_predicted.png')
    plt.close()
    
    # Plotting residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/residual_plot.png')
    plt.close()
    
    print("Model evaluation completed.")
    return mae, mse, r2, mape

# Update the base models for stacking
base_models = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('dt', DecisionTreeRegressor()),
    ('svr', SVR()),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1)),
    ('bagging', BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)),
    ('poly', make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
]

# Update the stacking model
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
    
    for i in range(window_size, len(dates)):
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
    'days_since_last_game', 'back_to_back','eFG%', 'rolling_dk_fpts_3', 'rolling_dk_fpts_21', 'rolling_fouls_5',
    'rolling_turnovers_5', 'TOV%', 'fantasy_points_per_minute',
    'opp_points_allowed', 'hot_streak', 'usage_proxy', 'OREB%', 'DREB%',
    'days_since_last_game', 'back_to_back',
]

if __name__ == "__main__":
    start_time = time.time()
    
    # Loading dataset
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/nba_oct_7name_display_csk/merged_output_nba_4.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values(by=['Player', 'Date'], inplace=True)
    df.fillna(0, inplace=True)

    # Load or create LabelEncoders
    le_name, le_team, le_opponent = load_or_create_label_encoders(df)
    df['Name_encoded'] = le_name.transform(df['Player'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    df['Opponent_encoded'] = le_opponent.transform(df['Opp'])

    # Calculate DK fantasy points if not already present
    if 'calculated_dk_fpts' not in df.columns:
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    # Feature engineering
    df = engineer_features(df)

    # Clean infinite values
    df = clean_infinite_values(df)

    # Define numeric and categorical features
    numeric_features = [
        'MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
        'PF', 'PTS', 'BPM', 'Name_encoded', 'Team_encoded', 'Opponent_encoded',
        '5_game_avg', 'rolling_mean_fpts', 'lag_mean_fpts_7', 'rolling_mean_fpts_49',
        'opp_recent_avg_pts', 'opp_defensive_strength',
    ]
    

    numeric_features = [f for f in numeric_features if f in df.columns]  # Ensure features exist in df
    categorical_features = ['Player', 'Date', 'Team', 'Opp']
    categorical_features = [f for f in categorical_features if f in df.columns]  # Ensure features exist in df

    # Load or create scaler
    scaler = load_or_create_scaler(df, numeric_features)

    # Create a preprocessor that includes both numeric and categorical transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Feature selection and modeling
    k = min(20, len(numeric_features) + len(categorical_features))
    selector = SelectKBest(f_regression, k=k)

    # Define the polynomial regression model
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    base_models = [
        ('poly', poly_model),  # Replacing linear regression with polynomial regression
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('dt', DecisionTreeRegressor()),
        ('svr', SVR()),
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1)),
        ('bagging', BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10))
    ]

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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', stacking_model)
    ])

    # Train-test split
    train_data, test_data = train_test_split(df, test_size=0.4, shuffle=False)
    features_to_use = [f for f in selected_features if f in df.columns and f != 'calculated_dk_fpts']
    train_features = train_data[features_to_use]
    train_labels = train_data['calculated_dk_fpts']

    # Fit the pipeline
    pipeline.fit(train_features, train_labels)

    # Make predictions on test data
    test_features = test_data[features_to_use]
    test_predictions = pipeline.predict(test_features)

    # Evaluate the model
    mae, mse, r2, mape = evaluate_model(test_data['calculated_dk_fpts'], test_predictions)
    print(f"Model Evaluation Results:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Save the final model
    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/nba_ensemble_model_pipeline_1_2_sep6.pkl')
    print("Final model pipeline saved.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")

    # Load the schedule data
    schedule_df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/schedual.csv', parse_dates=['Date'])

    # Print the column names to verify
    print("Schedule DataFrame columns:", schedule_df.columns)

    # Ensure the date format is consistent
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'], errors='coerce')

    # Adjust column names based on your CSV file
    schedule_df['is_home'] = schedule_df['Home/Neutral'] == schedule_df['Visitor/Neutral']
    schedule_df['opponent'] = np.where(schedule_df['is_home'], schedule_df['Visitor/Neutral'], schedule_df['Home/Neutral'])

    # Merge schedule data with existing data
    df = df.merge(schedule_df, on='Date', how='left', suffixes=('', '_schedule'))

    # Feature engineering with schedule data
    df['is_home'] = df['Home/Neutral'] == df['Team']
    df['opponent'] = np.where(df['is_home'], df['Visitor/Neutral'], df['Home/Neutral'])

    # Example: Calculate opponent's average points in the last 5 games
    df['opp_recent_avg_pts'] = df.groupby('opponent')['PTS'].transform(lambda x: x.rolling(5, min_periods=1).mean())









