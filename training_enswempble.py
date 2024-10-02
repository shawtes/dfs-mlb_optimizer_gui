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

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Define paths for saving the label encoders and scaler
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/name_encoder_1_sep4.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/team_encoder_1_sep4.pkl'
opponent_encoder_path = '/Users/sineshawmesfintesfaye/newenv/opponent_encoder_1_sep4.pkl'
scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler_1_sep4.pkl'  # Use this path consistently

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
    
    'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI','AB', 'WAR'  
]
#  df['Name_Opponent_Interaction'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
#     df['Name_Team_Interaction'] = df['Name_encoded'] * df['Team_encoded']
#     df['Team_Opponent_Interaction'] = df['Team_encoded'] * df['Opponent_encoded']
#     df['Date_Team_Interaction'] = df['year'].astype(str) + '_' + df['Team_encoded'].astype(str)
    

engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast', 'Name_encoded', 'team_encoded',
    'interaction_name_opponent','interaction_name_team', 'interaction_name_date',
    'Name_Opponent_Interaction','Team_Opponent_Interaction','Name_Team_Interaction','Date_Team_Interaction'
]
selected_features += engineered_features

def load_or_create_label_encoders(df):
    le_name = LabelEncoder()
    le_team = LabelEncoder()
    
    if 'Opponent' in df.columns:
        if os.path.exists(opponent_encoder_path):
            le_opponent = joblib.load(opponent_encoder_path)
            # Add unknown class if not present
            if 'Unknown' not in le_opponent.classes_:
                le_opponent.classes_ = np.append(le_opponent.classes_, 'Unknown')
        else:
            le_opponent = LabelEncoder()
        # Fill NaN values with 'Unknown' before fitting
        df['Opponent'] = df['Opponent'].fillna('Unknown')
        # Fit with all unique values including 'Unknown'
        le_opponent.fit(df['Opponent'].unique())
        joblib.dump(le_opponent, opponent_encoder_path)
    else:
        print("Warning: 'Opponent' column not found. Skipping opponent encoding.")
        le_opponent = None

    # Fill NaN values for Name and Team as well
    df['Name'] = df['Name'].fillna('Unknown')
    df['Team'] = df['Team'].fillna('Unknown')

    le_name.fit(df['Name'].unique())
    le_team.fit(df['Team'].unique())
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    
    return le_name, le_team, le_opponent

def load_or_create_scaler(df, numeric_features):
    if os.path.exists(scaler_path):  # Use the consistent path
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        joblib.dump(scaler, scaler_path)  # Use the consistent path
    return scaler

def add_teammate_interactions(df):
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Step 1: Sort the data by team, player, and date for clarity
    df = df.sort_values(by=['Team', 'Name', 'date'])

    # Step 2: Group the data by 'Team' and 'date' to aggregate team-level statistics
    team_stats = df.groupby(['Team', 'date']).agg({
        'calculated_dk_fpts': 'sum',  # Total fantasy points scored by the team
        'AB': 'sum',  # Total at-bats for the team
        'HR': 'sum',  # Total home runs for the team
        'H': 'sum',   # Total hits for the team
    }).reset_index()

    # Step 3: Merge the team statistics back into the original dataframe
    df = pd.merge(df, team_stats, on=['Team', 'date'], suffixes=('', '_team'))

    # Step 4: Calculate interaction terms based on teammates' stats
    df['teammate_dk_fpts'] = df['calculated_dk_fpts_team'] - df['calculated_dk_fpts']
    df['teammate_AB'] = df['AB_team'] - df['AB']
    df['teammate_HR'] = df['HR_team'] - df['HR']
    df['teammate_H'] = df['H_team'] - df['H']

    # Step 5: Add additional interaction terms for teammate performance and player performance
    df['interaction_teammate_fpts'] = df['calculated_dk_fpts'] * df['teammate_dk_fpts']
    df['interaction_teammate_AB'] = df['AB'] * df['teammate_AB']
    df['interaction_teammate_HR'] = df['HR'] * df['teammate_HR']

    # Step 6: Create more complex interaction terms for player, team, and date
    df['player_team_date_interaction'] = df['Name_encoded'].astype(str) + '_' + df['Team_encoded'].astype(str) + '_' + df['date'].astype(str)

    return df

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
    df['week_of_season'] = (date_series - date_series.min()).dt.days // 7
    df['day_of_year'] = date_series.dt.dayofyear

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
    
    # Player consistency features
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    # Interaction terms
    df['Name_Opponent_Interaction'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    df['Name_Team_Interaction'] = df['Name_encoded'] * df['Team_encoded']
    df['Team_Opponent_Interaction'] = df['Team_encoded'] * df['Opponent_encoded']
    df['Date_Team_Interaction'] = df['year'].astype(str) + '_' + df['Team_encoded'].astype(str)
    
    # Calculate rolling averages for trends
    df['rolling_mean_fpts'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['rolling_mean_fpts_opponent'] = df.groupby(['Name', 'Opponent'])['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    
    # Assuming 'full_name_encoded' is the encoded version of the player's name
    df['interaction_name_date'] = df['Name_encoded'].astype(str) + '_' + df['date'].astype(str)
    df['interaction_name_team'] = df['Name_encoded'].astype(str) + '_' + df['Team_encoded'].astype(str)
    df['interaction_name_opponent'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    
    # Add teammate interactions
    df = add_teammate_interactions(df)

    return df

def engineer_chunk(chunk):
    return engineer_features(chunk)

def concurrent_feature_engineering(df, chunksize):
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        processed_chunks = list(executor.map(engineer_chunk, chunks))
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
            
            # Use all available data, up to 20 most recent games
            player_df = player_df.head(15)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and calculated correctly
            if 'calculated_dk_fpts' in player_df.columns:
                synthetic_row['calculated_dk_fpts'] = player_df['calculated_dk_fpts'].mean()
            else:
                synthetic_row['calculated_dk_fpts'] = np.nan  # Placeholder, replace with actual calculation if needed
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
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
    # Sort the DataFrame by 'Name' and 'date' to ensure the last 5 games are correctly identified
    df = df.sort_values(by=['Name', 'date'])
    
    # Create a new column to identify the last 5 games for each player
    df['last_5_games'] = df.groupby('Name')['date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=5)))

    # Filter the DataFrame to include only the last 5 games
    last_5_games_df = df[df['last_5_games']]

    # Calculate the difference
    last_5_games_df['difference'] = last_5_games_df['predicted_dk_fpts'] - last_5_games_df['calculated_dk_fpts']
    
    player_adjustments = last_5_games_df.groupby('Name').agg({
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

def adjust_predictions(row, player_adjustments):
    """Adjust predictions based on player-specific average differences."""
    prediction = row['predicted_dk_fpts']
    player = row['Name']
    
    if player in player_adjustments.index:
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_positive_diff'] )
        else:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_negative_diff'] )
    else:
        # If no player-specific adjustment, use overall average
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments['avg_positive_diff'].mean())
        else:
            adjusted_prediction = prediction - (player_adjustments['avg_negative_diff'].mean())
    
    return max(0, adjusted_prediction)  # Ensure non-negative prediction

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
        train_data_filtered = train_data[train_data['date'] < current_date]  # Exclude data on or after the current date
        synthetic_rows = create_synthetic_rows_for_all_players(train_data_filtered, train_data_filtered['Name'].unique(), current_date)
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
    dates = df['date'].sort_values().unique()
    
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
        
        train_data = df[df['date'].isin(train_dates)]
        test_data = df[df['date'] == test_date]
        
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

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/merged_opponents_mlb.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['Name', 'date'], inplace=True)

    # After loading the dataset
    print("Checking for NaN values in the dataset...")
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with NaN values: {nan_columns}")
        print("NaN value counts:")
        print(df[nan_columns].isna().sum())
    else:
        print("No NaN values found in the dataset.")

    # Handle NaN values for all object columns
    for col in df.select_dtypes(include=['object']).columns:
        print(f"Handling NaN values in '{col}' column...")
        df[col] = df[col].fillna('Unknown')

    # Load or create LabelEncoders
    le_name, le_team, le_opponent = load_or_create_label_encoders(df)

    # Encoding categorical columns
    print("Encoding categorical columns...")
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    if le_opponent is not None:
        df['Opponent_encoded'] = le_opponent.transform(df['Opponent'])

    # Debug print
    print("Encoded columns created successfully.")
    print(df[['Name', 'Name_encoded', 'Team', 'Team_encoded', 'Opponent', 'Opponent_encoded']].head())

    chunksize = 20000

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
    df['week_of_season'] = (date_series - date_series.min()).dt.days // 7
    df['day_of_year'] = date_series.dt.dayofyear

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
    
    # Player consistency features
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    # Interaction terms
    df['Name_Opponent_Interaction'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    df['Name_Team_Interaction'] = df['Name_encoded'] * df['Team_encoded']
    df['Team_Opponent_Interaction'] = df['Team_encoded'] * df['Opponent_encoded']
    df['Date_Team_Interaction'] = df['year'].astype(str) + '_' + df['Team_encoded'].astype(str)
    
    # Calculate rolling averages for trends
    df['rolling_mean_fpts'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['rolling_mean_fpts_opponent'] = df.groupby(['Name', 'Opponent'])['calculated_dk_fpts'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    
    # Assuming 'full_name_encoded' is the encoded version of the player's name
    df['interaction_name_date'] = df['Name_encoded'].astype(str) + '_' + df['date'].astype(str)
    df['interaction_name_team'] = df['Name_encoded'].astype(str) + '_' + df['Team_encoded'].astype(str)
    df['interaction_name_opponent'] = df['Name_encoded'].astype(str) + '_' + df['Opponent_encoded'].astype(str)
    
    # Add teammate interactions
    df = add_teammate_interactions(df)

    return df

def engineer_chunk(chunk):
    return engineer_features(chunk)

def concurrent_feature_engineering(df, chunksize):
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        processed_chunks = list(executor.map(engineer_chunk, chunks))
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
            
            # Use all available data, up to 20 most recent games
            player_df = player_df.head(15)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and calculated correctly
            if 'calculated_dk_fpts' in player_df.columns:
                synthetic_row['calculated_dk_fpts'] = player_df['calculated_dk_fpts'].mean()
            else:
                synthetic_row['calculated_dk_fpts'] = np.nan  # Placeholder, replace with actual calculation if needed
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
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
    # Sort the DataFrame by 'Name' and 'date' to ensure the last 5 games are correctly identified
    df = df.sort_values(by=['Name', 'date'])
    
    # Create a new column to identify the last 5 games for each player
    df['last_5_games'] = df.groupby('Name')['date'].transform(lambda x: x >= (x.max() - pd.Timedelta(days=5)))

    # Filter the DataFrame to include only the last 5 games
    last_5_games_df = df[df['last_5_games']]

    # Calculate the difference
    last_5_games_df['difference'] = last_5_games_df['predicted_dk_fpts'] - last_5_games_df['calculated_dk_fpts']
    
    player_adjustments = last_5_games_df.groupby('Name').agg({
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

def adjust_predictions(row, player_adjustments):
    """Adjust predictions based on player-specific average differences."""
    prediction = row['predicted_dk_fpts']
    player = row['Name']
    
        if player in player_adjustments.index:
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_positive_diff'] )
        else:
            adjusted_prediction = prediction - (player_adjustments.loc[player, 'avg_negative_diff'] )
    else:
        # If no player-specific adjustment, use overall average
        if prediction > row['calculated_dk_fpts']:
            adjusted_prediction = prediction - (player_adjustments['avg_positive_diff'].mean())
        else:
            adjusted_prediction = prediction - (player_adjustments['avg_negative_diff'].mean())
    
    return max(0, adjusted_prediction)  # Ensure non-negative prediction

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
        train_data_filtered = train_data[train_data['date'] < current_date]  # Exclude data on or after the current date
        synthetic_rows = create_synthetic_rows_for_all_players(train_data_filtered, train_data_filtered['Name'].unique(), current_date)
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
    dates = df['date'].sort_values().unique()
    
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
        
        train_data = df[df['date'].isin(train_dates)]
        test_data = df[df['date'] == test_date]
        
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

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/merged_opponents_mlb.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Load or create LabelEncoders
    le_name, le_team, le_opponent = load_or_create_label_encoders(df)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    df['Opponent_encoded'] = le_opponent.transform(df['Opponent'])

    chunksize = 20000
    df = concurrent_feature_engineering(df, chunksize)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Define the list of all selected and engineered features
    features = selected_features + ['date']

    # Define numeric and categorical features
    numeric_features = [
        'wOBA', 'BABIP', 'ISO', 'wRAA', 'wRC', 'wRC+', 'flyBalls', 'year', 
        'month', 'day', 'day_of_week', 'day_of_season', 'wOBA_Statcast',
        'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',    
        'RE24', 'REW', 'SLG', 'WPA/LI', 'AB',
        'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 'WPA/LI_Statcast',
        'week_of_season', 'day_of_year', 'fpts_std', 'fpts_volatility'
    ]

    # Remove duplicates while preserving order
    numeric_features = list(dict.fromkeys(numeric_features))

    categorical_features = ['Name', 'Team','Opponent',
    'interaction_name_opponent','interaction_name_team', 'interaction_name_date',
    'Name_Opponent_Interaction','Team_Opponent_Interaction','Name_Team_Interaction','Date_Team_Interaction'
]

    # Update the list of features to include the new interaction terms
    numeric_features += [
        'teammate_dk_fpts', 'teammate_AB', 'teammate_HR', 'teammate_H',
        'interaction_teammate_fpts', 'interaction_teammate_AB', 'interaction_teammate_HR'
    ]
    categorical_features += ['player_team_date_interaction']

    # Remove duplicates while preserving order
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    all_features = numeric_features + categorical_features
    all_features = list(dict.fromkeys(all_features))

    print("All features after adding teammate interactions:")
    print(all_features)

    # Use all_features instead of numeric_features + categorical_features
    features = df[all_features]

    # Debug prints to check feature lists and data types
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Data types in DataFrame:")
    print(df.dtypes)

    # Clean infinite values in the dataset before fitting the scaler
    print("Cleaning infinite values in the dataset...")
    df = clean_infinite_values(df)

    # Load or create Scaler
    scaler = load_or_create_scaler(df, numeric_features)

    # Define transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor that includes both numeric and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in all_features if col in numeric_features]),
            ('cat', categorical_transformer, [col for col in all_features if col in categorical_features])
        ])

    # Clean the data before fitting the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Before preparing features for preprocessing
    print("Preparing features for preprocessing...")
    features = df[numeric_features + categorical_features]

    # Remove duplicate columns
    features = features.loc[:, ~features.columns.duplicated()]

    # Debug print to check data types in features DataFrame
    print("Data types in features DataFrame before preprocessing:")
    print(features.dtypes)

    # Clean the data before fitting the preprocessor
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    # Fit the preprocessor
    print("Fitting preprocessor...")
    preprocessed_features = preprocessor.fit_transform(features)
    n_features = preprocessed_features.shape[1]

    # Feature selection based on the actual number of features
    k = min(50, n_features)  # Increase from 35 to 50

    selector = SelectKBest(f_regression, k=k)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)  # Use final_model directly instead of grid_search
    ])

    # Clean the data before fitting the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Sort the dataframe by date
    df = df.sort_values('date')

    # Split the data into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

    # Fit the pipeline with the training data
    leaky_features = ['calculated_dk_fpts', '5_game_avg', 'rolling_mean_fpts', 'rolling_mean_fpts_opponent']
    features_to_use = [f for f in all_features if f not in leaky_features]

    train_features = train_data[features_to_use]
    test_features = test_data[features_to_use]

    pipeline.fit(train_features, train_data['calculated_dk_fpts'])

    # Before calling rolling_window_prediction
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data date range: {test_data['date'].min()} to {test_data['date'].max()}")
    print(f"Number of unique dates in test data: {test_data['date'].nunique()}")

    rolling_preds = rolling_window_prediction(test_data, pipeline)

    # After calling rolling_window_prediction
    print(f"Rolling predictions shape: {rolling_preds.shape}")

    # Check if 'predicted_dk_fpts' column exists
    if 'predicted_dk_fpts' not in rolling_preds.columns:
        print("Warning: 'predicted_dk_fpts' column not found. Using 'calculated_dk_fpts' for evaluation.")
        rolling_preds['predicted_dk_fpts'] = rolling_preds['calculated_dk_fpts']

    # Evaluate rolling predictions
    y_true = rolling_preds['calculated_dk_fpts']
    y_pred = rolling_preds['predicted_dk_fpts']
    mae, mse, r2, mape = evaluate_model(y_true, y_pred)

    print(f'Rolling predictions MAE: {mae}')
    print(f'Rolling predictions MSE: {mse}')
    print(f'Rolling predictions R2: {r2}')
    print(f'Rolling predictions MAPE: {mape}')

    # Add a plot to visualize predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual DK Points")
    plt.ylabel("Predicted DK Points")
    plt.title("Actual vs Predicted DK Points")
    plt.savefig('/Users/sineshawmesfintesfaye/newenv/actual_vs_predicted_rolling.png')
    plt.close()

    # Save the final model
    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/batters_final_ensemble_model_pipeline_1_2_sep5.pkl')
    print("Final model pipeline saved.")

    # Save the final data to a CSV file
    df.to_csv('/users/sineshawmesfintesfaye/newenv/api__max_mlb_game_logs_2024_sep_29_1.csv', index=False)
    print("Final dataset with all features saved.")

    # Save the LabelEncoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    joblib.dump(le_opponent, opponent_encoder_path)
    print("LabelEncoders saved.")

    # Save player adjustments
    player_adjustments.to_csv('/Users/sineshawmesfintesfaye/newenv/player_adjustments.csv')
    print("Player-specific adjustments saved.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")

    # After cleaning infinite values and before fitting the preprocessor
    print("Columns in DataFrame:")
    print(df.columns.tolist())

    print("Columns in train_features:")
    print(train_features.columns.tolist())

    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Warning: Duplicate columns found in DataFrame: {duplicate_columns}")

    duplicate_features = train_features.columns[train_features.columns.duplicated()].tolist()
    if duplicate_features:
        print(f"Warning: Duplicate columns found in train_features: {duplicate_features}")

    # Check the new interaction terms
    print("Sample of new interaction terms:")
    print(df[['Name', 'Team', 'date', 'calculated_dk_fpts', 'teammate_dk_fpts', 'interaction_teammate_fpts']].head())
