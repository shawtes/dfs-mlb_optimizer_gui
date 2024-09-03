#new_aug29th_ensemble_library.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import multiprocessing
import os

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Define paths for saving the label encoders
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/name_encoder.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/team_encoder.pkl'

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

def load_or_create_label_encoders(df):
    if os.path.exists(name_encoder_path) and os.path.exists(team_encoder_path):
        le_name = joblib.load(name_encoder_path)
        le_team = joblib.load(team_encoder_path)
    else:
        le_name = LabelEncoder()
        le_team = LabelEncoder()
        le_name.fit(df['Name'])
        le_team.fit(df['Team'])
        joblib.dump(le_name, name_encoder_path)
        joblib.dump(le_team, team_encoder_path)
    return le_name, le_team

def load_or_create_scaler(df, numeric_features):
    scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        joblib.dump(scaler, scaler_path)
    return scaler

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
            
            # Use all available data, up to 45 most recent games
            player_df = player_df.head(45)
            
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
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

def process_predictions(chunk, pipeline):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    # Clean the features to ensure no infinite or excessively large values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    
    # Debug: Print before adjustment
    print("Before adjustment:", chunk[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    # Adjust predictions to be within the range of 5-game average ± 4
    chunk['predicted_dk_fpts'] = chunk.apply(
        lambda row: max(row['5_game_avg'] - 4, min(row['predicted_dk_fpts'], row['5_game_avg'] + 4)), axis=1
    )
    
    # Debug: Print after adjustment
    print("After adjustment:", chunk[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, train_data['Name'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Model evaluation completed.")
    return mae, mse, r2, mape

def process_fold(fold_data):
    fold, (train_index, test_index), X, y, date_series, numeric_features, categorical_features, final_model = fold_data
    print(f"Processing fold {fold}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    date_train, date_test = date_series.iloc[train_index], date_series.iloc[test_index]

    # Ensure 'calculated_dk_fpts' is included in the training and testing sets
    X_train['calculated_dk_fpts'] = y_train
    X_test['calculated_dk_fpts'] = y_test

    # Engineer features separately for train and test sets
    X_train = engineer_features(X_train.copy(), date_train)
    X_test = engineer_features(X_test.copy(), date_test)

    # Debug: Print 5-game average in test set
    print("5-game average in test set:", X_test[['Name', 'date', '5_game_avg']].head(10))

    # Drop the target column before training
    X_train = X_train.drop(columns=['calculated_dk_fpts'])
    X_test = X_test.drop(columns=['calculated_dk_fpts'])

    # Clean infinite values
    X_train = clean_infinite_values(X_train)
    X_test = clean_infinite_values(X_test)

    # Prepare preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
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

    # Fit preprocessor on training data and transform both train and test
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Feature selection
    selector = SelectKBest(f_regression, k=min(25, X_train_preprocessed.shape[1]))
    X_train_selected = selector.fit_transform(X_train_preprocessed, y_train)
    X_test_selected = selector.transform(X_test_preprocessed)

    # Prepare and fit the model
    model = final_model  # Your stacking model
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Debug: Print before adjustment
    print("Before adjustment:", X_test[['Name', '5_game_avg']].head())
    print("Predictions before adjustment:", y_pred[:5])
    
    # Adjust predictions to be within the range of 5-game average ± 4
    X_test['predicted_dk_fpts'] = y_pred
    X_test['predicted_dk_fpts'] = X_test.apply(
        lambda row: max(row['5_game_avg'] - 4, min(row['predicted_dk_fpts'], row['5_game_avg'] + 4)), axis=1
    )
    
    # Debug: Print after adjustment
    print("After adjustment:", X_test[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    # Evaluate the model
    mae, mse, r2, mape = evaluate_model(y_test, X_test['predicted_dk_fpts'])

    # Create a DataFrame with predictions, actual values, names, and dates
    results_df = pd.DataFrame({
        'Name': X.iloc[test_index]['Name'],
        'Date': date_test,
        'Actual': y_test,
        'Predicted': X_test['predicted_dk_fpts']
    })

    return mae, mse, r2, mape, results_df

# Define final_model outside of the main block
base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('svr', SVR()),
    ('gb', GradientBoostingRegressor())
]

meta_model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

# Voting Regressor
voting_model = VotingRegressor(
    estimators=base_models
)

# Combine all models into a final ensemble pipeline
ensemble_models = [
    ('stacking', stacking_model),
    ('voting', voting_model)
]

final_model = StackingRegressor(
    estimators=ensemble_models,
    final_estimator=XGBRegressor(objective='reg:squarederror', n_jobs=-1)
)

def clean_infinite_values(df):
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, replace NaN with the mean of the column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # For non-numeric columns, replace NaN with a placeholder value (e.g., 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

# Define paths for saving and loading LabelEncoders and Scalers
name_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_name_sep2.pkl'
team_encoder_path = '/Users/sineshawmesfintesfaye/newenv/label_encoder_team_sep2.pkl'
scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler_sep2.pkl'

def load_or_create_label_encoders(df):
    if os.path.exists(name_encoder_path) and os.path.exists(team_encoder_path):
        le_name = joblib.load(name_encoder_path)
        le_team = joblib.load(team_encoder_path)
    else:
        le_name = LabelEncoder()
        le_team = LabelEncoder()
        le_name.fit(df['Name'])
        le_team.fit(df['Team'])
        joblib.dump(le_name, name_encoder_path)
        joblib.dump(le_team, team_encoder_path)
    return le_name, le_team

def load_or_create_scaler(df, numeric_features):
    scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        joblib.dump(scaler, scaler_path)
    return scaler

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
            
            # Use all available data, up to 45 most recent games
            player_df = player_df.head(20)
            
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
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

def process_predictions(chunk, pipeline):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    # Clean the features to ensure no infinite or excessively large values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    
    # Debug: Print before adjustment
    print("Before adjustment:", chunk[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    # Adjust predictions to be within the range of 5-game average ± 4
    chunk['predicted_dk_fpts'] = chunk.apply(
        lambda row: max(row['5_game_avg'] - 4, min(row['predicted_dk_fpts'], row['5_game_avg'] + 4)), axis=1
    )
    
    # Debug: Print after adjustment
    print("After adjustment:", chunk[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, train_data['Name'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Model evaluation completed.")
    return mae, mse, r2, mape

def process_fold(fold_data):
    fold, (train_index, test_index), X, y, date_series, numeric_features, categorical_features, final_model = fold_data
    print(f"Processing fold {fold}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    date_train, date_test = date_series.iloc[train_index], date_series.iloc[test_index]

    # Ensure 'calculated_dk_fpts' is included in the training and testing sets
    X_train['calculated_dk_fpts'] = y_train
    X_test['calculated_dk_fpts'] = y_test

    # Engineer features separately for train and test sets
    X_train = engineer_features(X_train.copy(), date_train)
    X_test = engineer_features(X_test.copy(), date_test)

    # Debug: Print 5-game average in test set
    print("5-game average in test set:", X_test[['Name', 'date', '5_game_avg']].head(10))

    # Drop the target column before training
    X_train = X_train.drop(columns=['calculated_dk_fpts'])
    X_test = X_test.drop(columns=['calculated_dk_fpts'])

    # Clean infinite values
    X_train = clean_infinite_values(X_train)
    X_test = clean_infinite_values(X_test)

    # Prepare preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
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

    # Fit preprocessor on training data and transform both train and test
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Feature selection
    selector = SelectKBest(f_regression, k=min(25, X_train_preprocessed.shape[1]))
    X_train_selected = selector.fit_transform(X_train_preprocessed, y_train)
    X_test_selected = selector.transform(X_test_preprocessed)

    # Prepare and fit the model
    model = final_model  # Your stacking model
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Debug: Print before adjustment
    print("Before adjustment:", X_test[['Name', '5_game_avg']].head())
    print("Predictions before adjustment:", y_pred[:5])
    
    # Adjust predictions to be within the range of 5-game average ± 4
    X_test['predicted_dk_fpts'] = y_pred
    X_test['predicted_dk_fpts'] = X_test.apply(
        lambda row: max(row['5_game_avg'] - 4, min(row['predicted_dk_fpts'], row['5_game_avg'] + 4)), axis=1
    )
    
    # Debug: Print after adjustment
    print("After adjustment:", X_test[['Name', 'predicted_dk_fpts', '5_game_avg']].head())
    
    # Evaluate the model
    mae, mse, r2, mape = evaluate_model(y_test, X_test['predicted_dk_fpts'])
    
    # Create a DataFrame with predictions, actual values, names, and dates
    results_df = pd.DataFrame({
        'Name': X.iloc[test_index]['Name'],
        'Date': date_test,
        'Actual': y_test,
        'Predicted': X_test['predicted_dk_fpts']
    })

    return mae, mse, r2, mape, results_df

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/fangraphs-leaderboards_full_sep1_full_list_batters_traning_mini.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Load or create LabelEncoders
    le_name, le_team = load_or_create_label_encoders(df)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    chunksize = 20000
    df = concurrent_feature_engineering(df, chunksize)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Define the list of all selected and engineered features
    features = selected_features + ['date']

    # Define numeric and categorical features
    numeric_features = [
        'wOBA', 'BABIP', 'ISO',  'wRAA', 'wRC', 'wRC+', 'flyBalls', 'year', 
        'month', 'day',
        'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
        'rolling_mean_fpts_49', 
        'wOBA_Statcast',
        'SLG_Statcast', 'RAR_Statcast', 'Offense_Statcast', 'Dollars_Statcast',
        'WPA/LI_Statcast', 'Off', 'WAR', 'Dol', 'RAR',    
        'RE24', 'REW', 'SLG', 'WPA/LI','AB'
    ]

    categorical_features = ['Name', 'Team']

    # Debug prints to check feature lists and data types
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Data types in DataFrame:")
    print(df.dtypes)

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
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Clean the data before fitting the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Before fitting the preprocessor
    print("Preparing features for preprocessing...")
    features = df[numeric_features + categorical_features]

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
    k = min(25, n_features)  # Select the minimum of 25 or the actual number of features

    selector = SelectKBest(f_regression, k=k)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    # Time series split
    print("Performing time series split...")
    tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits

    features = df.drop(columns=['calculated_dk_fpts'])
    target = df['calculated_dk_fpts']
    date_series = df['date']

    fold_data = [
        (fold, split, features, target, date_series, numeric_features, categorical_features, final_model) 
        for fold, split in enumerate(tscv.split(features), 1)
    ]

    # In the main block, before the concurrent processing:
    print("Cleaning infinite values in the dataset...")
    df = clean_infinite_values(df)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_fold, fold_data))

    mae_scores, mse_scores, r2_scores, mape_scores = [], [], [], []
    for mae, mse, r2, mape, results_df in results:
        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)
        mape_scores.append(mape)
        fold = len(mae_scores)
        results_df.to_csv(f'/Users/sineshawmesfintesfaye/newenv/fold_{fold}_predictions.csv', index=False)
        print(f"Predictions for fold {fold} saved.")

    # Calculate average scores
    avg_mae = sum(mae_scores) / len(mae_scores)
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_mape = sum(mape_scores) / len(mape_scores)

    print(f'Average MAE across folds: {avg_mae}')
    print(f'Average MSE across folds: {avg_mse}')
    print(f'Average R2 across folds: {avg_r2}')
    print(f'Average MAPE across folds: {avg_mape}')

    # Fit the pipeline with the entire dataset before rolling predictions
    print("Fitting the pipeline with the entire dataset...")
    pipeline.fit(features, target)

    # Generate rolling predictions for test dates
    test_dates = df['date'].unique()[-10:]  # Example: last 10 unique dates for testing
    rolling_preds = rolling_predictions(df, pipeline, test_dates, chunksize)

    # Evaluate rolling predictions
    y_true = rolling_preds['calculated_dk_fpts']
    y_pred = rolling_preds['predicted_dk_fpts']
    mae, mse, r2, mape = evaluate_model(y_true, y_pred)

    print(f'Rolling predictions MAE: {mae}')
    print(f'Rolling predictions MSE: {mse}')
    print(f'Rolling predictions R2: {r2}')
    print(f'Rolling predictions MAPE: {mape}')

    # Save the final model
    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/batters_final_ensemble_model_pipeline.pkl')
    print("Final model pipeline saved.")

    # Save the final data to a CSV file
    df.to_csv('/Users/sineshawmesfintesfaye/newenv/battersfinal_dataset_with_features.csv', index=False)
    print("Final dataset with all features saved.")

    # Save the LabelEncoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("LabelEncoders saved.")

    save_feature_importance(pipeline, '/Users/sineshawmesfintesfaye/newenv/feature_importances.csv', '/Users/sineshawmesfintesfaye/newenv/feature_importances_plot.png')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")
