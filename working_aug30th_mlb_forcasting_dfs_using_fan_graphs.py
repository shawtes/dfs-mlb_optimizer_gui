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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, BaggingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing

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

# Top 20 features based on user's selection (ensuring uniqueness)
# selected_features = list(set([
#     'Off', 'WAR', 'Dol', 'RAR', 'wRC', 'wRAA', 'Bat', 'OPS', 'wOBA', 'RE24',
#     'REW', 'SLG', 'WPA/LI', 'SLG_Statcast', 'RAR_Statcast',
#     'Dollars_Statcast',
#     # 'interaction_team_opponent',
#     # 'Name_encoded', 'opponent_encoded', 
#     # 'team_encoded', 
#     # 'interaction_name_opponent', 'interaction_name_team', 
#     'WPA/LI_Statcast', 'rolling_mean_fpts_7'
# ])) '1B', '2B', '3B', 'HR', 'BB', 'IBB', 'HBP', 'SF', 'SH', 'ROE', 'GIDP','rolling_mean_fpts_7',
    # 'rolling_mean_fpts_49', 'lag_mean_fpts_3', 'lag_max_fpts_3', 'rolling_max_fpts_7',  'rolling_min_fpts_7'
selected_features = [
     'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+', 
    'flyBalls', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
    # 'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7', 'rolling_mean_fpts_49',
    'singles', 'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI','AB', 'WAR'  
    # 'lag_mean_fpts_3', 'lag_max_fpts_3', 'lag_min_fpts_3',
    # 'lag_mean_fpts_7', 'lag_max_fpts_7', 'lag_min_fpts_7', 'lag_mean_fpts_14', 'lag_max_fpts_14', 'lag_min_fpts_14',
    # 'lag_mean_fpts_28', 'lag_max_fpts_28', 'lag_min_fpts_28'
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

# # Initial list of selected features based on importance
# selected_features = [
#     'Off', 'WAR', 'Dol', 'RAR', 'wRC', 'wRAA', 'Bat', 'wRC+',
#     'wOBA', 'RE24', 'REW', 'SLG', 'WPA/LI'
# ]

# # Add new engineered features to selected_features 
# engineered_features = [
#     'wOBA_Statcast', 
#     'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
#     'WPA/LI_Statcast'
# ]
# selected_features += engineered_features

# # Ensure no duplicates in selected_features
# selected_features = list(set(selected_features))

# def calculate_statcast_metrics(df):
#     new_columns = {}
#     df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
#     new_columns['wOBA_Statcast'] = (
#         (wOBA_weights_2020['uBB'] * df['BB']) + 
#         (wOBA_weights_2020['HBP'] * df['HBP']) + 
#         (wOBA_weights_2020['1B'] * df['1B']) + 
#         (wOBA_weights_2020['2B'] * df['2B']) + 
#         (wOBA_weights_2020['3B'] * df['3B']) + 
#         (wOBA_weights_2020['HR'] * df['HR'])
#     ) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])

#     new_columns['SLG_Statcast'] = (
#         df['1B'] + (2 * df['2B']) + (3 * df['3B']) + (4 * df['HR'])
#     ) / df['AB']

#     new_columns['Offense_Statcast'] = df['wRAA'] + df['BsR'] + (df['wOBA'] - df['xwOBA'])
#     new_columns['RAR_Statcast'] = df['WAR'] * 10
#     new_columns['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor
#     new_columns['WPA/LI_Statcast'] = df['WPA/LI']

#     # Ensure all columns are 1-dimensional
#     for key, value in new_columns.items():
#         if isinstance(value, pd.DataFrame):
#             new_columns[key] = value.iloc[:, 0]

#     return pd.concat([df, pd.DataFrame(new_columns)], axis=1)

# def engineer_features(df):
#     print("Starting feature engineering...")
#     new_columns = {}
    
#     if not pd.api.types.is_datetime64_any_dtype(df['date']):
#         df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), errors='coerce')
    
#     new_columns['year'] = df['date'].dt.year
#     new_columns['month'] = df['date'].dt.month
#     new_columns['day'] = df['date'].dt.day
#     new_columns['day_of_week'] = df['date'].dt.dayofweek
#     new_columns['day_of_season'] = (df['date'] - df['date'].min()).dt.days
    
#     for lag in range(1, 50):
#         new_columns[f'lag_{lag}'] = df.groupby('Name')['calculated_dk_fpts'].shift(lag)
    
#     new_columns['rolling_min_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).min())
#     new_columns['rolling_max_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).max())
#     new_columns['rolling_mean_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).mean())
#     new_columns['rolling_mean_fpts_49'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).mean())
    
#     for lag in range(1, 51):
#         new_columns[f'lag_HR_{lag}'] = df.groupby('Name')['HR'].shift(lag)
#         new_columns[f'lag_RBI_{lag}'] = df.groupby('Name')['RBI'].shift(lag)
#         new_columns[f'lag_R_{lag}'] = df.groupby('Name')['R'].shift(lag)

#     new_columns['OPS'] = df['OBP'] + df['SLG']

#     ema_features = ['AVG', 'OBP', 'SLG']
#     for feature in ema_features:
#         new_columns[f'ema_{feature}_30'] = df.groupby('Name')[feature].transform(lambda x: x.ewm(span=30).mean())

#     new_columns['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
    
#     # Ensure all columns are 1-dimensional
#     for key, value in new_columns.items():
#         if isinstance(value, pd.DataFrame):
#             new_columns[key] = value.iloc[:, 0]
    
#     # Add all new columns at once
#     df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
#     df.fillna(0, inplace=True)
#     df = calculate_statcast_metrics(df)
#     print("Feature engineering completed.")

#     return df
# # Feature Engineering function
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
    df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + df['SF'])
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA[x['year']]) / 1.15) * x['AB'], axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA[x['year']] * 100) if x['AB'] > 0 else 0, axis=1)

    df['flyBalls'] = df['HR'] / df.apply(lambda x: league_avg_HR_FlyBall[x['year']], axis=1)

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
    ) / (x['AB'] + x['BB'] - x['IBB'] + x['SF'] + x['HBP']), axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = (
        df['1B'] + (2 * df['2B']) + (3 * df['3B']) + (4 * df['HR'])
    ) / df['AB']

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

    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def process_chunk(chunk, date_series=None):
    return engineer_features(chunk, date_series)

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        processed_chunks = list(executor.map(process_chunk, chunks, [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows(df, current_date):
    print(f"Creating synthetic rows for date: {current_date}...")
    synthetic_rows = []
    for player_id in df['Name'].unique():
        player_df = df[df['Name'] == player_id].tail(45)
        if player_df.empty:
            continue
        numeric_averages = player_df.mean(numeric_only=True)
        synthetic_row = pd.DataFrame([numeric_averages], columns=player_df.columns)
        synthetic_row['date'] = current_date
        synthetic_row['Name'] = player_df['Name'].iloc[0]
        for col in player_df.select_dtypes(include=['object']).columns:
            if col not in ['date', 'Name']:
                synthetic_row[col] = player_df[col].mode()[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
        synthetic_rows.append(synthetic_row)
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {current_date}.")
    return synthetic_df

def process_predictions(chunk, pipeline):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    # Clean the features to ensure no infinite or excessively large values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows(train_data, current_date)
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

def save_feature_importance(pipeline, output_csv_path, output_plot_path):
    print("Saving feature importances...")
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    if hasattr(model.final_estimator_, 'feature_importances_'):
        feature_importances = model.final_estimator_.feature_importances_
    else:
        raise ValueError("Model does not have feature importances.")
    
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names = np.concatenate([numeric_features, cat_features])
    
    if len(feature_importances) != len(feature_names):
        raise ValueError("The number of feature importances does not match the number of feature names.")
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
   
    feature_importance_df.to_csv(output_csv_path, index=False)
    print(f"Feature importances saved to {output_csv_path}")

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importances)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()
    print(f"Feature importance plot saved to {output_plot_path}")

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

def process_fold(fold_data):
    fold, (train_index, test_index), X, y, date_series, numeric_features, categorical_features, final_model = fold_data
    print(f"Processing fold {fold}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    date_train, date_test = date_series.iloc[train_index], date_series.iloc[test_index]

    X_train = engineer_features(X_train.copy(), date_train)
    X_test = engineer_features(X_test.copy(), date_test)

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

    # Evaluate the model
    mae, mse, r2, mape = evaluate_model(y_test, y_pred)
    
    # Create a DataFrame with predictions, actual values, names, and dates
    results_df = pd.DataFrame({
        'Name': X.iloc[test_index]['Name'],
        'Date': date_test,
        'Actual': y_test,
        'Predicted': y_pred
    })

    return mae, mse, r2, mape, results_df

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/aug29_merged_aug30th.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Ensure 'Name_encoded' column is created
    le_name = LabelEncoder()
    df['Name_encoded'] = le_name.fit_transform(df['Name'])

    # Ensure 'Team_encoded' column is created (if it doesn't exist already)
    if 'Team_encoded' not in df.columns:
        le_team = LabelEncoder()
        df['Team_encoded'] = le_team.fit_transform(df['Team'])

    # Update categorical_features list
    categorical_features = ['Name_encoded', 'Team_encoded']

    chunksize = 20000
    df = concurrent_feature_engineering(df, chunksize)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)


    # Define the list of all selected and engineered features
#     selected_features = [
#      'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+', 
#     'flyBalls', 'xFIP', 'SIERA', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
#     'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7', 'rolling_mean_fpts_49',
#     'singles', 'wOBA_Statcast', 'SLG_Statcast', 'lag_mean_fpts_3', 'lag_max_fpts_3', 'lag_min_fpts_3',
#     'lag_mean_fpts_7', 'lag_max_fpts_7', 'lag_min_fpts_7', 'lag_mean_fpts_14', 'lag_max_fpts_14', 'lag_min_fpts_14',
#     'lag_mean_fpts_28', 'lag_max_fpts_28', 'lag_min_fpts_28'
# ]

    # Add features from engineered columns
    features = selected_features + [
        # 'calculated_dk_fpts', 
        'date']

    # # Ensure that categorical features are correctly identified
    df['team_encoded'] = df['Team']
    # df['player_encoded'] = df['Name']
    df['Name_encoded'] = df['Name']
    # df['opponent_encoded'] = df['opponent']
    

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
        # 'lag_mean_fpts_3', 'lag_max_fpts_3', 'lag_min_fpts_3', 'lag_mean_fpts_7', 'lag_max_fpts_7', 'lag_min_fpts_7',
        # 'lag_mean_fpts_14', 'lag_max_fpts_14', 'lag_min_fpts_14', 'lag_mean_fpts_28', 'lag_max_fpts_28', 'lag_min_fpts_28'
    ]

    categorical_features = ['Name_encoded','team_encoded'
       
    ]


    # Define transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor that includes both numeric and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer,categorical_features)
        ])

    # Clean the data before fitting the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Before fitting the preprocessor
    print("Preparing features for preprocessing...")
    features = df[numeric_features + categorical_features]

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

    # Train the final model on all data
    print("Training final model on all data...")
    pipeline.fit(features, target)

    # Make predictions on the entire dataset
    all_predictions = pipeline.predict(features)

    # Create a DataFrame with all predictions, actual values, names, and dates
    final_results_df = pd.DataFrame({
        'Name': df['Name'],
        'Date': df['date'],
        'Actual': target,
        'Predicted': all_predictions
    })

    # Save the final results
    final_results_df.to_csv('/Users/sineshawmesfintesfaye/newenv/final_predictions.csv', index=False)
    print("Final predictions saved.")

    # Save the final model
    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/final_ensemble_model_pipeline.pkl')
    print("Final model pipeline saved.")

    # Save the final data to a CSV file
    df.to_csv('/Users/sineshawmesfintesfaye/newenv/final_dataset_with_features.csv', index=False)
    print("Final dataset with all features saved.")

    save_feature_importance(pipeline, '/Users/sineshawmesfintesfaye/newenv/feature_importances.csv', '/Users/sineshawmesfintesfaye/newenv/feature_importances_plot.png')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")
