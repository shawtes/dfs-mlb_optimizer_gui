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
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

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
    2020: {'uBB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'uBB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'uBB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'uBB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'uBB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

# Top 20 features based on user's selection (ensuring uniqueness)
top_20_features = list(set([
    'Off', 'WAR', 'Dol', 'RAR', 'wRC', 'wRAA', 'Bat', 'OPS', 'wOBA', 'RE24',
    'REW', 'SLG', 'WPA/LI', 'wOBA_Statcast', 'SLG_Statcast', 'RAR_Statcast',
    'Dollars_Statcast', 'interaction_name_date', 'interaction_team_opponent',
    'full_name_encoded', 'opponent_encoded', 'team_encoded', 'interaction_name_date',
    'interaction_name_opponent', 'interaction_name_team', 'WPA/LI_Statcast', 'rolling_mean_fpts_7'
]))

# Feature Engineering function
def engineer_features(df):
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_season'] = (df['date'] - df['date'].min()).dt.days

    # Normalize primary positions
    df['primary_position'] = df['primary_position'].fillna('Unknown')
    df['normalized_position'] = df['primary_position'].apply(lambda x: 'P' if 'P' in x else 'B')

    # Create lag features
    for position in ['P', 'B']:
        pos_df = df[df['normalized_position'] == position].copy()
        for lag in range(1, 50):
            pos_df[f'lag_{lag}'] = pos_df.groupby('player_id')['calculated_dk_fpts'].shift(lag)
        lag_columns = [f'lag_{lag}' for lag in range(1, 50)]
        df.loc[pos_df.index, lag_columns] = pos_df[lag_columns]

    # Calculate rolling statistics
    df['rolling_min_fpts_7'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).min())
    df['rolling_max_fpts_7'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).max())
    df['rolling_mean_fpts_7'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_mean_fpts_49'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).mean())

    # Calculate key statistics
    df['wOBA'] = (df['baseOnBalls']*0.69 + df['hitByPitch']*0.72 + (df['hits'] - df['doubles'] - df['triples'] - df['homeRuns'])*0.88 + df['doubles']*1.24 + df['triples']*1.56 + df['homeRuns']*2.08) / (df['atBats'] + df['baseOnBalls'] - df['intentionalWalks'] + df['sacFlies'] + df['hitByPitch'])
    df['BABIP'] = (df['hits'] - df['homeRuns']) / (df['atBats'] - df['strikeOuts'] - df['homeRuns'] + df['sacFlies'])
    df['ISO'] = df['slg'] - df['avg']
    df['FIP'] = ((13*df['homeRuns'] + 3*(df['baseOnBalls'] + df['hitByPitch']) - 2*df['strikeOuts']) / df['inningsPitched']) + 3.1

    # Advanced Sabermetric Metrics
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA[x['year']]) / 1.15) * x['atBats'], axis=1)
    df['wRC'] = df['wRAA'] + (df['atBats'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['atBats'] / league_avg_wOBA[x['year']] * 100) if x['atBats'] > 0 else 0, axis=1)

    df['flyBalls'] = df['homeRuns'] / df.apply(lambda x: league_avg_HR_FlyBall[x['year']], axis=1)
    df['xFIP'] = df.apply(lambda x: ((13 * (x['flyBalls'] * league_avg_HR_FlyBall[x['year']]) + 3 * (x['baseOnBalls'] + x['hitByPitch']) - 2 * x['strikeOuts']) / x['inningsPitched']) + 3.1 if x['inningsPitched'] > 0 else 0, axis=1)
    df['SIERA'] = df.apply(lambda x: (x['strikeOuts'] / x['inningsPitched']) - (x['baseOnBalls'] / x['inningsPitched']) + ((x['baseOnBalls'] / x['inningsPitched']) * (x['strikeOuts'] / x['inningsPitched'])) + 2.9 if x['inningsPitched'] > 0 else 0, axis=1)  # Simplified calculation

    # Calculate WAR if not present in the CSV
    if 'WAR' not in df.columns:
        df['WAR'] = np.random.uniform(0, 10, size=len(df))  # Placeholder or use a proper method to calculate WAR

    # Calculate singles
    df['singles'] = df['hits'] - df['doubles'] - df['triples'] - df['homeRuns']

    # Calculate wOBA using year-specific weights
    df['wOBA_Statcast'] = df.apply(lambda x: (
        (wOBA_weights[x['year']]['uBB'] * x['baseOnBalls']) +
        (wOBA_weights[x['year']]['HBP'] * x['hitByPitch']) +
        (wOBA_weights[x['year']]['1B'] * x['singles']) +
        (wOBA_weights[x['year']]['2B'] * x['doubles']) +
        (wOBA_weights[x['year']]['3B'] * x['triples']) +
        (wOBA_weights[x['year']]['HR'] * x['homeRuns'])
    ) / (x['atBats'] + x['baseOnBalls'] - x['intentionalWalks'] + x['sacFlies'] + x['hitByPitch']) if (x['atBats'] + x['baseOnBalls'] - x['intentionalWalks'] + x['sacFlies'] + x['hitByPitch']) > 0 else 0, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = (
        df['singles'] + (2 * df['doubles']) + (3 * df['triples']) + (4 * df['homeRuns'])
    ) / df['atBats']

    # Calculate RAR_Statcast
    df['RAR_Statcast'] = df['WAR'] * 10

    # df['Offense_Statcast'] = df['wRAA'] + df['BsR'] + (df['wOBA'] - df['xwOBA'])

    # Calculate Dollars_Statcast
    WAR_conversion_factor = 8.0  # Example conversion factor, can be adjusted
    df['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor

    # Calculate WPA/LI_Statcast
    df['WPA/LI_Statcast'] = df['WPA/LI']

    # Label encode the team, opponent, and full_name features
    le_team = LabelEncoder()
    df['team_encoded'] = le_team.fit_transform(df['team'])

    le_opponent = LabelEncoder()
    df['opponent_encoded'] = le_opponent.fit_transform(df['opponent'])

    le_name = LabelEncoder()
    df['full_name_encoded'] = le_name.fit_transform(df['full_name'])

    # Add interaction features
    df['interaction_name_date'] = df['full_name_encoded'].astype(str) + '_' + df['date'].astype(str)
    df['interaction_name_team'] = df['full_name_encoded'].astype(str) + '_' + df['team_encoded'].astype(str)
    df['interaction_name_opponent'] = df['full_name_encoded'].astype(str) + '_' + df['opponent_encoded'].astype(str)
    
    for window in [3, 7, 14, 28]:
        df[f'lag_mean_fpts_{window}'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'lag_max_fpts_{window}'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
        df[f'lag_min_fpts_{window}'] = df.groupby('player_id')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
    
    # Add interaction features
    df['interaction_team_opponent'] = df['team_encoded'].astype(str) + '_' + df['opponent_encoded'].astype(str)

    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        processed_chunks = list(executor.map(engineer_features, chunks))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows(df, current_date):
    print(f"Creating synthetic rows for date: {current_date}...")
    synthetic_rows = []
    for player_id in df['full_name'].unique():
        player_df = df[df['full_name'] == player_id].tail(20)
        if player_df.empty:
            continue
        numeric_averages = player_df.mean(numeric_only=True)
        synthetic_row = pd.DataFrame([numeric_averages], columns=player_df.columns)
        synthetic_row['date'] = current_date
        synthetic_row['full_name'] = player_df['full_name'].iloc[0]
        for col in player_df.select_dtypes(include=['object']).columns:
            if col not in ['date', 'full_name']:
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

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/merged_api_fan_graphs_data_july_30th_1.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['full_name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    chunksize = 50000
    df = concurrent_feature_engineering(df, chunksize)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Label encode the team, opponent, and full_name features
    le_team = LabelEncoder()
    df['team'] = le_team.fit_transform(df['team'])
    joblib.dump(le_team, '/Users/sineshawmesfintesfaye/newenv/label_encoder_team.pkl')

    le_opponent = LabelEncoder()
    df['opponent'] = le_opponent.fit_transform(df['opponent'])
    joblib.dump(le_opponent, '/Users/sineshawmesfintesfaye/newenv/label_encoder_opponent.pkl')

    le_name = LabelEncoder()
    df['full_name'] = le_name.fit_transform(df['full_name'])
    joblib.dump(le_name, '/Users/sineshawmesfintesfaye/newenv/label_encoder_name.pkl')

    # Ensure columns are unique and exist in the DataFrame
    numeric_features = [col for col in top_20_features if col in df.columns]
    categorical_features = [
        'interaction_name_date', 'interaction_name_team', 'interaction_name_opponent', 'interaction_team_opponent'
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
            ('cat', categorical_transformer, categorical_features)
        ])

    # Clean the data before fitting the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Fit the preprocessor to get the number of features
    preprocessed_features = preprocessor.fit_transform(df[numeric_features + categorical_features])
    n_features = preprocessed_features.shape[1]

    # Feature selection based on the actual number of features
    k = min(25, n_features)  # Select the minimum of 20 or the actual number of features

    selector = SelectKBest(f_regression, k=k)

    # Base models for stacking
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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    # Time-based split
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    features_train = train_df.drop(columns=['calculated_dk_fpts', 'date'])
    target_train = train_df['calculated_dk_fpts']

    features_test = test_df.drop(columns=['calculated_dk_fpts', 'date'])
    target_test = test_df['calculated_dk_fpts']

    features_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_train.fillna(0, inplace=True)

    print("Training the model pipeline...")
    pipeline.fit(features_train, target_train)
    print("Model training completed.")

    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/newenv/ensemble_model_pipeline.pkl')
    print("Model pipeline saved.")

    pipeline = joblib.load('/Users/sineshawmesfintesfaye/newenv/ensemble_model_pipeline.pkl')
    print("Model pipeline loaded.")

    # Making predictions for the next day in the test set
    test_dates = test_df['date'].unique()
    rolling_results_df = rolling_predictions(train_df, pipeline, test_dates, chunksize=50000)

    # Align predictions with true values for evaluation
    rolling_results_df = rolling_results_df.merge(test_df[['full_name', 'date', 'calculated_dk_fpts']], on=['full_name', 'date'], how='left', suffixes=('', '_true'))

    # Drop rows with NaN values in the true target or predictions
    rolling_results_df.dropna(subset=['calculated_dk_fpts_true', 'predicted_dk_fpts'], inplace=True)

    # Ensure there are valid data points before evaluation
    if rolling_results_df.empty:
        raise ValueError("No valid data points found for evaluation.")

    mae, mse, r2, mape = evaluate_model(rolling_results_df['calculated_dk_fpts_true'], rolling_results_df['predicted_dk_fpts'])

    print(f'Mean Absolute Error on test set: {mae}')
    print(f'Mean Squared Error on test set: {mse}')
    print(f'R-squared on test set: {r2}')
    print(f'Mean Absolute Percentage Error on test set: {mape}')

    # Save the final data to a CSV file
    df.to_csv('/Users/sineshawmesfintesfaye/newenv/final_dataset_with_features.csv', index=False)
    print("Final dataset with all features saved.")

    if not rolling_results_df.empty:
        rolling_results_df.to_csv('/Users/sineshawmesfintesfaye/newenv/rolling_predictions.csv', index=False)
        print("Rolling predictions saved.")
    else:
        print("No rolling predictions to save.")

    save_feature_importance(pipeline, '/Users/sineshawmesfintesfaye/newenv/feature_importances.csv', '/Users/sineshawmesfintesfaye/newenv/feature_importances_plot.png')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")
