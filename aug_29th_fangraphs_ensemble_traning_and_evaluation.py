import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

# Define constants for calculations
WAR_conversion_factor = 8e6
wOBA_weights_2020 = {
    'uBB': 0.69,
    'HBP': 0.72,
    '1B': 0.88,
    '2B': 1.27,
    '3B': 1.62,
    'HR': 2.10
}

def calculate_dk_fpts(row):
    return (row['1B'] * 3 + row['2B'] * 5 + row['3B'] * 8 + row['HR'] * 10 +
            row['RBI'] * 2 + row['R'] * 2 + row['BB'] * 2 + row['HBP'] * 2 + row['SB'] * 5)

# Initial list of selected features based on importance
selected_features = [
    'Off', 'WAR', 'Dol', 'RAR', 'wRC', 'wRAA', 'Bat', 'wRC+',
    'wOBA', 'RE24', 'REW', 'SLG', 'WPA/LI'
]

# Add new engineered features to selected_features 
engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast'
]
selected_features += engineered_features

# Ensure no duplicates in selected_features
selected_features = list(set(selected_features))

def calculate_statcast_metrics(df):
    new_columns = {}
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    new_columns['wOBA_Statcast'] = (
        (wOBA_weights_2020['uBB'] * df['BB']) + 
        (wOBA_weights_2020['HBP'] * df['HBP']) + 
        (wOBA_weights_2020['1B'] * df['1B']) + 
        (wOBA_weights_2020['2B'] * df['2B']) + 
        (wOBA_weights_2020['3B'] * df['3B']) + 
        (wOBA_weights_2020['HR'] * df['HR'])
    ) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])

    new_columns['SLG_Statcast'] = (
        df['1B'] + (2 * df['2B']) + (3 * df['3B']) + (4 * df['HR'])
    ) / df['AB']

    new_columns['Offense_Statcast'] = df['wRAA'] + df['BsR'] + (df['wOBA'] - df['xwOBA'])
    new_columns['RAR_Statcast'] = df['WAR'] * 10
    new_columns['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor
    new_columns['WPA/LI_Statcast'] = df['WPA/LI']

    # Ensure all columns are 1-dimensional
    for key, value in new_columns.items():
        if isinstance(value, pd.DataFrame):
            new_columns[key] = value.iloc[:, 0]

    return pd.concat([df, pd.DataFrame(new_columns)], axis=1)

def engineer_features(df):
    print("Starting feature engineering...")
    new_columns = {}
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), errors='coerce')
    
    new_columns['year'] = df['date'].dt.year
    new_columns['month'] = df['date'].dt.month
    new_columns['day'] = df['date'].dt.day
    new_columns['day_of_week'] = df['date'].dt.dayofweek
    new_columns['day_of_season'] = (df['date'] - df['date'].min()).dt.days
    
    for lag in range(1, 50):
        new_columns[f'lag_{lag}'] = df.groupby('Name')['calculated_dk_fpts'].shift(lag)
    
    new_columns['rolling_min_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).min())
    new_columns['rolling_max_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).max())
    new_columns['rolling_mean_fpts_7'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    new_columns['rolling_mean_fpts_49'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(49, min_periods=1).mean())
    
    for lag in range(1, 51):
        new_columns[f'lag_HR_{lag}'] = df.groupby('Name')['HR'].shift(lag)
        new_columns[f'lag_RBI_{lag}'] = df.groupby('Name')['RBI'].shift(lag)
        new_columns[f'lag_R_{lag}'] = df.groupby('Name')['R'].shift(lag)

    new_columns['OPS'] = df['OBP'] + df['SLG']

    ema_features = ['AVG', 'OBP', 'SLG']
    for feature in ema_features:
        new_columns[f'ema_{feature}_30'] = df.groupby('Name')[feature].transform(lambda x: x.ewm(span=30).mean())

    new_columns['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
    
    # Ensure all columns are 1-dimensional
    for key, value in new_columns.items():
        if isinstance(value, pd.DataFrame):
            new_columns[key] = value.iloc[:, 0]
    
    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    df.fillna(0, inplace=True)
    df = calculate_statcast_metrics(df)
    print("Feature engineering completed.")

    return df

def concurrent_feature_engineering(df, chunksize=50000):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    start_time = time.time()
    processed_chunks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # Reduced max_workers
        futures = [executor.submit(engineer_features, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                processed_chunks.append(result)
            except Exception as e:
                print(f"An error occurred during feature engineering: {str(e)}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows(df, current_date):
    print(f"Creating synthetic rows for date: {current_date}...")
    synthetic_rows = []
    for player_id in df['player_encoded'].unique():
        player_df = df[df['player_encoded'] == player_id].tail(15)
        if player_df.empty:
            continue
        numeric_averages = player_df.select_dtypes(include=[np.number]).mean()
        
        # Create a dictionary for the synthetic row
        synthetic_row = numeric_averages.to_dict()
        synthetic_row['date'] = current_date
        synthetic_row['player_encoded'] = player_df['player_encoded'].iloc[0]
        
        # Handle non-numeric columns
        for col in player_df.select_dtypes(exclude=[np.number]).columns:
            if col not in ['date', 'player_encoded']:
                synthetic_row[col] = player_df[col].mode()[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
        
        synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.DataFrame(synthetic_rows)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {current_date}.")
    return synthetic_df

def process_predictions(chunk, pipeline):
    if chunk.shape[0] == 0:
        return chunk
    features = chunk.drop(columns=['calculated_dk_fpts'])
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    predictions = pipeline.named_steps['model'].predict(features_preprocessed)
    chunk['predicted_dk_fpts'] = predictions.flatten()  # Ensure 1D
    return chunk

def concurrent_rolling_predictions(train_data, model_pipeline, test_dates, chunksize=10000):
    print("Starting concurrent rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows(train_data, current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        print(f"Columns in synthetic_rows: {synthetic_rows.columns.tolist()}")
        print(f"Shape of synthetic_rows: {synthetic_rows.shape}")
        
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    print(f"Shape of y_true: {y_true.shape}")
    print(f"Shape of y_pred: {y_pred.shape}")
    print(f"Type of y_true: {type(y_true)}")
    print(f"Type of y_pred: {type(y_pred)}")
    
    # Ensure y_true is 1D
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0]
    elif isinstance(y_true, np.ndarray) and y_true.ndim > 1:
        y_true = y_true.flatten()
    
    # Ensure y_pred is 1D
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0]
    elif isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    print(f"Shape of y_true after processing: {y_true.shape}")
    print(f"Shape of y_pred after processing: {y_pred.shape}")
    
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
    
    feature_names = preprocessor.get_feature_names_out()
    
    if len(feature_importances) != len(feature_names):
        raise ValueError("The number of feature importances does not match the number of feature names.")
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
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
    plt.close()  # Close the plot to free up memory
    print(f"Feature importance plot saved to {output_plot_path}")

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/aug29_select.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'].str.strip(), errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Label encode the team column
    le_team = LabelEncoder()
    df['team_encoded'] = le_team.fit_transform(df['Team'])

    # Label encode the player names
    le_player = LabelEncoder()
    df['player_encoded'] = le_player.fit_transform(df['Name'])

    # Save the encoders for later use
    joblib.dump(le_team, '/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/team_label_encoder.pkl')
    joblib.dump(le_player, '/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/player_label_encoder.pkl')

    try:
        df = concurrent_feature_engineering(df, chunksize=10000)  # Reduced chunksize
    except Exception as e:
        print(f"An error occurred during concurrent feature engineering: {str(e)}")
        print("Falling back to non-concurrent feature engineering...")
        df = engineer_features(df)

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Filter for selected features
    features = selected_features + ['calculated_dk_fpts', 'team_encoded', 'player_encoded']
    df = df[features + ['date', 'Name']]

    numeric_features = [col for col in selected_features if col not in df.select_dtypes(include=['object']).columns] + ['team_encoded', 'player_encoded']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = [col for col in selected_features if col in df.select_dtypes(include=['object']).columns]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

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

    # Adding Bagging and Boosting models
    bagging_model = BaggingRegressor(base_estimator=Ridge(), n_estimators=10, random_state=42)
    boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Adding Voting Regressor
    voting_model = VotingRegressor(estimators=[
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('svr', SVR()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_model)  # You can change to bagging_model, boosting_model, or voting_model for different ensemble methods
    ])

    # Time-based split
    train_size = int(len(df) * 0.6)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Add this before creating target_train
    print("\nShape of calculated_dk_fpts:")
    print(train_df['calculated_dk_fpts'].shape)

    print("\nFirst few rows of calculated_dk_fpts:")
    print(train_df['calculated_dk_fpts'].head())

    features_train = train_df.drop(columns=['calculated_dk_fpts', 'date', 'Name'])
    target_train = train_df['calculated_dk_fpts'].iloc[:, 0].values  # Take the first column and convert to 1D array

    features_test = test_df.drop(columns=['calculated_dk_fpts', 'date', 'Name'])
    target_test = test_df['calculated_dk_fpts'].iloc[:, 0].values  # Take the first column and convert to 1D array

    features_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_train.fillna(0, inplace=True)
     # Add debug statements here
    print("Columns in features_train:")
    print(features_train.columns.tolist())

    print("\nUnique columns in features_train:")
    print(features_train.columns.unique().tolist())

    print("\nDuplicate columns in features_train:")
    duplicate_columns = features_train.columns[features_train.columns.duplicated()].tolist()
    print(duplicate_columns)

    print("\nColumns in selected_features:")
    print(selected_features)

    print("\nColumns in numeric_features:")
    print(numeric_features)

    print("\nColumns in categorical_features:")
    print(categorical_features)

    print("\nShape of features_train:")
    print(features_train.shape)

    # Check for columns in selected_features that are not in features_train
    missing_columns = set(selected_features) - set(features_train.columns)
    print("\nColumns in selected_features but not in features_train:")
    print(list(missing_columns))

    # Add these debug prints before fitting the model
    print("\nShape of target_train:")
    print(target_train.shape)

    print("\nFirst few values of target_train:")
    print(target_train[:5])

    print("\nTraining the model pipeline...")
    pipeline.fit(features_train, target_train)
    print("Model training completed.")

    joblib.dump(pipeline, '/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/ensemble_model_pipeline_fangraphs.pkl')
    print("Model pipeline saved.")

    pipeline = joblib.load('/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/ensemble_model_pipeline_fangraphs.pkl')
    print("Model pipeline loaded.")

    # Making predictions for the next day in the test set
    test_dates = test_df['date'].unique()
    rolling_results_df = concurrent_rolling_predictions(train_df, pipeline, test_dates, chunksize=10000)  # Reduced chunksize

    if rolling_results_df is not None and not rolling_results_df.empty:
        # Align predictions with true values for evaluation
        rolling_results_df = rolling_results_df.merge(test_df[['Name', 'date', 'calculated_dk_fpts']], on=['Name', 'date'], how='left', suffixes=('', '_true'))

        # Drop rows with NaN values in the true target or predictions
        rolling_results_df.dropna(subset=['calculated_dk_fpts_true', 'predicted_dk_fpts'], inplace=True) 

        # When calling evaluate_model, add some debugging information
        print("Shape of rolling_results_df:", rolling_results_df.shape)
        print("Columns in rolling_results_df:", rolling_results_df.columns.tolist())
        print("First few rows of calculated_dk_fpts_true:")
        print(rolling_results_df['calculated_dk_fpts_true'].head())
        print("First few rows of predicted_dk_fpts:")
        print(rolling_results_df['predicted_dk_fpts'].head())

        mae, mse, r2, mape = evaluate_model(rolling_results_df['calculated_dk_fpts_true'], rolling_results_df['predicted_dk_fpts'])

        print(f'Mean Absolute Error on test set: {mae}')
        print(f'Mean Squared Error on test set: {mse}')
        print(f'R-squared on test set: {r2}')
        print(f'Mean Absolute Percentage Error on test set: {mape}')

        rolling_results_df.to_csv('/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/rolling_predictions_fan_graphs.csv', index=False)
        print("Rolling predictions saved to /Users/sineshawmesfintesfaye/Downloads/annotated_s&d/rolling_predictions_fan_graphs.csv")
    else:
        print("No rolling predictions to evaluate or save.")
    
    save_feature_importance(pipeline, '/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/feature_importances.csv', '/Users/sineshawmesfintesfaye/Downloads/annotated_s&d/feature_importances_plot.png')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")
