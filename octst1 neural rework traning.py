import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from sklearn.model_selection import GroupShuffleSplit

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

# Load dataset
input_file = '/users/sineshawmesfintesfaye/newenv/merged_output_29_sep_01.csv'
df = pd.read_csv(input_file, low_memory=False)
df['date'] = pd.to_datetime(df['date'])

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

# Feature engineering function
def extend_label_encoder(le, new_labels):
    current_classes = set(le.classes_)
    new_classes = set(new_labels) - current_classes
    if new_classes:
        le.classes_ = np.concatenate([le.classes_, list(new_classes)])

def engineer_features(df):
    # Extend label encoders first
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
    
    # Handle zero values in 5-game average
    df['5_game_avg'] = df['5_game_avg'].replace(0, np.nan).fillna(df['calculated_dk_fpts'].mean())
    
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

# Before calling engineer_features
print("le_name classes before:", le_name.classes_)
print("le_team classes before:", le_team.classes_)

# Apply Feature Engineering
df = engineer_features(df)

# After calling engineer_features
print("le_name classes after:", le_name.classes_)
print("le_team classes after:", le_team.classes_)
print("Final columns:", df.columns.tolist())

# Extend the label encoders to handle unseen categories
def extend_label_encoder(le, new_labels):
    # Get the classes seen so far
    current_classes = list(le.classes_)
    
    # Find any new labels not seen before
    unseen_labels = [label for label in new_labels if label not in current_classes]
    
    # Extend the classes with new labels
    le.classes_ = np.array(current_classes + unseen_labels)

# ***Only include selected features for training/prediction*** 
# We remove any features not included in the training process.
selected_features = ['Off', 'WAR', 'Dol', 'RAR', 'RE24', 'REW', 'SLG', 'WPA/LI', 'AB', 'WAR']
engineered_features = [
    'wOBA_Statcast', 'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast','Name_encoded', 'team_encoded',
    'Opponent_encoded', 'interaction_name_opponent','interaction_name_date', 'interaction_name_team',
    'Date_Team_Interaction', 'Name_Opponent_Interaction', 
    'Name_Team_Interaction', 'Team_Opponent_Interaction', '5_game_pos_diff_avg', '5_game_neg_diff_avg'
    ]
selected_features += engineered_features

# Update selected_features to include the new columns
selected_features += ['5_game_pos_diff_avg', '5_game_neg_diff_avg']

# Ensure all selected features are in the dataframe
selected_features = [f for f in selected_features if f in df.columns]

print("Final selected features:", selected_features)

# Save the list of selected features
feature_list_path = '/Users/sineshawmesfintesfaye/newenv/numeric_features_selected.pkl'
joblib.dump(selected_features, feature_list_path)

# Scale selected features
scaler_path = '/Users/sineshawmesfintesfaye/newenv/scaler_selected.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])
    joblib.dump(scaler, scaler_path)

# Select Features and Target
X = df[selected_features]
y = df['calculated_dk_fpts']

# Check if X and y are empty before splitting
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Input data X or y is empty. Please check your data preparation steps.")

# Proceed with train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Deep Learning Model
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))  # Predicting 'calculated_dk_fpts'
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# Create and Train Model
input_shape = X_train_scaled.shape[1]
model = create_model(input_shape)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=10000, batch_size=100, callbacks=[early_stopping])

# Evaluate Model
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Metrics:")
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train R^2: {train_r2}, Test R^2: {test_r2}")

# Save the model and scaler
model_save_path = '/Users/sineshawmesfintesfaye/newenv/dk_fpts_model.keras'
model.save(model_save_path)
scaler_save_path = '/Users/sineshawmesfintesfaye/newenv/scaler.pkl'
joblib.dump(scaler, scaler_save_path)
feature_list_path = '/Users/sineshawmesfintesfaye/newenv/selected_features.pkl'
joblib.dump(selected_features, feature_list_path)

print(f"Model saved at {model_save_path}")
print(f"Scaler saved at {scaler_save_path}")
print(f"Feature list saved at {feature_list_path}")

# Save label encoders
joblib.dump(le_name, '/Users/sineshawmesfintesfaye/newenv/label_encoder_name.pkl')
joblib.dump(le_team, '/Users/sineshawmesfintesfaye/newenv/label_encoder_team.pkl')

# Plot Training History
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
