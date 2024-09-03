import pandas as pd

# Read the CSV files
cheatsheet = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/DFF_MLB_cheatsheet_2024-09-01.csv')
predictions = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/batters_predictions_20240901.csv')

# Combine first and last name in cheatsheet
cheatsheet['Name'] = cheatsheet['first_name'] + ' ' + cheatsheet['last_name']

# Convert date columns to datetime
cheatsheet['game_date'] = pd.to_datetime(cheatsheet['game_date'])
predictions['date'] = pd.to_datetime(predictions['date'])

# Merge the dataframes
merged = pd.merge(cheatsheet, predictions, 
                  left_on=['Name', 'game_date'], 
                  right_on=['Name', 'date'], 
                  how='left')

# Handle unmatched players
unmatched = merged[merged['date'].isna()]
if not unmatched.empty:
    # Use L5_fppg_avg for unmatched players
    merged.loc[merged['date'].isna(), 'predicted_dk_fpts'] = merged.loc[merged['date'].isna(), 'L5_fppg_avg']

# Drop unnecessary columns and fill NaN values
merged = merged.drop(['first_name', 'last_name', 'date'], axis=1)
merged['predicted_dk_fpts'] = merged['predicted_dk_fpts'].fillna(merged['L5_fppg_avg'])

# Save the merged dataframe
merged.to_csv('/Users/sineshawmesfintesfaye/newenv/merged_cheatsheet_predictions_1:35_sep1st_4.csv', index=False)

print(f"Merged data saved. Shape: {merged.shape}")
print(f"Number of players using L5_fppg_avg: {unmatched.shape[0]}")