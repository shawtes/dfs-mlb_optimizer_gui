import pandas as pd

# Load the datasets
merged_cheatsheet_df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/merged_cheatsheet_predictions_1:35_sep1st_4.csv')
contest_standings_df = pd.read_csv('/Users/sineshawmesfintesfaye/newenv/contest-standings-166171337.csv')

# Filter out pitchers
non_pitchers_df = merged_cheatsheet_df[merged_cheatsheet_df['position'] != 'P']

# Merge datasets on player identifier
merged_df = pd.merge(non_pitchers_df, contest_standings_df, left_on='Name', right_on='Player', how='inner')

# Calculate the Mean Absolute Error (MAE)
mae = (merged_df['FPTS'] - merged_df['predicted_dk_fpts']).abs().mean()

# Calculate the average of positive differences
positive_diff_avg = (merged_df['FPTS'] - merged_df['predicted_dk_fpts'])[merged_df['FPTS'] > merged_df['predicted_dk_fpts']].mean()

# Calculate the average of negative differences
negative_diff_avg = (merged_df['FPTS'] - merged_df['predicted_dk_fpts'])[merged_df['FPTS'] < merged_df['predicted_dk_fpts']].mean()

# Print the results
print(f"MAE: {mae}")
print(f"Average Positive Difference: {positive_diff_avg}")
print(f"Average Negative Difference: {negative_diff_avg}")