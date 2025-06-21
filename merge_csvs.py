import os
import pandas as pd

def merge_csvs(folder_path, output_file):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged {len(all_files)} files into {output_file}")
    else:
        print("No CSV files found to merge.")

if __name__ == "__main__":
    folder = "/Users/sineshawmesfintesfaye/FangraphsDailyLogs"
    output = os.path.join(folder, "merged_fangraphs_logs.csv")
    merge_csvs(folder, output) 