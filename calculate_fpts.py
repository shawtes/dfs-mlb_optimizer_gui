import pandas as pd
import concurrent.futures

# Define the function to calculate DraftKings Fantasy Points
def calculate_dk_fpts(row):
    position = row['primary_position']
    
    if position in ["P", "SP", "RP"]:
        # Pitcher points
        innings_pitched = row.get("inningsPitched", 0) * 2.25
        strikeouts = row.get("strikeOuts", 0) * 2
        wins = row.get("wins", 0) * 4
        earned_runs = row.get("earnedRuns", 0) * -2
        hits_against = row.get("hitsAgainst", 0) * -0.6
        walks_against = row.get("baseOnBallsAgainst", 0) * -0.6
        hit_batsman = row.get("hitBatsmen", 0) * -0.6
        complete_game = row.get("completeGames", 0) * 2.5
        shutout = row.get("completeGameShutouts", 0) * 2.5
        no_hitter = row.get("noHitters", 0) * 5

        dk_fpts = (innings_pitched + strikeouts + wins + earned_runs + 
                   hits_against + walks_against + hit_batsman + 
                   complete_game + shutout + no_hitter)
    else:
        # Hitter points
        singles = (row.get("hits", 0) - row.get("doubles", 0) - 
                   row.get("triples", 0) - row.get("homeRuns", 0)) * 3
        doubles = row.get("doubles", 0) * 5
        triples = row.get("triples", 0) * 8
        home_runs = row.get("homeRuns", 0) * 10
        runs = row.get("runs", 0) * 2
        walks = row.get("baseOnBalls", 0) * 2
        hit_by_pitch = row.get("hitByPitch", 0) * 2
        stolen_bases = row.get("stolenBases", 0) * 5
        rbis = 4.7 * 2  # Using average RBI per season

        dk_fpts = (singles + doubles + triples + home_runs + rbis + 
                   runs + walks + hit_by_pitch + stolen_bases)

    return dk_fpts

# Function to process a chunk of the data and calculate dk_fpts
def process_chunk(chunk):
    chunk['dk_fpts'] = chunk.apply(calculate_dk_fpts, axis=1)
    return chunk

# Main function to read the CSV, process data in parallel, and write to a new CSV
def main(input_csv, output_csv):
    # Read the CSV in chunks
    chunksize = 1000  # Adjust based on your memory capacity
    df_iter = pd.read_csv(input_csv, chunksize=chunksize)
    
    processed_chunks = []
    
    # Use ThreadPoolExecutor to process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in df_iter}
        for future in concurrent.futures.as_completed(future_to_chunk):
            processed_chunks.append(future.result())
    
    # Concatenate all processed chunks
    df_processed = pd.concat(processed_chunks)
    
    # Write the processed data to a new CSV file
    df_processed.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    input_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/merged_fangraphs_logs.csv'
    output_csv = '/Users/sineshawmesfintesfaye/FangraphsDailyLogs/merged_fangraphs_logs.csv'
    main(input_csv, output_csv)
