import pandas as pd

# Load the CSV files
combined_df = pd.read_csv('./data/filtered_file.csv')
filtered_df = pd.read_csv('./data/filtered_threads_file.csv')

merged_df = pd.concat([combined_df, filtered_df])

merged_df = merged_df.drop_duplicates()

merged_df.to_csv('./data/merged_filtered_output.csv', index=False)

print("Files have been merged and saved to merged_output.csv")
