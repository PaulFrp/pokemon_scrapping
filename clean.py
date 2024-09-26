import pandas as pd

# Load the CSV file
data = pd.read_csv('./data/combined_data.csv')

print("Number of rows before filtering:", len(data))

filtered_data = data[data['comment'].str.len() >= 50]

print("Number of rows after filtering:", len(filtered_data))

filtered_data.to_csv('./data/filtered_file.csv', index=False)
