import pandas as pd
from pathlib import Path

current_file = Path(__file__)
# Load the CSV file
input = current_file.parent.parent / 'data' / 'cancer' / 'breast_cancer_12features.csv'
output_file = current_file.parent.parent / 'data' / 'cancer' / 'breast_cancer_12features2.csv' # Replace with your desired output file name

# Read the CSV into a DataFrame
df = pd.read_csv(input)

# Invert the values in the last column
last_column = df.columns[-1]  # Get the last column name
df[last_column] = df[last_column].apply(lambda x: 1 if x == 0 else 0)

# Rename the last column
df.rename(columns={last_column: "Status_Dead"}, inplace=True)

# Save the updated DataFrame to a new file
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")
