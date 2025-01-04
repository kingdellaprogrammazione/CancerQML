import pandas as pd
from pathlib import Path

# Load your dataset

current_file = Path(__file__)

# modify here to select the correct training data

input_path = current_file.parent.parent / 'data' / 'cancer' 
input_file = input_path / 'breast_cancer_12features.csv'

print('I am using the data file at the path:' + str(input_file))

df = pd.read_csv(input_file)

# Assuming the target column is named 'label' with values 0 and 1
class_0 = df[df['Status_Dead'] == 0]
class_1 = df[df['Status_Dead'] == 1]

# Determine the downsampling size (e.g., to match the smaller class)
downsample_size = min(len(class_0), len(class_1))

# Downsample the larger class
class_0_downsampled = class_0.sample(n=downsample_size, random_state=42)
class_1_downsampled = class_1.sample(n=downsample_size, random_state=42)

# Combine the downsampled datasets
downsampled_df = pd.concat([class_0_downsampled, class_1_downsampled])

# Shuffle the combined dataset
downsampled_df = downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the downsampled dataset
output_file = input_path / 'downsampled_breast_cancer_12features.csv'
downsampled_df.to_csv(output_file, index=False)

print("Original dataset size:", len(df))
print("Downsampled dataset size:", len(downsampled_df))
