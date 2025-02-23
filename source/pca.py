from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from VQC import VQC

current_file = Path(__file__)

# modify here to select the correct training data
input_file = current_file.parent.parent / 'data' / 'processed' / 'downsampled' / 'downsampled_breast_cancer_dead.csv'
print('I am using the data file at the path:' + str(input_file))

data = pd.read_csv(input_file)
target = data['Status_Dead']
data = data.drop(columns=['Status_Dead'])

# Step 6: Scale the data to range [-1, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# Step 7: Apply PCA
pca = PCA()
pca.fit(data_scaled)

dark_background = True
if dark_background == True:
    style_path = current_file.parent.parent / 'transparent.mplstyle'
    suffix = '_transparent'

else:
    style_path = current_file.parent.parent / 'normal_plot.mplstyle'
    suffix = '_normal'

plt.style.use(style_path)
# Plot the feature importances

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color ='#ae77d6')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')

file = 'variance' + suffix + '.png'

saving_path = current_file.parent.parent / 'results' / file
# Check if the file already exists
if saving_path.exists():
    print(f"The file '{saving_path.name}' already exists.")
else:
    # Save the final DataFrame if the file does not exist
    plt.savefig(saving_path)

plt.show()

# Step 9: Select the number of components (e.g., 95% variance threshold)

variance_threshold = float(input("Enter the variance threshold (e.g., 0.95 for 95%): "))
desired_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold)
print(f"Number of components selected to retain {variance_threshold} variance: {desired_components}")

# Step 10: Transform the data using the selected number of components
pca = PCA(n_components=desired_components)
data_pca = pca.fit_transform(data_scaled)

# Create the PCA DataFrame and re-add target
final_data = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(desired_components)])
final_data['Status_Dead'] = target.reset_index(drop=True)

# Save the final DataFrame
output_file = current_file.parent.parent / 'data' / 'processed' / 'downsampled_pca' / f'downsampled_pca_breast_cancer_dead_{desired_components}f.csv'

# Check if the file already exists
if output_file.exists():
    print(f"The file '{output_file.name}' already exists.")
else:
    # Save the final DataFrame if the file does not exist
    final_data.to_csv(output_file, index=False)
    print(f"File saved as '{output_file.name}'.")

