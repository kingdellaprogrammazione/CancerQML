import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import numpy as np

current_file = Path(__file__)

input_file = current_file.parent.parent / 'data' / 'onehot' / 'breast_cancer_tonumbers_dead.csv'

df = pd.read_csv(input_file)
# Split data into features (X) and target (y)
X = df.drop('Status_Dead', axis=1)
y = df['Status_Dead']

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X,y)

# Get the feature importances
feature_importances = clf.feature_importances_

# Get the feature names
feature_names = X.columns

# Sort the feature importances and names
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_names = feature_names[sorted_indices]

dark_background = True
if dark_background == True:
    style_path = current_file.parent.parent / 'transparent.mplstyle'
    suffix = '_transparent'

else:
    style_path = current_file.parent.parent / 'normal_plot.mplstyle'
    suffix = '_normal'

plt.style.use(style_path)
# Plot the feature importances

plt.barh(range(len(sorted_importances)), sorted_importances, zorder = 2, color = '#ae77d6')
plt.yticks(range(len(sorted_importances)), sorted_names, zorder =2 )
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # This line inverts the y-axis
plt.tight_layout()
plt.show()

file = 'feature_importance' + suffix + '.png'

saving_path = current_file.parent.parent / 'results' / file
# Check if the file already exists
if saving_path.exists():
    print(f"The file '{saving_path.name}' already exists.")
else:
    # Save the final DataFrame if the file does not exist
    plt.savefig(saving_path)

