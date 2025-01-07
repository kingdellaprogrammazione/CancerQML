from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from functions import *
from itertools import product
import numpy as np
from tqdm import tqdm

from VQC import VQC
import re
import tkinter as tk
from tkinter import filedialog

GUI = True
current_file = Path(__file__)

if GUI == True:
    # Create a root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file selection dialog
    input_path = filedialog.askopenfilename(title="Select the data file")
    input_file = Path(input_path)

    # Print the selected file path
    print('I am using the data file at the path:' + str(input_file))

else:
    # modify here to select the correct training data
    input_file = current_file.parent.parent / 'data' / 'cancer' / 'downsampled_PCA_breast_cancer_dead_8_features.csv'
    print('I am using the data file at the path:' + str(input_file))


data = data_split(input_file, 'Status_Dead')  
column_names = list(data['X_train'].columns)
num_features = data['X_train'].shape[1]

if GUI == True:
    # Ask the user to select a directory containing weights files
    selected_dir = filedialog.askdirectory(title="Select the directory with weights files")

    # Convert the selected directory path to a Path object
    directory_path = Path(selected_dir)
    print("I am looking inside the dir:" + str(directory_path))
    # Get a list of all weights files in the selected directory (assuming .h5 files)
    # weights_files = [file for file in directory_path.glob("*.pth")]

else:
    model_path_dir = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ 'cancer'  / 'best_8_features' / 'downsampled' / 'PCA'  
    print("I am looking inside the dir:" + str(model_path_dir))

    # match values
    file_to_load = input("Enter the file name\n")

    complete_path = model_path_dir / file_to_load

# Define the regex pattern
pattern = r"quantum_weights-enc_(\w+)-ans_(\w+)-lay_(\d+)-lr_(\d+\.\d+)-ep_(\d+)\.pth"

# Initialize a list to store the extracted information
extracted_info = []

# Initialize the plot
plt.figure(figsize=(10, 8))

# Loop through all .pth files in the selected directory
for file in directory_path.glob("*.pth"):
    # Extract the file name
    partial_name = file.name

    file_to_load = directory_path / partial_name
    
    # Match the pattern with the file name
    match = re.match(pattern, partial_name)
    
    if match:
        # Extract the groups from the match object
        enc = match.group(1)
        ans = match.group(2)
        lay = int(match.group(3))
        lr = float(match.group(4))
        ep = int(match.group(5))
        
        # Store the extracted information in a dictionary or tuple
        extracted_info.append({
            "file_name": file_to_load,
            "encoding": enc,
            "answer": ans,
            "layers": lay,
            "learning_rate": lr,
            "epochs": ep
        })
        
        # Print or use the extracted information
        print(f"File: {file_to_load}, Encoding: {enc}, Answer: {ans}, "
              f"Layers: {lay}, Learning Rate: {lr}, Epochs: {ep}")
    else:
        print(f"File: {file_to_load} does not match the expected pattern.")

# If you want to further process the extracted info:
# For example, accessing the first file's info:
for i in extracted_info:
    if i["learning_rate"] == 0.001:
        model = VQC(num_wires=8, num_outputs=1, num_layers=i["layers"], encoding=i["encoding"], reuploading=False) 
        model.load_model(i["file_name"])  # Load model weights
        roc_data = model.evaluate_model(data)

        # Extract the data
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        auc_value = roc_data['auc']

        # Plot the ROC curve for this model
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {i["layers"]}, 0.001 (AUC = {auc_value:.2f})')

else:
    print("Invalid file name format.")

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Labels and title
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.grid(True)
plt.show()

