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

def get_intermediate_subdirectories(full_path, parent_folder):
    """
    Get all intermediate subdirectory names between a parent folder and a full path.

    Parameters:
        full_path (str or Path): The full path to a target directory or file.
        parent_folder (str or Path): The parent folder to serve as a reference point.

    Returns:
        list[str]: List of intermediate subdirectory names.

    Raises:
        ValueError: If the parent folder is not an ancestor of the full path.
    """
    # Convert inputs to Path objects
    full_path = Path(full_path).resolve()
    parent_folder = Path(parent_folder).resolve()

    # Ensure the parent folder is an ancestor of the full path
    if not full_path.is_relative_to(parent_folder):
        raise ValueError(f"{parent_folder} is not an ancestor of {full_path}")

    # Get relative path from parent folder to full path
    relative_path = full_path.relative_to(parent_folder)

    # Return all intermediate subdirectory names
    return [part for part in relative_path.parts[:-1]]  # Exclude the last part if it's a file

def match_dataset_name(dataset_name, words_to_check, words_to_remove):
    """
    Split a dataset name, remove specified words, and check if the remaining words match another list.

    Parameters:
        dataset_name (str): The name of the dataset (words separated by underscores).
        words_to_check (list[str]): List of words to match against the modified dataset name.
        words_to_remove (list[str]): List of words to remove from the dataset name.

    Returns:
        bool: True if the remaining words in the dataset name match the words_to_check, False otherwise.
    """
    # Split the dataset name into a list of words
    dataset_name_no_ext = Path(dataset_name).stem

    # Split the dataset name into a list of words
    name_parts = dataset_name_no_ext.split("_")    
    # Remove specified words
    filtered_parts = [word for word in name_parts if word not in words_to_remove]
    
    # Check if the remaining words match words_to_check
    return sorted(filtered_parts) == sorted(words_to_check)

GUI = False
current_file = Path(__file__)

if GUI == True:
    # Ask the user to select a directory containing weights files
    weights_dir = filedialog.askdirectory(title="Select the directory with weights files")

    # Convert the selected directory path to a Path object
    weights_path = Path(weights_dir)
    print("I am looking inside the dir:" + str(weights_path))
    # Get a list of all weights files in the selected directory (assuming .h5 files)
    # weights_files = [file for file in directory_path.glob("*.pth")]
else:
    weights_path = current_file.parent.parent / 'data' / 'weights' / 'quantum' / 'cancer'

if GUI == True:
    dataset_dir = filedialog.askdirectory(title="Select the dataset dir")
    dataset_path = Path(dataset_dir)
    # Get the file name from the Path object
    print("I am looking inside the dir:" + str(dataset_path))

else:
    dataset_path = current_file.parent.parent / 'data' / 'cancer'

file_dataset_map = {}

# Use Path.rglob() to recursively search for all files in the directory
for file in weights_path.rglob('*'):  # '*' matches all files and folders
    if file.is_file():  # Only add files, not directories
        name_list = get_intermediate_subdirectories(file, weights_path)
        for dataset in dataset_path.rglob('*'):
            if dataset.is_file():  # Only add files, not directories
                if match_dataset_name(dataset.name, name_list, ["breast", "cancer", "dead"]):
                    file_dataset_map[file] = dataset
                else:
                    print("File not found, skipping...")

plt.figure(figsize=(10, 10))

for model_file, dataset_file in file_dataset_map.items():
    data = data_split(dataset_file, 'Status_Dead')  

    # Loop through all .pth files in the selected directory
    # Extract the file name
    partial_name = model_file.name
    names = get_intermediate_subdirectories(model_file, weights_path)
    pattern_features = r"(\d+)features"
    regex = re.compile(pattern_features)
    # Check if any string in the list matches the pattern
    # Check each string and return the first match found
    for string in names:
        match = regex.search(string)
        if match:
            num_features = int(match.group(1))
    
        # Match the pattern with the file name
    pattern = r"quantum_weights-enc_(\w+)-ans_(\w+)-lay_(\d+)-lr_(\d+\.\d+)-ep_(\d+)\.pth"
    match = re.match(pattern, partial_name)

    if match:
        # Extract the groups from the match object
        enc = match.group(1)
        ans = match.group(2)
        lay = int(match.group(3))
        lr = float(match.group(4))
        ep = int(match.group(5))
    else:
        print(f"File: {partial_name} does not match the expected pattern.")

    # If you want to further process the extracted info:
    # For example, accessing the first file's info:
    model = VQC(num_wires=num_features, num_outputs=1, num_layers=lay, encoding=enc, reuploading=False) 
    model.load_model(model_file)  # Load model weights

    data = data_split(dataset_file, 'Status_Dead')  
    roc_data = model.evaluate_model(data)

    # Extract the data
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_value = roc_data['auc']
    # Plot the ROC curve for this model
    ref = ''.join(names)
    plt.plot(fpr, tpr, lw=1, label=f'ROC: {ref}, {lay}, {lr} (AUC = {auc_value:.2f})')


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
