from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np
from tqdm import tqdm

from VQC import VQC
import tkinter as tk
from tkinter import filedialog

GUI = True

if GUI == True:
    # Create a root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file dialog to select a file
    input_path = filedialog.askopenfilename(title="Select a File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    input_file = Path(input_path)

    # Print the selected file path
    print('I am using the data file at the path:' + str(input_file))
else:
    current_file = Path(__file__)

    # modify here to select the correct training data
    input_file = current_file.parent.parent / 'data' / 'cancer' / 'downsampled_PCA_breast_cancer_dead_8_features.csv'
    print('I am using the data file at the path:' + str(input_file))

data = data_split(input_file, 'Status_Dead')  
column_names = list(data['X_train'].columns)
num_features = data['X_train'].shape[1]

#here fill with the chosen hyperparameters
encoding_options = ["angle"]
layer_options = [10]
ansatz_options= ["strong"]
learning_rates_options = [1e-3]
epochs_options =[1000]

dict_keys = ['encoding', 'layers', 'ansatz', 'learning_rate', 'epochs']

combinations = list(product(encoding_options, layer_options, ansatz_options, learning_rates_options, epochs_options))
dict_combinations = [dict(zip(dict_keys, tpl)) for tpl in combinations]

for i in dict_combinations:

    downsample = False
    pca = False

    # Check if 'downsample' and 'pca' are in the file name
    if "downsample" in input_file.name.lower():  # Case-insensitive check
        downsample = True

    if "pca" in input_file.name.lower():  # Case-insensitive check
        pca = True

    model = VQC(num_wires=num_features, num_outputs=1, num_layers=i['layers'], encoding=i['encoding'], reuploading=False) 
    model.train_model(data, epochs=i['epochs'], lr=i['learning_rate'], verbose = True)
    model.evaluate_model(data)


    filename = f"quantum_weights-enc_{i['encoding']}-ans_{i['ansatz']}-lay_{i['layers']}-lr_{i['learning_rate']}-ep_{i['epochs']}.pth"
    subfolder_name_feature = 'best_' + str(num_features) + '_features'

    prefix_path_dir = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ 'cancer' 

    #create directory
    if pca == True:
        model_path_dir = model_path_dir / 'PCA' 
    if downsample == True:
        model_path_dir = prefix_path_dir / 'downsampled'

    model_path_dir = model_path_dir / subfolder_name_feature
    model_path_dir.mkdir(parents=True, exist_ok=True)  # `parents=True` creates intermediate directories

    weights_complete_path = model_path_dir / filename
    if weights_complete_path.exists():
        print(f"File '{weights_complete_path}' already exists. Skipping...")
    else:
        # Write to the file
        model.save_model(weights_complete_path)
        print(f"File '{weights_complete_path}' created successfully. Weights saved.")
