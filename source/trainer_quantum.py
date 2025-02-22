from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np
from tqdm import tqdm
import re

from VQC import VQC
import tkinter as tk
from tkinter import filedialog

GUI = True
current_file = Path(__file__)

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

    # modify here to select the correct training data
    input_file = current_file.parent.parent / 'data' / 'processed' / 'downsampled_pca' / 'downsampled_pca_breast_cancer_dead_8f.csv'
    print('I am using the data file at the path:' + str(input_file))

#capture feature number
match = re.search(r'(\d+)f', str(input_file))

if match:
    # Extract the matched number
    num_features = int(match.group(1))
else:
    print("No match for feature number found.")
    exit()

data = data_split(input_file, 'Status_Dead')  
column_names = list(data['X_train'].columns)
# num_features = data['X_train'].shape[1]

#here fill with the chosen hyperparameters
encoding_options = ["angle"]
layer_options = [15,20]
ansatz_options= ["strong"]
learning_rates_options = [1e-2]
epochs_options =[1000]

dict_keys = ['encoding', 'layers', 'ansatz', 'learning_rate', 'epochs']

combinations = list(product(encoding_options, layer_options, ansatz_options, learning_rates_options, epochs_options))
dict_combinations = [dict(zip(dict_keys, tpl)) for tpl in combinations]

for i in dict_combinations:

    downsample = False
    pca = False

    # Check if 'downsample' and 'pca' are in the file name
    if "downsample" in input_file.name.lower():  # Case-insensitive check
        downsample_string = 'downsampled-'
    else:
        downsample_string=''

    if "pca" in input_file.name.lower():  # Case-insensitive check
        pca_string = 'pca-'
    else:
        pca_string = 'pca-'

    model = VQC(num_wires=num_features, num_outputs=1, num_layers=i['layers'], encoding=i['encoding'], reuploading=False) 
    model.train_model(data, epochs=i['epochs'], lr=i['learning_rate'], verbose = True)
    model.evaluate_model(data)


    first_piece = 'q_weights-enc_ang-ans_'
    final_piece = f"ans_strong-lay_{i['layers']}-lr_{i['learning_rate']}-ep_{i['epochs']}-{i[num_features]}f.pth"
    
    complete_filename = first_piece + downsample_string + pca_string + final_piece

    model_path_dir = current_file.parent.parent / 'weights' / 'quantum'/ 'angle' 

    weights_complete_path = model_path_dir / complete_filename

    if weights_complete_path.exists():
        print(f"File '{weights_complete_path}' already exists. Skipping...")
    else:
        # Write to the file
        model.save_model(weights_complete_path)
        print(f"File '{weights_complete_path}' created successfully. Weights saved.")
