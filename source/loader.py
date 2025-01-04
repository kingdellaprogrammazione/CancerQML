from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np
from tqdm import tqdm

from VQC import VQC
import re

current_file = Path(__file__)

# modify here to select the correct training data
input_file = current_file.parent.parent / 'data' / 'cancer' / 'breast_cancer_12features.csv'
print('I am using the data file at the path:' + str(input_file))

data = data_split(input_file, 'Status_Dead')  
column_names = list(data['X_train'].columns)
num_features = data['X_train'].shape[1]

model_path_dir = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ 'cancer'  / 'best_12_features'
print("I am looking inside the dir:" + str(model_path_dir))

# match values
file_to_load = input("Enter the file name\n")
pattern = r"quantum_weights-enc:(\w+)-ans:(\w+)-lay:(\d+)-lr:(\d+\.\d+)-ep:(\d+)\.pth"
match = re.match(pattern, file_to_load)
complete_path = model_path_dir / file_to_load

if match:
    # Extract the parameters using the groups
    enc = match.group(1)
    ans = match.group(2)
    lay = int(match.group(3))
    lr = float(match.group(4))
    ep = int(match.group(5))

    # Print the extracted parameters
    print(f"Extracted Parameters:")
    print(f"  Encoding: {enc}")
    print(f"  Ansatz: {ans}")
    print(f"  Layers: {lay}")
    print(f"  Learning Rate : {lr}")
    print(f"  Epochs: {ep}")

    model = VQC(num_wires=12, num_outputs=1, num_layers=lay, encoding=enc, reuploading=False) 
    model.load_model(complete_path)  # Load model weights
    
    model.evaluate_model(data, draw_roc = 'True')

else:
    print("Invalid file name format.")


