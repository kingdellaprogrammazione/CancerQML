from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np

from VQC import VQC

current_file = Path(__file__)
input = current_file.parent.parent / 'data' / 'diabete' / 'diabetes_cleaned.csv'
print('I am using the data file at the path:' + str(input))

data = data_split(input)  
column_names = list(data['X_train'].columns)
num_features = data['X_train'].shape[1]
# check the number of qubits 

encoding_options = ["angle"]
layer_options = [10,20]
ansatz_options= ["strong"]
learning_rates_options = [1e-3]
epochs_options =[1000]

# we'll apply later
threshold_range = np.linspace(0, 1, num=20)

dict_keys = ['encoding', 'layers', 'ansatz', 'learning_rate', 'epochs']

combinations = list(product(encoding_options, layer_options, ansatz_options, learning_rates_options, epochs_options))
dict_combinations = [dict(zip(dict_keys, tpl)) for tpl in combinations]

for i in dict_combinations:

    model = VQC(num_wires=8, num_outputs=1, num_layers=i['layers'], encoding=i['encoding'], reuploading=False) 
    model.train_model(data, epochs=i['epochs'], lr=i['learning_rate'], verbose = False)
    model.evaluate_model(data)

    filename = f"quantum_weights-features:{'-'.join(column_names)}-encoding:{i['encoding']}-ansatz:{i['ansatz']}-layers:{i['layers']}-learning_rate:{i['learning_rate']}-epochs:{i['epochs']}.pth"
    model_path_dict = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ 'diabete' / filename
    if model_path_dict.exists():
        print(f"File '{model_path_dict}' already exists. Skipping...")
    else:
        # Write to the file
        print(f"File '{model_path_dict}' created successfully. Weights saved.")
        model.save_model(model_path_dict)
