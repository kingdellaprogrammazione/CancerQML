from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np

from VQC import VQC

current_file = Path(__file__)

# modify here to select the correct training data
input = current_file.parent.parent / 'data' / 'cancer' / 'breast_cancer_12features.csv'
print('I am using the data file at the path:' + str(input))

data = data_split(input, 'Status_Alive')  
column_names = list(data['X_train'].columns)
num_features = data['X_train'].shape[1]

#here fill with the chosen hyperparameters
encoding_options = ["angle"]
layer_options = [1]
ansatz_options= ["strong"]
learning_rates_options = [1e-3]
epochs_options =[1]

# we'll apply later
threshold_range = np.linspace(0, 1, num=20)

dict_keys = ['encoding', 'layers', 'ansatz', 'learning_rate', 'epochs']

combinations = list(product(encoding_options, layer_options, ansatz_options, learning_rates_options, epochs_options))
dict_combinations = [dict(zip(dict_keys, tpl)) for tpl in combinations]

for i in dict_combinations:

    model = VQC(num_wires=num_features, num_outputs=1, num_layers=i['layers'], encoding=i['encoding'], reuploading=False) 
    model.train_model(data, epochs=i['epochs'], lr=i['learning_rate'], verbose = False)
    model.evaluate_model(data)

    filename = f"quantum_weights-enc:{i['encoding']}-ans:{i['ansatz']}-lay:{i['layers']}-lr:{i['learning_rate']}-ep:{i['epochs']}.pth"
    subfolder_name_feature = 'best_' + str(num_features) + '_features'

    #create directory
    model_path_dir = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ 'cancer'  / subfolder_name_feature
    model_path_dir.mkdir(parents=True, exist_ok=True)  # `parents=True` creates intermediate directories
   
    weights_complete_path = model_path_dir / filename
    if weights_complete_path.exists():
        print(f"File '{weights_complete_path}' already exists. Skipping...")
    else:
        # Write to the file
        model.save_model(weights_complete_path)
        print(f"File '{weights_complete_path}' created successfully. Weights saved.")
