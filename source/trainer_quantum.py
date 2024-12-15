from __future__ import annotations

from pathlib import Path
from functions import *
from itertools import product
import numpy as np

from VQC import VQC

current_file = Path(__file__)
input = current_file.parent.parent / 'data' / 'diabetes_cleaned.csv'
print('I am using the data file at the path:' + str(input))

data = data_split(input)  

encoding_options = ["angle"]
layer_options = [10]
ansatz_options= ["strong", "weak"]
learning_rates_options = [1e-3]

# we'll apply later
threshold_range = np.linspace(0, 1, num=20)

dict_keys = ['encoding', 'layers', 'ansatz', 'learning_rate']

combinations = list(product(encoding_options, layer_options, ansatz_options, learning_rates_options))
dict_combinations = [dict(zip(dict_keys, tpl)) for tpl in combinations]


for i in dict_combinations:
    model = VQC(num_wires=8, num_outputs=1, num_layers=i['layers'], encoding=i['encoding'], reuploading=False) 
    model.train(model, data , epochs=1000, lr=i['learning_rate'], verbose = False)


    filename = f"quantum_weights-{i['encoding']}-{i['ansatz']}-{i['layers']}-{i['learning_rate']}.pth"
    model_path_dict = current_file.parent.parent / 'data' / 'weights' / 'quantum'/ filename
    
    model.save_model(model_path_dict)
