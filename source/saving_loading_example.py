from __future__ import annotations

from pathlib import Path

from itertools import product
from sklearn.metrics import classification_report
import numpy as np
from VQC import VQC
from functions import *

current_file = Path(__file__)
print(current_file) 
input = current_file.parent.parent / 'data' / 'diabete' / 'diabetes_cleaned.csv'
print(input)

data = data_split(input)  

model = VQC(num_wires=8, num_outputs=1, num_layers=10, encoding='angle', reuploading=False) 
model.train_model(data, epochs=1, lr=0.0001, verbose = False)
model.evaluate_model(data)

model_path_dict = current_file.parent / 'prova.test'

#torch.save(model, model_path_dict)
model.save_model(model_path_dict)

model_1 = VQC(num_wires=8, num_outputs=1, num_layers=10, encoding='angle', reuploading=False) 
model_1.load_model(model_path_dict)
model_1.evaluate_model(data)
