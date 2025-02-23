from __future__ import annotations

import matplotlib.pyplot as plt
from functions import *
from VQC import VQC
import tkinter as tk
from gui_loader import MetadataGUI
from pathlib import Path
from sklearn.metrics import auc

current_file = Path(__file__)

root = tk.Tk()
root.title("Metadata GUI")
gui = MetadataGUI(root, "metadata.json")
root.mainloop()

dark_background = True
if dark_background == True:
    style_path = current_file.parent.parent / 'transparent.mplstyle'
    suffix = '_transparent'
     #plt.xlabel("False Positive Rate", color='#ffffff', fontsize=26)
    # plt.xlabel("False Positive Rate", color='#ffffff')
    # plt.ylabel("True Positive Rate", color='#ffffff')

else:
    style_path = current_file.parent.parent / 'normal_plot.mplstyle'
    suffix = '_normal'
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")

plt.style.use(style_path)

colors = ['#ae77d6', '#76f3fb','#5a1387','#729bcb']

# Set the default color cycle for matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

plt.plot([0, 1], [0, 1], color= '#cccccc' , lw=3, linestyle='--', label="Random guess", zorder = 0)


if gui.result:
    metadata_list = gui.result

    for i, result in enumerate(metadata_list):
        if result:
            metadata = result['metadata']
            abs_model_path = result['model_file']
            abs_dataset_path = metadata['dataset_file']

            data = data_split(abs_dataset_path, 'Status_Dead')  
            column_names = list(data['X_train'].columns)
            num_features = data['X_train'].shape[1]

            model = VQC(num_wires=metadata['features'], num_outputs=1, num_layers=metadata['layers'], encoding=metadata['encoding'], reuploading=False) 
            model.load_model(abs_model_path)  # Load model weights

            # Add things to the legend

            # Evaluate the model and get the ROC curve data
            fpr, tpr, _ = model.evaluate_model(data, draw_roc=False)
            
            # Compute the AUC
            auc_value = auc(fpr, tpr)

            # Build the label
            label = f"{metadata['features']} features, {metadata['layers']} layers, AUC = {auc_value:.3f}"
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, label = label)

        else:
            print("No result found")
else:
    print("No result found")

# Add title and labels
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize = 16)
plt.show()

# change the name of the file
file = 'comparison' + suffix 

print(f"Please complete the file name: {file}")
# Prompt the user to input the remaining part of the string
personalized = input()

file = file + personalized + '.png'

saving_path = current_file.parent.parent / 'results' / file
# Check if the file already exists
if saving_path.exists():
    print(f"The file '{saving_path.name}' already exists.")
else:
    # Save the final DataFrame if the file does not exist
    plt.savefig(saving_path)






