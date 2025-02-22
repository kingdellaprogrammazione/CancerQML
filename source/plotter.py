from __future__ import annotations

import matplotlib.pyplot as plt
from functions import *
from VQC import VQC
import tkinter as tk
from gui_loader import MetadataGUI

root = tk.Tk()
root.title("Metadata GUI")
gui = MetadataGUI(root, "metadata.json")
root.mainloop()

transparent = False
plt.figure(figsize=(16, 12))

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
            
            # Plot the ROC curve
            plt.plot(fpr, tpr)

        else:
            print("No result found")
else:
    print("No result found")

# Add title and labels
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()


if transparent == True:
    plt.plot([0, 1], [0, 1], color= '#ffffff' , lw=3, linestyle='--', label="Random guess", zorder = 0)
    plt.xlabel("False Positive Rate", color='#ffffff', fontsize=26)
    plt.ylabel("True Positive Rate", color='#ffffff', fontsize=26)
    ax = plt.gca()  # Get the current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.title("Receiver Operating Characteristic (ROC) Curve", color='#ffffff', fontsize=20)
      # Add legend with white text
    legend = plt.legend(loc="lower right", fontsize=24)
    plt.setp(legend.get_texts(), color='#ffffff')  # Set legend text color to white
    legend.get_frame().set_facecolor('none')       # Make legend background transparent
    legend.get_frame().set_edgecolor('none')       # Remove legend borde
    # Customize grid and background
    plt.grid(color='#ffffff', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().patch.set_alpha(0)  # Make the axes background transparent
    plt.gcf().patch.set_alpha(0)  # Make the figure background transparent
    # Set tick colors
    plt.tick_params(colors='#ffffff', labelsize=20)

    #         colours = ['#ae77d6', '#76f3fb','#5a1387','#729bcb']
    # 

plt.plot([0, 1], [0, 1],color = '#cccccc' , lw=3, linestyle='--', label="Random guess", zorder = 0)

# Show the plot
plt.show()


