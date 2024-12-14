from __future__ import annotations
import joblib 
from pathlib import Path
from functions import *
import joblib
from itertools import product
from sklearn.metrics import classification_report

current_file = Path(__file__)
print(current_file) 
input = current_file.parent.parent / 'data' / 'diabetesrenewed.csv'
print(input)

X_train, X_test, y_train, y_test, X_valid, y_valid = prepare_angle_data(input)  
# this automates data extraction, for angle

now_encoding = ["angle"]
now_num_layers = [10,20,30]
now_ansatz= ["Strong"]
learning_rates = [1e-3, 5e-4, 1e-4]

threshold =[ 0.48, 0.5 ,0.52]

combinations = list(product(now_encoding, now_num_layers, now_ansatz, learning_rates))
# Add counter to each combination
lista_tot = [list(combo) + [idx + 1] for idx, combo in enumerate(combinations)]


for i in lista_tot:
    model = VQC(num_wires=8, num_outputs=1, num_layers=i[1], encoding=i[0], reuploading=False) 
    model = train(model, X_train, y_train,X_valid, y_valid, epochs=1000, lr=i[3], verbose = False)
    divider= '-'
    filename_weights='model' 
    filename_confusion = 'model' 
    for j in i:
        filename_weights += divider
        filename_weights += str(j)
        filename_confusion += divider
        filename_confusion += str(j)

    filename_weights += '-weights.pth'

    model_path_dict = current_file.parent.parent / 'models' / '8features' / 'raw'/ filename_weights
    
    
    for h in  threshold:
        evaluation(model, X_test, y_test, threshold_classification=float(h))
        
        conf_matrix = get_confusion(model, X_test, y_test, threshold_classification=h)
        class_report = get_report(model, X_test, y_test, threshold_classification=h)
        # print(class_report)
        # Save everything into a dictionary
        evaluation_results = {
            "confusion_matrix": conf_matrix , # Convert to list for easy saving in joblib
            "classification_report": class_report
        }   
        final_filename = filename_confusion + '-' + str(h) + '-report.pkl'
        
        model_path_results = current_file.parent.parent / 'models' / '8features' / 'raw' /  final_filename
#        Save the results using joblib
        joblib.dump(evaluation_results, model_path_results)

        readable_filename = filename_confusion +  '-' + str(h) + '-readablereport.txt'
        model_path_results_readable = current_file.parent.parent / 'models' / '8features' / readable_filename
        with open(model_path_results_readable, 'w+') as file:
            file.write("Confusion Matrix:\n")
            file.write(str(conf_matrix))  # Convert confusion matrix to string for saving

            file.write("\n\nClassification Report:\n")
            file.write(class_report)  # The classification report is already a string
            file.close()

    model.save_model(model_path_dict)
