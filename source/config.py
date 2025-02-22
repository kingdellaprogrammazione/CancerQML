import os
import re
import json

current_dir = os.getcwd()

# Go up to the project directory
while True:
    if os.path.basename(current_dir) == 'CancerQML':  # Replace 'project' with your project name
        break
    current_dir = os.path.dirname(current_dir)

# Define the weights directory
weights_dir = os.path.join(current_dir, 'weights')
data_dir = os.path.join(current_dir, 'data')

# Initialize an empty dictionary to store the model parameters
model_parameters = {}

# Iterate over the files in the weights directory
for root, dirs, files in os.walk(weights_dir):
    for file in files:
        # Get the file path
        file_path = os.path.join(current_dir, root, file)
        
        # Check if the file is a PyTorch model weight file
        if file.endswith('.pth'):
            # Extract the model parameters from the file name
            file_name = os.path.basename(file_path)
            parameters = re.split('-', os.path.splitext(file_name)[0])

            # Initialize an empty dictionary to store the current model parameters
            current_parameters = {}
            
            # Iterate over the parameters and extract the relevant information
            for parameter in parameters:
                if parameter == 'q_weights':
                    current_parameters['model_type'] = 'quantum'
                elif parameter == 'enc_ang':
                    current_parameters['encoding'] = 'angle'                    
                elif parameter == 'ans_strong':
                    current_parameters['ansatz'] = 'strong'
                elif re.match(r'lay_\d+', parameter):
                    current_parameters['layers'] = int(re.search(r'\d+', parameter).group()) 
                elif re.match(r'lr_\d+\.\d+', parameter):
                    current_parameters['learning_rate'] = float(re.search(r'\d+\.\d+', parameter).group())
                elif re.match(r'ep_\d+', parameter):
                    current_parameters['epochs'] = int(re.search(r'\d+', parameter).group())
                elif re.match(r'(\d+)f', parameter):
                    current_parameters['features'] = int(re.search(r'\d+', parameter).group())

                current_parameters['downsampling'] = 'downsampled' in parameters or 'downsampled_pca' in parameters
                current_parameters['pca'] = 'pca' in parameters or 'downsampled_pca' in parameters
            
            flag = 0
            for root_, dirs_, files_ in os.walk(data_dir):
                if flag == 1:
                    break
                for file_ in files_:
                    if file_.endswith('.csv'):
                        # Get the dataset name
                        dataset_name = os.path.basename(file_)
                        dataset_file_path_ = os.path.join(root_, file_)

                        parameters_ = re.split('_', os.path.splitext(dataset_name)[0])
                        if 'features' not in current_parameters:
                            continue
                        if (str(current_parameters['features'])+ 'f') in parameters_ and (current_parameters['downsampling'] == ( 'downsampled' in parameters_)) and (current_parameters['pca'] ==  ('pca' in parameters_)):
                            current_parameters['dataset_file'] = dataset_file_path_
                            flag = 1
                            break
                
            print(current_parameters)

                        # Append the current parameters to the model parameters dictionary
            model_parameters[file_path] = current_parameters

# Write the model parameters to the metadata file
with open(os.path.join(current_dir, 'metadata.json'), 'w') as f:
    json.dump(model_parameters, f, indent=4)