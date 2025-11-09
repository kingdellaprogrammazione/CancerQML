import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc
import torch
import torch.nn as nn
import json

# --- CONFIGURATION VARIABLES ---
FEATURE_COUNTS_TO_EVAL = [8, 10, 12] # Feature sets to evaluate

# --- DATA FILE MAPPING ---
DATA_FILEPATHS = {
    8: 'CML/data/data8f.csv',   # <<<--- UPDATE THESE FILE NAMES
    10: 'CML/data/data10f.csv', # <<<--- UPDATE THESE FILE NAMES
    12: 'CML/data/data12f.csv', # <<<--- UPDATE THESE FILE NAMES
}

# block counts to compare
EVALUATION_BLOCK_COUNTS = [5, 10, 15, 20] 

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. PyTorch Model Definitions (Residual Network) ---

class ResidualBlock(nn.Module):
    """The skip-connection block that forms the basis of the Residual Network."""
    def __init__(self, in_features, internal_features):
        super(ResidualBlock, self).__init__()
        
        # two linear layers with batch normalization
        self.fc1 = nn.Linear(in_features, internal_features)
        self.bn1 = nn.BatchNorm1d(internal_features)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(internal_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        residual = x # store input for skip
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual # add skip connection (residual link)
        out = self.relu(out)
        return out

class ResidualNet(nn.Module):
    """The main deep learning model using a configurable number of Residual Blocks."""
    def __init__(self, input_dim, num_blocks):
        super(ResidualNet, self).__init__()
        
        # initial projection to the block dimension (64)
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # create sequence of residual blocks
        blocks = []
        for _ in range(num_blocks):
            # all blocks operate on the 64-dimension space
            blocks.append(ResidualBlock(in_features=64, internal_features=32))
        self.residual_blocks = nn.Sequential(*blocks)
        
        # fin output layer projects back to 1 for binary classification
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        # fin sigmoid activation for binary probability output
        return torch.sigmoid(x)


# --- Utility Function: Multi-Model ROC Plotting ---
def plot_roc_curve(roc_data_list, block_count):
    """
    Plots multiple ROC curves (comparing feature counts) on a single graph 
    with transparent background and white text.
    """
    # figure and axes background transparent/none
    plt.figure(figsize=(8, 8), facecolor='none') 
    ax = plt.gca()
    ax.set_facecolor('none') 
    
    text_color = 'white'

    # loop and plot ROC
    for fpr, tpr, auc_score, label in roc_data_list:
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.4f})')

    # plot random guess line
    plt.plot([0, 1], [0, 1], 'w--', label='Random Guess (AUC = 0.50)', linewidth=1)
    
    # set style and labels to white
    ax.tick_params(colors=text_color, which='both')
    
    plt.xlabel('False Positive Rate (FPR)', color=text_color)
    plt.ylabel('True Positive Rate (TPR)', color=text_color)
    plt.title(f'Feature Set Comparison (Fixed Depth: {block_count} Blocks)', color=text_color)
    
    # spine (border) color white
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    
    # legend box transparency, text color manual set
    legend = plt.legend() 
    if legend:
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')
        # color of the legend text objects
        for text in legend.get_texts():
            text.set_color(text_color)

    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.show()

# --- MAIN EXECUTION ---
print("--- Starting Cross-Feature Comparison by Block Depth ---")

# outer loop: depth for current plot
for block_count in EVALUATION_BLOCK_COUNTS:
    
    roc_data_for_current_block = []
    print(f"\n--- Processing models with {block_count} fixed blocks ---")

    # inner loop: evaluate models for each feature count
    for K_FEATURES in FEATURE_COUNTS_TO_EVAL:
        
        print(f"  -> Preparing data for {K_FEATURES} features...")
        
        # --- 2. Data Preparation for this K_FEATURES set ---
        data_filepath = DATA_FILEPATHS.get(K_FEATURES)
        if not data_filepath:
            print(f"  -> ERROR: No CSV file defined for {K_FEATURES} features.")
            continue
        
        try:
            df = pd.read_csv(data_filepath)
        except FileNotFoundError:
            print(f"  -> ERROR: CSV file not found at '{data_filepath}'.")
            continue

        # load weights
        weights_filepath = f'CML/weights/feature_weight_{K_FEATURES}.json'
        try:
            with open(weights_filepath, 'r') as f:
                feature_weights = json.load(f)
        except FileNotFoundError:
            print(f"  -> ERROR: Weights file '{weights_filepath}' not found.")
            continue
            
        # separate features and target, impute, split
        X_full = df.drop(columns=['Status_Dead'])
        y = df['Status_Dead']
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed = imputer.fit_transform(X_full)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X_full.columns)
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_imputed_df, y, test_size=0.3, random_state=42, stratify=y
        )

        # feature selection
        sorted_features = sorted([item for item in feature_weights.items()], key=lambda item: item[1], reverse=True)
        selected_features = [name for name, weight in sorted_features[:K_FEATURES]]
        
        # subset and scale test data
        X_train_final = X_train_full[selected_features]
        X_test_final = X_test_full[selected_features]
        scaler = StandardScaler()
        scaler.fit(X_train_final)
        X_test_scaled = scaler.transform(X_test_final)

        # convert to tensors
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        input_dim = X_test_scaled.shape[1] 

        # --- 3. Model Loading and Evaluation ---
        
        file_name = f'CML/weights/resnet_weights_{K_FEATURES}_{block_count}blocks.pth' # FIX THIS
        label = f'{K_FEATURES} Features'
        
        model = ResidualNet(input_dim, block_count).to(DEVICE)

        try:
            model.load_state_dict(torch.load(file_name, map_location=DEVICE, weights_only=True))
        except (FileNotFoundError, RuntimeError) as e:
            # Catch file not found OR dimension mismatch errors
            error_type = "File Not Found" if isinstance(e, FileNotFoundError) else "Runtime (Size Mismatch)"
            print(f"    -> SKIPPING {K_FEATURES}F: {error_type} for '{file_name}'.")
            continue

        # predict probabilities
        model.eval() 
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
            
        y_test_np = y_test_tensor.cpu().numpy().flatten()

        # calculate ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # store for plotting
        roc_data_for_current_block.append((fpr, tpr, roc_auc, label))
        print(f"    -> OK: {K_FEATURES}F model loaded. AUC: {roc_auc:.4f}")


    # 4. Plot the ROC curves for the fixed block count
    if roc_data_for_current_block:
        plot_roc_curve(roc_data_for_current_block, block_count=block_count)
    else:
        print(f"No models were successfully loaded for the {block_count}-block comparison.")

# --- 5. Conclusion ---
print("\n--- Conclusion ---")
print("All cross-feature comparisons by block depth are complete.")