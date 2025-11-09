import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc
import torch
import torch.nn as nn

# --- SINGLE LINE CONFIGURATION SELECTOR ---
# change for to test different amount of features (8, 10, or 12).
TEST_FEATURE_COUNT = 8
DATA_FILEPATH = 'CML/data/data8f.csv' # <<<--- INPUT CSV FILE PATH HERE

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- DYNAMIC CONFIGURATION MAPPINGS ---
WEIGHT_MAPPINGS = {
    8: {
        'weights': {
    "Survival Months": 61.93,
    "Reginol Node Positive": 14.86,
    "N Stage_N1": 10.63,
    "Progesterone Status_Positive": 6.82,
    "Tumor Size": 3.87,
    "Marital Status_Married": 0.97,
    "Age": 0.66,
    "Regional Node Examined": 0.26
    }
    },
    10: {
        'weights': {
    "Survival Months": 55.12,
    "Reginol Node Positive": 13.23,
    "N Stage_N3": 10.79,
    "N Stage_N1": 9.46,
    "Progesterone Status_Positive": 6.07,
    "Tumor Size": 3.44,
    "Marital Status_Married": 0.86,
    "Age": 0.59,
    "Regional Node Examined": 0.23,
    "Marital Status_Divorced": 0.21
    }
    },
    12: {
        'weights': {
    "Reginol Node Positive": 60.52,
    "Tumor Size": 16.21,
    "T Stage _T2": 5.61,
    "Race_White": 5.26,
    "6th Stage_IIIC": 3.81,
    "N Stage_N3": 3.29,
    "Age": 2.74,
    "Marital Status_Married": 0.93,
    "Estrogen Status_Positive": 0.84,
    "Regional Node Examined": 0.69,
    "Progesterone Status_Positive": 0.08,
    "Marital Status_Single ": 0.02
    }
    }
}


try:
    K_FEATURES = TEST_FEATURE_COUNT
    feature_weights = WEIGHT_MAPPINGS[K_FEATURES]['weights']
except KeyError:
    print(f"Error: Invalid feature count {TEST_FEATURE_COUNT}. Using default (8 features).")
    K_FEATURES = 8
    feature_weights = WEIGHT_MAPPINGS[8]['weights']

# define models to be tested for the selected feature set
MODEL_CONFIGS = [
    {'blocks': 5, 'file': f'CML/weights/resnet_weights_{K_FEATURES}_5blocks.pth', 'label': f'{K_FEATURES}F / 5 Blocks'},
    {'blocks': 10, 'file': f'CML/weights/resnet_weights_{K_FEATURES}_10blocks.pth', 'label': f'{K_FEATURES}F / 10 Blocks'},
    {'blocks': 15, 'file': f'CML/weights/resnet_weights_{K_FEATURES}_15blocks.pth', 'label': f'{K_FEATURES}F / 15 Blocks'},
    {'blocks': 20, 'file': f'CML/weights/resnet_weights_{K_FEATURES}_20blocks.pth', 'label': f'{K_FEATURES}F / 20 Blocks'},
]


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
def plot_roc_curve(roc_data_list, feature_count):
    """
    Plots multiple ROC curves on a single graph with transparent background and white text.
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
    plt.title(f'ResNet Depth Comparison ({feature_count} Features)', color=text_color)
    
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

# --- 2. Data Preparation and Feature Selection ---
print("--- Data Preparation and Feature Selection ---")

try:
    df = pd.read_csv(DATA_FILEPATH)
except FileNotFoundError:
    print(f"FATAL ERROR: CSV file not found at '{DATA_FILEPATH}'. Please update DATA_FILEPATH.")
    exit()

print(f"Current Test Configuration: {K_FEATURES} Features")

# separate features and target, impute, split
X = df.drop(columns=['Status_Dead'])
y = df['Status_Dead']


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X_imputed_df, y, test_size=0.3, random_state=42, stratify=y
)

# feature selection
sorted_features_by_weight = sorted(
    [(weight, name) for name, weight in feature_weights.items()],
    reverse=True
)

selected_features = [name for weight, name in sorted_features_by_weight[:K_FEATURES]]

# subset and scale test data
X_test_final = X_test[selected_features]

print(f"Selected {K_FEATURES} features:\n{selected_features}\n")

scaler = StandardScaler()
X_train_temp = X_train[selected_features]
scaler.fit(X_train_temp)
X_test_scaled = scaler.transform(X_test_final)

# convert to tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
input_dim = X_test_scaled.shape[1] # input dimension is the number of selected features


# --- 3. Multi-Model Loading and Evaluation ---
print("\n--- Multi-Model Loading and Evaluation ---")
roc_data_list = []

# loop through different amount of blocks (5, 10, 15, 20 blocks)
for config in MODEL_CONFIGS:
    num_blocks = config['blocks']
    file_name = config['file']
    label = config['label']

    # initialize resnet
    model = ResidualNet(input_dim, num_blocks).to(DEVICE)

    # load trained weights
    try:
        model.load_state_dict(torch.load(file_name, map_location=DEVICE, weights_only=True))
        print(f"\n--- Evaluating Model: {label} ---")
    except FileNotFoundError:
        print(f"\nError: Weight file '{file_name}' not found. Did you train this {K_FEATURES}-feature model?")
        continue
    except RuntimeError as e:
        print(f"\nRuntime Error loading {file_name}: {e}")
        print(f"FAILURE: Input dimension mismatch. The file expects a model with {K_FEATURES} inputs.")
        continue


    # predict probabilities
    model.eval() 
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
        
    # convert test labels back to numpy for scikit-learn metrics
    y_test_np = y_test_tensor.cpu().numpy().flatten()

    # classification report
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    print(f"Classification Report ({label}):")
    print(classification_report(y_test_np, y_pred_class))

    # calculate ROC data
    fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"Area Under the ROC Curve (AUC) for {label}: {roc_auc:.4f}")

    # store for plots
    roc_data_list.append((fpr, tpr, roc_auc, label))

# --- 4. Plot all ROC curves together ---
if roc_data_list:
    plot_roc_curve(roc_data_list, feature_count=K_FEATURES)
else:
    print("No models were successfully loaded to plot the ROC curve.")

# --- 5. Conclusion ---
print("\n--- Conclusion ---")
print(f"The evaluation for the {K_FEATURES}-feature set is complete. To test a different set, change TEST_FEATURE_COUNT variable at the top of the script.")