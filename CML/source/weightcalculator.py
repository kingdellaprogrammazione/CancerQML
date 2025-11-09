import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif, SelectKBest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION VARIABLES ---
DATA_FILEPATH = 'CML/data/data8f.csv' # <<<--- INPUT CSV FILE PATH HERE
EPOCHS = 30             
BATCH_SIZE = 16
# IMPORTANT: CHANGE THIS VALUE (8, 10, or 12)
K_FEATURES = 12         
BLOCK_COUNTS = [5, 10, 15, 20] 

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- PyTorch Residual Block Definition ---
class ResidualBlock(nn.Module):
    """The skip-connection block that forms the basis of the Residual Network."""
    def __init__(self, in_features, internal_features):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, internal_features)
        self.bn1 = nn.BatchNorm1d(internal_features)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(internal_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        residual = x # store input
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual # add skip connection
        out = self.relu(out)
        return out

# --- PyTorch Residual Network (ResNet) Model ---
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
        
        # create sequence of resblocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(in_features=64, internal_features=32))
        self.residual_blocks = nn.Sequential(*blocks)
        
        # final output layer
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

# --- Training Function ---
def train_and_save_model(input_dim, num_blocks, train_loader, file_name):
    """Initializes, trains, and saves a PyTorch ResidualNet model."""
    
    model = ResidualNet(input_dim, num_blocks).to(DEVICE)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    print(f"\n--- Training Model with {num_blocks} Blocks ---")

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # save model's trained weights
    torch.save(model.state_dict(), file_name)
    print(f"Training complete. Model saved to: {file_name}")
    return model

# --- Data Preparation and Weight Calculation ---
print("--- Data Loading and Weight Analysis ---")

# --- Data Loading ---
try:
    df = pd.read_csv(DATA_FILEPATH)
except FileNotFoundError:
    print(f"FATAL ERROR: CSV file not found at '{DATA_FILEPATH}'. Please update DATA_FILEPATH.")
    exit()

X = df.drop(columns=['Status_Dead'])
y = df['Status_Dead']

# impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# calculate feature weights: ANOVA F-value
selector = SelectKBest(f_classif, k='all')
selector.fit(X_imputed, y)

raw_scores = selector.scores_
feature_names = X.columns
total_score = np.sum(raw_scores)
weight_percentages = (raw_scores / total_score) * 100
weights = {name: round(weight, 2) for name, weight in zip(feature_names, weight_percentages)}
sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

# save weights to JSON file (DYNAMIC FILENAME ADDED HERE)
output_filepath = f'CML/weights/feature_weight_{K_FEATURES}.json' # <-- MODIFIED LINE
with open(output_filepath, 'w') as f:
    json.dump(sorted_weights, f, indent=4)
print(f"Feature weights calculated and saved to '{output_filepath}'.")

# feature selection
sorted_features = sorted([item for item in sorted_weights.items()], key=lambda item: item[1], reverse=True)
selected_features = [name for name, weight in sorted_features[:K_FEATURES]]
print(f"Selected {K_FEATURES} features for training: {selected_features}")

# fin data split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed_df, y, test_size=0.3, random_state=42, stratify=y
)

X_train_final = X_train[selected_features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)

# convert to tensors and dataloader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE) 

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. Train and Save Models for Multiple Depths ---
print("\n--- Starting PyTorch Model Training and Saving ---")

input_dimension = X_train_scaled.shape[1]

# loop through block counts and train a model for each
for num_blocks in BLOCK_COUNTS:
    # FILENAME includes K_FEATURES so the predictor knows which file to load!!!
    file_name = f'CML/weights/resnet_weights_{K_FEATURES}_{num_blocks}blocks.pth'
    train_and_save_model(
        input_dimension, 
        num_blocks, 
        train_loader, 
        file_name
    )

print(f"\nTraining for K_FEATURES={K_FEATURES} complete. Run again with 8,10 or 12 features.")