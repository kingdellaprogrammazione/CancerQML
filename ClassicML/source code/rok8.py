import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Use a non-interactive backend if desired, e.g. Agg for PNG
matplotlib.use("Agg")

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class ImprovedClassicalNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(ImprovedClassicalNet, self).__init__()

        # Input layer with batch norm
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.output_layer(x)
        return x

def data_split(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def train_and_evaluate(data, model, optimizer, criterion, epochs, batch_size=32):
    X_train = torch.FloatTensor(data['X_train'].values)
    y_train = torch.FloatTensor(data['y_train'].values)
    X_test = torch.FloatTensor(data['X_test'].values)
    y_test = data['y_test'].values

    # Create data loader for batch training
    dataset = torch.utils.data.TensorDataset(X_train, y_train.view(-1, 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy().ravel()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc

def main():
    GUI = True
    current_file = Path(__file__)

    if GUI:
        root = tk.Tk()
        root.withdraw()
        input_path = filedialog.askopenfilename(title="Select the data file")
        input_file = Path(input_path)
    else:
        input_file = 'roc8.csv'

    print('Using data file at path: ' + str(input_file))

    data = data_split(input_file, 'Status_Dead')
    num_features = data['X_train'].shape[1]

    configs = [
        {'layers': 5, 'lr': 0.001},
        {'layers': 10, 'lr': 0.001},
        {'layers': 15, 'lr': 0.001},
        {'layers': 20, 'lr': 0.001}
    ]

    # Prepare figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # We'll store the points in a list of dicts so we can output them to a CSV
    csv_data = []  # Will hold rows of [config_label, fpr, tpr, thresholds]

    colors = ['blue', 'orange', 'green', 'red']
    for config, color in zip(configs, colors):
        model = ImprovedClassicalNet(
            num_features=num_features,
            hidden_size=64,
            num_layers=config['layers']
        )

        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
        criterion = nn.BCELoss()

        fpr, tpr, thresholds, roc_auc = train_and_evaluate(
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epochs=200,
            batch_size=32
        )

        # Plot
        ax.plot(fpr, tpr, color=color,
                label=f'Layers={config["layers"]}, LR={config["lr"]}, AUC={roc_auc:.2f}')

        # Store data for CSV
        for i in range(len(fpr)):
            csv_data.append({
                'config': f'{config["layers"]}layers_lr{config["lr"]}',
                'fpr': fpr[i],
                'tpr': tpr[i],
                'threshold': thresholds[i]
            })

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    leg = ax.legend(loc='lower right')
    leg.get_frame().set_facecolor('none')  # Make legend background transparent
    ax.grid(True)

    # Make the figure background and axes transparent
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    # Save the figure with transparent background
    plt.savefig('roc_curves.png', transparent=True, bbox_inches='tight')
    plt.close(fig)  # Close the figure if you don't need to display it

    # Save points to CSV
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv('roc8.csv', index=False)
    print("Saved ROC plot as 'roc_curves.png' and data points to 'roc_points.csv'.")

if __name__ == "__main__":
    main()
