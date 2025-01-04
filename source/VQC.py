from __future__ import annotations  # PEP 563: Delayed type hint evaluation

# Standard Library Imports
import math
from typing import Any, Dict

# Third-Party Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import torch
from torch import nn, load, save
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler


class VQC(nn.Module):
    def __init__(
        self,
        num_wires: int,
        num_outputs: int,
        num_layers: int,
        encoding: str = "angle",
        reuploading: bool = False,
    ) -> None:
        super().__init__()
        """
        Constructor
        @encoding: String which represents the gates used for the Angle encoding
        @Ansatz: String which represents the ansatz used for quantum circuit
        @Reuploading: Boolean indicating whether or not to use reuploading
        @hadamard: Boolean indicating whether or not to use Hadamard gates
        @num_layers: Integer representing the number of layers in the quantum circuit
        @num_wires: Integer representing the number of wires in the quantum circuit
        @num_outputs: Integer representing the number of output qubits
        @gate_used: String representing the encoding gate used
        @name_ansatz: String representing the ansatz used
        """
        self.encoding = encoding
        self.reuploading = reuploading
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.num_outputs = num_outputs
        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.num_wires)
        

        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_shapes = {"weights": (self.num_layers, self.num_wires, 3)}
        # Create the quantum node
        self.qnode = self.create_qnode()
        # Define the quantum layer in PyTorch
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes).to(self.device)

    def create_qnode(self) -> qml.QNode:
        """Creates the quantum node for the hybrid model."""

        @qml.qnode(self.dev)
        def qnode(inputs: torch.Tensor, weights: torch.nn.parameter.Parameter) -> list[Any]:
            # Encoding and Ansatz logic
            if self.reuploading:
                if self.encoding == "angle":
                    for w in weights:
                        self.encoding_circuit(inputs)
                        self.apply_ansatz(w.unsqueeze(0))
                elif self.encoding == "amplitude":
                    msg = "Amplitude encoding is not supported with re-uploading."
                    raise ValueError(msg)
            else:
                self.encoding_circuit(inputs)
                self.apply_ansatz(weights)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_outputs)]

        return qnode

    def apply_ansatz(self, weights: torch.nn.parameter.Parameter) -> None:
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_wires))   ## here

    def encoding_circuit(self, inputs: torch.Tensor) -> None:
        """
        Apply encoding circuit based on the specified encoding method.
        @ inputs: array of input values in range [-1, 1]
        """
        if self.encoding == "angle":
            qml.AngleEmbedding(math.pi / 2 * inputs, wires=range(self.num_wires), rotation='Y')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model."""
        return self.qlayer(inputs)
            
    def load_model(self, filepath: str):
        # Recreate the model with the same architecture
        self.load_state_dict(torch.load(filepath, weights_only=True))
        self.eval()

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def train_model(
        self,
        data: Dict[str, torch.Tensor] ,
        optimizer: torch.optim.Optimizer = None,
        criterion: None | nn.Module = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr = 0.01,
        early_stopping: bool = True,
        patience: int = 5,
        threshold_classification: float = 0.5,
        verbose: bool = True,
    ):  
        
        if self.encoding == 'angle':
            data_angled_encoded = angle_encode_data(data)

        x_train = data_angled_encoded['X_train']
        y_train = data_angled_encoded['y_train']
        x_valid = data_angled_encoded['X_valid']
        y_valid = data_angled_encoded['y_valid']

        # Initialize metrics and losses
        if criterion is None:
            criterion = nn.BCELoss() if self.num_outputs == 1 else nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        min_loss = float("inf")
        patience_counter = 0
        best_model_state = self.state_dict()  # Track best model state

        x_train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            self.train()
            loss_res = 0
            for x_batch, y_batch in x_train_dataloader:            
                optimizer.zero_grad()
                y_pred = (1 - self(x_batch)) / 2  # convert range from -1, 1 to 0, 2
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()
                loss_res += loss.item()
            loss_res /= len(x_train_dataloader)

            if early_stopping and x_valid is not None and y_valid is not None:
                self.eval()
                with torch.no_grad():
                    y_pred_valid = (1 - self(x_valid)) / 2
                    loss_valid = criterion(y_pred_valid, y_valid)

                if loss_valid < min_loss:
                    min_loss = loss_valid
                    patience_counter = 0
                    best_model_state = self.state_dict()  # Save best model state
                else:
                    patience_counter += 1

                # Early stopping condition
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.load_state_dict(best_model_state)  # Restore best model
                    break

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_res:.4f}")
                print("Train")
                self.evaluate_model(
                    data,
                    str_x_eval='X_valid',
                    str_y_eval='y_valid',
                    threshold_classification=threshold_classification
                )
                if x_valid is not None and y_valid is not None:
                    print("Validation")
                    self.evaluate_model(
                    data,
                    str_x_eval='X_valid',
                    str_y_eval='y_valid',
                    threshold_classification=threshold_classification
                    )

    def evaluate_model(
        self,
        data: Dict[str, torch.Tensor],
        str_x_eval = 'X_test',
        str_y_eval = 'y_test',
        threshold_classification: float = 0.5,
        draw_roc = 'False'
    ):
        """
        Evaluate the model
        @model: nn.Module, model to evaluate, MANDATORY
        @x: torch.Tensor, data, MANDATORY
        @y: torch.Tensor, labels, MANDATORY
        @criterion: None|nn.Module, loss function
        @threshold_classification: float, threshold for classification
        @metrics: None|list[str], list of metrics to evaluate
        """
        if self.encoding == 'angle':
            data_angled_encoded = angle_encode_data(data)
        
        x = data_angled_encoded[str_x_eval]
        y =  data_angled_encoded[str_y_eval]
        with torch.no_grad():
            # function needed for VQC
            y_pred = (1 - self(x)) / 2
        print(classification_report(y, y_pred > threshold_classification))
        print(confusion_matrix(y, y_pred > threshold_classification))   

        if (draw_roc == 'True'):
            fpr, tpr, thresholds = roc_curve(y, y_pred)
            roc_auc = auc(fpr, tpr)  # Compute Area Under the Curve (AUC)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            plt.grid()
            plt.show()
        

    
def angle_encode_data(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    # Initialize MinMaxScaler with a range of (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the training data, then transform the test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)
    # Convert the arrays into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype=torch.float32).view(-1, 1)
    # Return all six datasets
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test, 
        'X_valid': X_valid, 
        'y_valid': y_valid
    }


def get_report(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    threshold_classification: float = 0.5
):
    """
    Evaluate the model
    @model: nn.Module, model to evaluate, MANDATORY
    @x: torch.Tensor, data, MANDATORY
    @y: torch.Tensor, labels, MANDATORY
    @criterion: None|nn.Module, loss function
    @threshold_classification: float, threshold for classification
    @metrics: None|list[str], list of metrics to evaluate
    """
    model.eval()
    with torch.no_grad():
        # function needed for VQC
        y_pred = (1 - model(x)) / 2
    return classification_report(y, y_pred > threshold_classification)

def get_confusion(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    threshold_classification: float = 0.5
):
    """
    Evaluate the model
    @model: nn.Module, model to evaluate, MANDATORY
    @x: torch.Tensor, data, MANDATORY
    @y: torch.Tensor, labels, MANDATORY
    @criterion: None|nn.Module, loss function
    @threshold_classification: float, threshold for classification
    @metrics: None|list[str], list of metrics to evaluate
    """
    model.eval()
    with torch.no_grad():
        # function needed for VQC
        y_pred = (1 - model(x)) / 2
    return confusion_matrix(y, y_pred > threshold_classification)


