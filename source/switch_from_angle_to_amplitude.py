# how to change encoding from angle to amplitude 

#Replace qml.AngleEmbedding with qml.AmplitudeEmbedding


#update code 
def encoding_circuit(self, inputs: torch.Tensor) -> None:
    """
    Apply encoding circuit based on the specified encoding method.
    @ inputs: array of input values
    """
    if self.encoding == "angle":
        qml.AngleEmbedding(math.pi / 2 * inputs, wires=range(self.num_wires), rotation='Y')
    elif self.encoding == "amplitude":
        # Normalize inputs to ensure unit norm
        inputs = inputs / torch.norm(inputs, p=2, dim=-1, keepdim=True)

        # Validate input size
        expected_size = 2 ** self.num_wires
        if inputs.shape[-1] != expected_size:
            raise ValueError(
                f"Input size for amplitude encoding must be {expected_size}, "
                f"but got {inputs.shape[-1]}"
            )

        # Apply Amplitude Embedding
        qml.AmplitudeEmbedding(inputs, wires=range(self.num_wires), normalize=False)
    else:
        raise ValueError(f"Unsupported encoding type: {self.encoding}")


#Dataset Preprocessing for Amplitude Encoding

# Pad or truncate features to match the expected size
num_qubits = 3  # Example: 3 qubits
expected_size = 2 ** num_qubits

# Assuming X_train, X_valid, and X_test are tensors
def adjust_feature_size(X, expected_size):
    num_features = X.shape[1]
    if num_features < expected_size:
        # Pad with zeros
        padding = expected_size - num_features
        X = torch.cat([X, torch.zeros((X.shape[0], padding))], dim=1)
    elif num_features > expected_size:
        # Truncate extra features
        X = X[:, :expected_size]
    return X

X_train = adjust_feature_size(X_train, expected_size)
X_valid = adjust_feature_size(X_valid, expected_size)
X_test = adjust_feature_size(X_test, expected_size)


#training model 
model = VQC(num_wires=3, num_outputs=1, num_layers=16, encoding="amplitude", reuploading=False)
