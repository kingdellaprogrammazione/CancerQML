def evaluation(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    threshold_classification: float = 0.5
) -> dict:
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
        print(y_pred)
    print(classification_report(y, y_pred > threshold_classification))
    print(confusion_matrix(y, y_pred > threshold_classification))    
    
def train(
    model: nn.Module,
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
) -> nn.Module:

    x_train = data['X_train']
    y_train = data['y_train']
    x_valid = data['X_valid']
    y_valid = data['y_valid']

    # Initialize metrics and losses
    if criterion is None:
        criterion = nn.BCELoss() if model.num_outputs == 1 else nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    min_loss = float("inf")
    patience_counter = 0
    best_model_state = model.state_dict()  # Track best model state

    x_train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        loss_res = 0
        for x_batch, y_batch in x_train_dataloader:            
            optimizer.zero_grad()
            y_pred = (1 - model(x_batch)) / 2  # convert range from -1, 1 to 0, 2
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            loss_res += loss.item()
        loss_res /= len(x_train_dataloader)

        if early_stopping and x_valid is not None and y_valid is not None:
            model.eval()
            with torch.no_grad():
                y_pred_valid = (1 - model(x_valid)) / 2
                loss_valid = criterion(y_pred_valid, y_valid)

            if loss_valid < min_loss:
                min_loss = loss_valid
                patience_counter = 0
                best_model_state = model.state_dict()  # Save best model state
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  # Restore best model
                break

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_res:.4f}")
            print("Train")
            evaluation(
                model=model,
                x=x_train,
                y=y_train,
                threshold_classification=threshold_classification
            )
            if x_valid is not None and y_valid is not None:
                print("Validation")
                evaluation(
                    model=model,
                    x=x_valid,
                    y=y_valid,
                    threshold_classification=threshold_classification
                )

    return model

