import torch
from torch import nn, optim
from torch.nn.functional import tanh, softmax
from tqdm import tqdm


class FeedForward(nn.Module):
    """
    A custom PyTorch Module class for a feed-forward neural network.

    Attributes:
        layer1 (nn.Linear): The first linear layer of the neural network.
        layer2 (nn.Linear): The second linear layer of the neural network.
        layer3 (nn.Linear): The third linear layer of the neural network.

    Methods:
        __init__(input_size, hidden_size, output_size): Initializes the FeedForward object.
        forward(x): Defines the forward pass of the neural network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the FeedForward object.

        Args:
            input_size (int): The size of the input vector.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The size of the output vector.
        """
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (Tensor): The input vector.

        Returns:
            Tensor: The output of the neural network.
        """
        x = tanh(self.layer1(x))
        x = tanh(self.layer2(x))
        x = softmax(self.layer3(x), dim=-1, dtype=torch.float32)
        return x


def train_epoch(model, loader, loss_fn, optimizer, epoch, num_epochs=256):

    model.train()
    total, num_correct = 0, 0
    for images, labels in tqdm(loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):

        images = images.view(images.size(0), -1)  # flatten images

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # make prediction by choosing most likely outcome
        predictions = torch.argmax(outputs, dim=-1)

        # keep tabs on accuracy score
        num_correct += (predictions == labels).sum().item()
        total += labels.shape[0]

    return num_correct / total


def train_model(model, loader, loss_fn, optimizer,
                num_epochs=256, verbose=False, tol=1-(1e-6),
                val_loader=None, switch_loader=None, switch_val_loader=None,
                switch_epoch=None, model_type='basic'):
    """
    Description:
        Trains the given model on the training data and optionally switches to a different
        data loader after a specified number of epochs.

    Parameters:
        model (nn.Module): The neural network model.
        loader (DataLoader): The data loader for the initial training data.
        loss_fn (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int, optional): The total number of epochs (default: 256).
        verbose (bool, optional): Whether to print the training accuracy for each epoch (default: False).
        tol (float, optional): The tolerance for early stopping based on validation accuracy (default: 1-(1e-6)).
        val_loader (DataLoader, optional): The data loader for the validation data.
        switch_loader (DataLoader, optional): The data loader to switch to after the specified number of epochs.
        switch_val_loader (DataLoader, optional): The data loader for the validation data after switching train data loader.
        switch_epoch (int, optional): The epoch number at which to switch to the new data loader.
        model_type (str, optional): The type of the model (default: 'basic').

    Returns:
        Tuple[nn.Module, float]: The trained model and the best validation accuracy

    Notes:
        This function can be used for curriculum training, where the model is first trained
        on a simpler dataset and then switched to a more complex dataset after a certain
        number of epochs.
    """
    best_acc = float('-inf')
    switched = False
    for epoch in range(num_epochs):
        val_acc = None
        if switch_epoch is not None and epoch >= switch_epoch:
            if not switched:
                switched = True
                print(
                    "Curriculum training on basic data complete. Now training curriculum model on complex data...")
            train_acc = train_epoch(
                model, switch_loader, loss_fn, optimizer, epoch, num_epochs)
            val_acc = eval_model(model, switch_val_loader,
                                 model_type=model_type, data_type='validation')
        else:
            train_acc = train_epoch(
                model, loader, loss_fn, optimizer, epoch, num_epochs)
            val_acc = eval_model(
                model, val_loader, model_type=model_type, data_type='validation')

        if verbose:
            print(f'Epoch {epoch + 1}, Training Accuracy: {train_acc}')

        best_acc = val_acc if val_acc > best_acc else best_acc

        # early exit based on validation accuracy
        if best_acc > tol:
            break

    print(f"Model validation accuracy after training: {best_acc}")
    return model, best_acc


def eval_model(model, loader, model_type='basic', data_type='test', verbose=False):
    """
    Description:
        Evaluates the given model on the provided data loader.

    Parameters:
        model (nn.Module): The neural network model.
        loader (DataLoader): The data loader for the evaluation data.
        model_type (str, optional): The type of the model (default: 'basic').
        data_type (str, optional): The type of the data (default: 'test').
        verbose (bool, optional): Whether to print the evaluation accuracy (default: False).

    Returns:
        float: The accuracy of the model on the evaluation data.
    """
    model.eval()
    total, num_correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1)  # flatten images
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=-1)
            num_correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    acc = num_correct/total
    if verbose:
        print(f"{model_type} {data_type} accuracy: {acc}")
    return acc


def grid_search(param_grid, loader, input_size, output_size=3, val_loader=None):
    """
    Description:
        Performs a grid search over the specified hyperparameters to find the best
        combination for the given data.

    Parameters:
        param_grid (dict): A dictionary containing lists of values for the hyperparameters.
        loader (DataLoader): The data loader for the training data.
        input_size (int): The size of the input vector.
        output_size (int, optional): The size of the output vector (default: 3).
        val_loader (DataLoader, optional): The data loader for the validation data.

    Returns:
        Tuple[dict, float]: The best hyperparameters and the corresponding validation accuracy.
    """
    best_params = None
    best_acc = float('-inf')

    for hidden_size in param_grid['hidden_sizes']:
        for learning_rate in param_grid['learning_rates']:
            print(
                f'Training with learning rate={learning_rate} and hidden size={hidden_size}')
            model = FeedForward(input_size=input_size,
                                hidden_size=hidden_size, output_size=output_size)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            model, _ = train_model(
                model, loader, criterion, optimizer, val_loader=val_loader, model_type='complex')

            val_acc = eval_model(
                model, val_loader, model_type='complex', data_type='validation')
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'hidden_size': hidden_size,
                               'learning_rate': learning_rate}

    return best_params, best_acc
