import torch
from torch import nn, optim
from torch.nn.functional import tanh, softmax
from tqdm import tqdm

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
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
