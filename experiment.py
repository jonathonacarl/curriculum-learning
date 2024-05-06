import torch
from torch import nn, optim
from model import FeedForward, train_model
import matplotlib.pyplot as plt


def curriculum_experiment(input_size, hidden_size, learning_rate,
                          loader_basic, loader_combined, switch_loader_name,
                          val_loader, loss_fn=nn.CrossEntropyLoss(), num_epochs=256,
                          gradual_switch=False, save_model=False, plot_title=None):

    gs = "gs" if gradual_switch else ""

    currciulum_accs = []
    print(
        f'Training curriculum models with {switch_loader_name} Switch Loader...')
    for i in range(0, 7 + 1):

        model_curriculum = FeedForward(input_size=input_size,
                                       hidden_size=hidden_size, output_size=3)

        optimizer_curriculum = optim.SGD(
            model_curriculum.parameters(), lr=learning_rate)

        switch_epoch = 0 if i == 0 else 2**i
        print(f'Training curriculum model at switch epoch={switch_epoch}')

        model_curriculum, acc = train_model(model=model_curriculum, loader=loader_basic,
                                            loss_fn=loss_fn, optimizer=optimizer_curriculum,
                                            num_epochs=num_epochs, verbose=False,
                                            switch_loader=loader_combined,
                                            val_loader=val_loader,
                                            model_type='curriculum',
                                            switch_epoch=switch_epoch,
                                            gradual_switch=gradual_switch,
                                            plot_title=f'{plot_title} (switch epoch={switch_epoch})')

        if save_model:
            torch.save(model_curriculum.state_dict(
            ), f'models/curriculum/model_curriculum_{switch_loader_name}_{gs}_{switch_epoch}.pth')

        currciulum_accs.append(acc)

        print(
            f"Validation accuracy of model at switch epoch {switch_epoch}: {acc}")

    return currciulum_accs


def plot_curriculum_experiment(curriculum_accs, title=None):

    x_indices = list(range(8))
    switch_epochs = [0] + [2**i for i in range(1, 8)]
    plt.plot(x_indices, curriculum_accs, marker='o', linestyle='None')
    plt.xlabel('Epoch Switch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(x_indices, switch_epochs)
    plt.tight_layout()
    plt.show()
