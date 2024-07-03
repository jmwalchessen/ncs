import torch
import matplotlib.pyplot as plt

def learning_rate_schedule(initial_learning_rate, epoch):

  return initial_learning_rate*(.99**(epoch))

def step_fn(state, batch, device):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
    state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
    batch: A mini-batch of training/evaluation data.

    Returns:
    loss: The average loss value of this state.
    """
    classifier = state['classifier']
    loss_function = state['loss_function']
    optimizer = state['optimizer']
    optimizer.zero_grad()
    batch_images = (batch[0]).to(device)
    batch_classes = (batch[1]).to(device)
    predicted_classes = classifier(batch_images)
    loss = loss_function(predicted_classes, batch_classes)
    loss.backward()
    optimizer.step()
    state['step'] += 1
    return loss

def evaluate_loss(state, batch, device):

    classifier = state['classifier']
    loss_function = state['loss_function']
    batch_images = (batch[0]).to(device)
    batch_classes = (batch[1]).to(device)
    predicted_classes = classifier(batch_images)
    loss = loss_function(predicted_classes, batch_classes)
    return loss

def visualize_loss(num_epochs, train_losses, evaluation_losses, fig_name):

    plt.plot([i for i in range(0, num_epochs)], evaluation_losses)
    plt.plot([i for i in range(0, num_epochs)], train_losses)
    plt.savefig(fig_name)