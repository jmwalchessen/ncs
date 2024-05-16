import torch
from torch import *
from losses import *



def train_and_evaluation_step_per_epoch(epoch, state, classifier, beta1, beta2,
                                        initial_learning_rate, train_dataloader,
                                        eval_dataloader, eval_train_dataloader,
                                        epsilon, weight_decay, eval_losses,
                                        eval_train_losses):

    train_iterator = iter(train_dataloader)
    while True:
        try:
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
            batch = next(train_iterator)
            # Execute one training step
            loss = step_fn(state, batch, device)
        except StopIteration:
            break

    eval_iterator = iter(eval_dataloader)
    eval_train_iterator = iter(eval_train_dataloader)
    optimizer = optim.Adam(classifier.parameters(),
                           lr=learning_rate_schedule(initial_learning_rate, epoch),
                           betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    while True:
        try:
            eval_batch = next(eval_iterator)
            eval_loss = evaluate_loss(state, eval_batch, device)
            eval_losses.append(float(eval_loss.detach().cpu().numpy()))
            eval_train_batch = next(eval_train_iterator)
            eval_train_loss = evaluate_loss(state, eval_train_batch, device)
            eval_train_losses.append(float(eval_train_loss.detach().cpu().numpy()))
            print(eval_train_loss)
            print(eval_loss)
        except StopIteration:
            break
    
    #not sure why these steps are necessary, will look at later
    lr = learning_rate_schedule(initial_learning_rate, epoch)
    state['optimizer'] = optimizer = optim.Adam((state['classifier']).parameters(), lr=lr,
                                                 betas=(beta1, beta2), eps=epsilon,
                                                 weight_decay=weight_decay)