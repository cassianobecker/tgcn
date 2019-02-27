import torch
import numpy as np

def gradient_ascent(model, init=None, input_shape=(1, 1, 28, 28), target=0, ep=.01, max_iter=10000, device='cuda'):
    """
    Performs gradient ascent on a model with given class probabilities and target classification
    If S_c(x) is the class scores for class c, then S_c(x) = model.class_scores(x)[0][target]
    uses l2 regularization with lambda=.1
    """
    if init is None:
        x = torch.randn(input_shape, requires_grad=True, device=device)
    else:
        x = init.clone().detach().requires_grad_(True)
    for i in range(max_iter):
        # Class probability for target given input x
        output = model.class_probabilities(x)[0][target]
        # regularization term
        output -= .01 * torch.norm(x)
        # calculate gradient
        output.backward()
        # take step in direction of gradient
        step = ep * x.grad
        next_img = x + step
        x = next_img.clone().detach().requires_grad_(True)
        i += 1
        if i % 20000 == 0:
            print(f'Target: {target}\t{i}/{max_iter} iterations updated')
    print(f'Target: {target}\tReached max_iter')
    return x