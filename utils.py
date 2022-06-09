

import torch

def shake_shake(device):
    ''' Shake-shake regularization'''
    
    alpha = torch.rand(1).to(device)
    alpha.requires_grad = False
    beta = torch.rand(1).to(device)
    beta.requires_grad = False
    gamma = torch.rand(1).to(device)
    gamma.requires_grad = False
    
    return alpha, beta, gamma, alpha + beta+ gamma