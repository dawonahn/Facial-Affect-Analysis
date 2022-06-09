import torch
import audtorch
from tqdm import tqdm


def RMSE(gt, pt):
    '''Calculate the RMSE.'''
    mse = torch.mean((gt-pt)**2)
    rmse = torch.sqrt(mse)
    return rmse
    
    
def PCC(gt, pt):
    '''Calculate the Pearson Correlation Coefficient (PCC) '''
#     return torch.corrcoef(gt, pt)[0, 1]
    x = audtorch.metrics.functional.pearsonr(gt, pt)
    return torch.nan_to_num(x)


def CCC(gt, pt):
    '''Calculate the Concordance Correlation Coefficient (CCC) '''
    mu_gt = torch.mean(gt)
    mu_pt = torch.mean(pt)
    sigma_gt = torch.std(gt)
    sigma_pt = torch.std(pt)
    
    pcc = PCC(gt, pt)
    pcc = torch.nan_to_num(pcc)
    num = 2 * sigma_gt * sigma_pt * pcc
    denum = sigma_gt**2 + sigma_pt **2 + (mu_gt - mu_pt) ** 2
    
    return num/denum


def evaluation(dataloader, model, DEVICE):
    ''' Evaluate the model with validation and test data'''
    result = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs.to(DEVICE))
            result.append(torch.hstack([outputs, labels.to(DEVICE)]))
       
        result = torch.vstack(result)
        arousal_rmse = RMSE(result[2], result[0])
        valence_rmse = RMSE(result[3], result[1])
    
    return result, arousal_rmse.item(), valence_rmse.item()

def evaluation(dataloader, model, DEVICE):
    ''' Evaluate the model with validation and test data (for my implementation'''
    result = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs.to(DEVICE)).reshape(-1, 2)
            result.append(torch.hstack([outputs, labels.to(DEVICE)]))
       
        result = torch.vstack(result)
        arousal_rmse = RMSE(result[2], result[0])
        valence_rmse = RMSE(result[3], result[1])
    
    return result, arousal_rmse.item(), valence_rmse.item()
