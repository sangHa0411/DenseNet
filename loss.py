import torch
import torch.nn.functional as F

def acc_fn(y_output , y_label) :
    y_arg = torch.argmax(y_output , dim = -1)
    y_acc = (y_arg == y_label).float()    
    y_acc = torch.mean(y_acc)
    return y_acc

def loss_fn(y_output, y_label) :
    y_log = -F.log_softmax(y_output, -1)
    y_loss = torch.mul(y_log, y_label)
    y_sum = torch.sum(y_loss, dim=1)
    y_mean = torch.mean(y_sum)
    return y_mean
