from torch import nn.optim 
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = "params/unet.pth"