import torch
import torch.nn as nn

'''
This implimentation is tested for multi label network
'''

class Autopool(nn.Module):


    '''
    input_size = no of output labels
    '''
    def __init__(self, input_size, device ):
    
        super(Autopool, self).__init__()
        self.alpha = nn.Parameter(requires_grad= True)
        self.alpha.data = torch.ones([input_size], dtype=torch.float64, requires_grad= True, device=device)
        self.sigmoid_layer = nn.Sigmoid()
        self.softmax_layer = nn.Softmax(dim=2)
        
        
    def forward(self,x):
        sigmoid_output = self.sigmoid_layer(x)
        alpa_mult_out = torch.mul(sigmoid_output, self.alpha)
        weights = self.softmax_layer(alpa_mult_out)
        weighted_output = torch.mul(sigmoid_output, weights)
        final_out = torch.sum(weighted_output, dim=1)
        return final_out
