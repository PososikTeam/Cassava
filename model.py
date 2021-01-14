from typing import Callable, List, Tuple 

import os
import torch
import catalyst

from catalyst.dl import utils
import senet


SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)


from torch import nn
import pretrainedmodels

num_classes = 5

from efficientnet_pytorch import EfficientNet


"""
EfficientNetB0 - (224, 224, 3)
EfficientNetB1 - (240, 240, 3)
EfficientNetB2 - (260, 260, 3)
EfficientNetB3 - (300, 300, 3)
EfficientNetB4 - (380, 380, 3)
EfficientNetB5 - (456, 456, 3)
EfficientNetB6 - (528, 528, 3)
EfficientNetB7 - (600, 600, 3)
"""

class EfnB2(nn.Module):
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(EfnB2, self).__init__()
        self.efn = EfficientNet.from_name('efficientnet-b2', include_top = False)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(1408, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
    
    def forward(self, x):
        x = self.efn(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x

class EfnB3(nn.Module):
    
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(EfnB3, self).__init__()
        self.efn = EfficientNet.from_name('efficientnet-b3', include_top = False)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(1536, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
    
    def forward(self, x):
        x = self.efn(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x

class EfnB4(nn.Module):
    
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(EfnB4, self).__init__()
        self.efn = EfficientNet.from_name('efficientnet-b4', include_top = False)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(1792, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
    
    def forward(self, x):
        x = self.efn(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x


class EfnB5(nn.Module):
    
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(EfnB5, self).__init__()
        self.efn = EfficientNet.from_name('efficientnet-b5', include_top = False)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(2048, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
    
    def forward(self, x):
        x = self.efn(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x

class EfnB6(nn.Module):
    
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(EfnB6, self).__init__()
        self.efn = EfficientNet.from_name('efficientnet-b6', include_top = False)
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(2304, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
    
    def forward(self, x):
        x = self.efn(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x

class SEResNext50(nn.Module):
    def __init__(self, middle_layer = 1024, output_layer = 5):
        super(SEResNext50, self).__init__()
        self.senet = senet.se_resnext50_32x4d()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(2048, middle_layer)
        self.fc2 = nn.Linear(middle_layer, output_layer)
        

    def forward(self, x):
        x = self.senet(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x
    

def get_model(model_name, middle_layer = 512, output_layer = 5):
    d = {
        'efficientnet-b2': EfnB2(middle_layer, output_layer),
        'efficientnet-b3': EfnB3(middle_layer, output_layer),
        'efficientnet-b4': EfnB4(middle_layer, output_layer),
        'efficientnet-b5': EfnB5(middle_layer, output_layer),
        'efficientnet-b6': EfnB6(middle_layer, output_layer),
        'SEResNext50': SEResNext50(middle_layer, output_layer)
    }
    return d[model_name]