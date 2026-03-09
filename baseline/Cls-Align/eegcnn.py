             
  
                                                                         
  
                                                                              
                                                                               
                                                                              
                                                                           
                                                                       
                                                          
  
                                                                                
                                                 
  
                                                                            
                                                                          
                                                                             
                                                                        
                                                                               
                                                                               
           

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
import math

class EEGcnn(nn.Module):
    def __init__(self, Chans=64, dropoutRate=0.5, kernLength1=100, kernLength2=50, F1=8, D=2, F2=16, P1=2, P2=5, dropoutType='Dropout'):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding='same', bias=False),                          
            nn.BatchNorm2d(F1),                          
            nn.Conv2d(F1, D*F1, (Chans, 1), groups=F1, bias=False),                        
            nn.BatchNorm2d(D*F1),                        
            nn.ELU(),                        
            nn.AvgPool2d((1, P1)),                           
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate)                           
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, kernLength2), groups=D*F1, padding='same', bias=False),                           
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),                         
            nn.BatchNorm2d(F2),                         
            nn.ELU(),                         
            nn.AvgPool2d((1, P2)),                         
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate)                         
        )

    def forward(self, input):                      
        input = torch.unsqueeze(input, dim=1)                         
        input = self.block1(input)
        input = self.block2(input)
        input = torch.squeeze(input, dim=2)                      
        return input

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)