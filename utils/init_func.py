import torch.nn as nn


__all__ = [
    'Init_Func',
]


# a colleciton of init functions for conv layers
class  Init_Func():
    def __init__(self,init_type):
        self.init_type = init_type

    def init(self,W):
        if self.init_type  == 'default':
            return nn.init.xavier_uniform_(W)
        elif self.init_type == 'orthogonal':
            return nn.init.orthogonal_(W)
        elif self.init_type == 'uniform':
            return nn.init.uniform_(W)
        elif self.init_type == 'normal':
            return nn.init.normal_(W)
        elif self.init_type == 'constant':
            return nn.init.constant_(W)
        elif self.init_type == 'xavier_normal':
            return nn.init.xavier_normal_(W)
        elif self.init_type == 'xavier_uniform':
            return nn.init.xavier_uniform_(W)
        elif self.init_type == 'kaiming_uniform':
            return nn.init.kaiming_uniform_(W)
        elif self.init_type == 'kaiming_normal':
            return nn.init.kaiming_normal_(W)
        else:
            raise Exception ("unknown initialization method")
