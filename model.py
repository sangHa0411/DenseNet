import torch
import torch.nn as nn

class ConvNBlock(nn.Module) :
    def __init__(self, in_channel, out_channel,  kernal_size) :
        super(ConvNBlock , self).__init__()
        pad_size = int(kernal_size/2)
        self.c_block = nn.Sequential(nn.BatchNorm2d(in_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channel, out_channel, kernal_size, padding=pad_size, stride=1))
        
    def forward(self , in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor


class Conv1Block(nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(Conv1Block , self).__init__()
        self.c_block = nn.Sequential(nn.BatchNorm2d(in_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channel, out_channel, 1))
        
    def forward(self , in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor

class ConvBlock(nn.Module) :
    def __init__(self, in_channel, inter_channel, out_channel, kernal_size) :
        super(ConvBlock , self).__init__()
        pad_size = int(kernal_size/2)
        self.c_block = nn.Sequential(Conv1Block(in_channel, inter_channel),
                                     ConvNBlock(inter_channel, out_channel, kernal_size))
        
    def forward(self , in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor

# DenseBlock
class DenseBlock(nn.Module) :
    def __init__(self, layer_size, in_channel, inter_channel, out_channel, kernal_size) :
        super(DenseBlock , self).__init__()
        self.layer_size = layer_size
        self.conv_list = nn.ModuleList()
        
        size_ptr = in_channel
        for i in range(layer_size) :
            self.conv_list.append(ConvBlock(size_ptr, inter_channel, out_channel, kernal_size))
            size_ptr += out_channel
        
    def forward(self, in_tensor) :
        tensor_ptr = in_tensor
        tensor_list = []
        for i in range(self.layer_size) :
            tensor_list.append(tensor_ptr)
            if i > 0 :
                tensor_ptr = torch.cat(tensor_list , dim = 1)
            tensor_ptr = self.conv_list[i](tensor_ptr)
            
        return tensor_ptr

# Transition Layer
class Transition(nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(Transition , self).__init__()
        self.tran = nn.Sequential(nn.BatchNorm2d(in_channel),
                                  nn.Conv2d(in_channel, out_channel , 1),
                                  nn.AvgPool2d(2, stride=2))

    def forward(self, in_tensor) :
        o_tensor = self.tran(in_tensor)
        return o_tensor

# DenseNet
class DenseNet(nn.Module) :
    def __init__(self, input_size, layer_list, growth_rate, in_kernal, kernal_size, class_size) :
        super(DenseNet , self).__init__()
        self.d_net = nn.ModuleList()
        self.growth_rate = growth_rate
        self.d_net.append(nn.Conv2d(3, (growth_rate*2), in_kernal, stride=2, padding=int(in_kernal/2))) 
        self.d_net.append(nn.MaxPool2d(2, stride=2))
        
        size_ptr = input_size/4
        for i in range(len(layer_list)) :
            cur_comp = self.get_compression(layer_list[i-1], growth_rate) if i > 0 else (growth_rate*2)
            next_comp = self.get_compression(layer_list[i], growth_rate)
            self.d_net.append(DenseBlock(layer_list[i], cur_comp, next_comp, growth_rate, kernal_size))
            if i < len(layer_list) - 1 :
                self.d_net.append(Transition(growth_rate, next_comp))
                size_ptr /= 2
        
        self.avg_pool = nn.AvgPool2d(int(size_ptr))
        self.o_layer = nn.Linear(growth_rate, class_size)
        self.layer_size = len(self.d_net)

        self.init_param()

    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)
                
    def get_compression(self, layer_size, channel_size) :
        return int(layer_size*channel_size/2)
        
    def forward(self , in_tensor) :
        batch_size = in_tensor.shape[0]
        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.d_net[i](tensor_ptr)

        avg_tensor = self.avg_pool(tensor_ptr)
        avg_tensor = torch.reshape(avg_tensor , (batch_size , self.growth_rate))
        o_tensor = self.o_layer(avg_tensor)
        return o_tensor
