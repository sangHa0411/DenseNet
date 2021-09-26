import torch
import torch.nn as nn

# Convolutional Layer
class ConvNBlock(nn.Module) :
    def __init__(self, in_channel, out_channel,  kernal_size) :
        super(ConvNBlock , self).__init__()
        pad_size = int(kernal_size/2)
        self.c_block = nn.Sequential(nn.BatchNorm2d(in_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channel, out_channel, kernal_size, padding=pad_size))
        
    def forward(self, in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor

# Bottlenect Layer
class Conv1Block(nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(Conv1Block , self).__init__()
        self.c_block = nn.Sequential(nn.BatchNorm2d(in_channel),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channel, out_channel, 1))
        
    def forward(self, in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor

# Conv Block
class ConvBlock(nn.Module) :
    def __init__(self, in_channel, inter_channel, out_channel, kernal_size) :
        super(ConvBlock , self).__init__()
        pad_size = int(kernal_size/2)
        self.c_block = nn.Sequential(Conv1Block(in_channel, inter_channel),
                                     ConvNBlock(inter_channel, out_channel, kernal_size))
        
    def forward(self, in_tensor) :
        o_tensor = self.c_block(in_tensor)
        return o_tensor

# Transition Layer 
class Transition(nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(Transition , self).__init__()
        self.tran = nn.Sequential(nn.BatchNorm2d(in_channel),
                                  nn.Conv2d(in_channel, out_channel , 1),
                                  nn.AvgPool2d(2))

    def forward(self, in_tensor) :
        o_tensor = self.tran(in_tensor)
        return o_tensor

# Dense Block
class DenseBlock(nn.Module) :
    def __init__(self, layer_size, in_channel, growth_rate, kernal_size, tran_flag) :
        super(DenseBlock , self).__init__()
        self.layers = layer_size
        self.in_ch = in_channel
        self.growth_rate = growth_rate
        self.k_size = kernal_size
        self.tran_flag = tran_flag

        self.block = nn.ModuleList()
        
        size_ptr = in_channel
        for i in range(layer_size) :
            self.block.append(ConvBlock(size_ptr, int(size_ptr*0.5), growth_rate, kernal_size))
            size_ptr += growth_rate

        if self.tran_flag == True :
            self.block.append(Transition(size_ptr, int(size_ptr*0.5)))
            self.layers += 1
        
    def forward(self, in_tensor) :
        tensor_ptr = in_tensor
        tensor_list = []

        for i in range(self.layers) :
            tensor_list.append(tensor_ptr)
            if i > 0 :
                tensor_ptr = torch.cat(tensor_list , dim = 1)
            tensor_ptr = self.block[i](tensor_ptr)

        return tensor_ptr

# DenseNet
class DenseNet(nn.Module) :
    def __init__(self, input_size, layer_list, growth_rate, in_kernal, kernal_size, class_size) :
        super(DenseNet , self).__init__()
        self.input_size = input_size
        self.layer_list = layer_list
        self.growth_rate = growth_rate
        self.in_kernal = in_kernal
        self.kernal_size = kernal_size
        self.class_size = class_size

        self.d_net = nn.ModuleList()

        self.d_net.append(nn.Conv2d(3, (growth_rate*2), in_kernal, stride=2, padding=int(in_kernal/2))) 
        self.d_net.append(nn.MaxPool2d(2, stride=2))
        
        ch_ptr = growth_rate*2
        size_ptr = input_size/4
        for i in range(len(layer_list)) :
            if i < len(layer_list)-1 :
                tran_flag = True
                size_ptr /= 2
            else :
                tran_flag = False

            block = DenseBlock(layer_list[i], int(ch_ptr), growth_rate, kernal_size, tran_flag)
            self.d_net.append(block)

            ch_ptr = self.get_size(ch_ptr, layer_list[i]) / 2

        self.layer_size = len(layer_list)+2
        self.avg_pool = nn.AvgPool2d(int(size_ptr))
        self.o_layer = nn.Linear(growth_rate, class_size)

        self.init_param()

    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.kaiming_normal_(p)

    def get_size(self, in_ch, layer_size) :
        return in_ch + (self.growth_rate * (layer_size))

    def forward(self , in_tensor) :
        batch_size = in_tensor.shape[0]

        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.d_net[i](tensor_ptr)

        avg_tensor = self.avg_pool(tensor_ptr)
        avg_tensor = torch.reshape(avg_tensor , (batch_size, self.growth_rate))
        o_tensor = self.o_layer(avg_tensor)
        return o_tensor

