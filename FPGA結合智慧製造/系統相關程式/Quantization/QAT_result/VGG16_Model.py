# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class VGG16_Model(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(VGG16_Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #VGG16_Model::input_0
        self.module_1 = py_nndct.nn.quant_input() #VGG16_Model::VGG16_Model/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[0]/input.3
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[2]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[3]/input.9
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[5]/1624
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #VGG16_Model::VGG16_Model/Sequential[vgg16]/MaxPool2d[6]/input.13
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[7]/input.15
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[9]/input.19
        self.module_9 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[10]/input.21
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[12]/1690
        self.module_11 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #VGG16_Model::VGG16_Model/Sequential[vgg16]/MaxPool2d[13]/input.25
        self.module_12 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[14]/input.27
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[16]/input.31
        self.module_14 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[17]/input.33
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[19]/input.37
        self.module_16 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[20]/input.39
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[22]/1782
        self.module_18 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #VGG16_Model::VGG16_Model/Sequential[vgg16]/MaxPool2d[23]/input.43
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[24]/input.45
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[26]/input.49
        self.module_21 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[27]/input.51
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[29]/input.55
        self.module_23 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[30]/input.57
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[32]/1874
        self.module_25 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #VGG16_Model::VGG16_Model/Sequential[vgg16]/MaxPool2d[33]/input.61
        self.module_26 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[34]/input.63
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[36]/input.67
        self.module_28 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[37]/input.69
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[39]/input.73
        self.module_30 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/Conv2d[40]/input.75
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #VGG16_Model::VGG16_Model/Sequential[vgg16]/ReLU[42]/1966
        self.module_32 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #VGG16_Model::VGG16_Model/Sequential[vgg16]/MaxPool2d[43]/input
        self.module_33 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #VGG16_Model::VGG16_Model/AdaptiveAvgPool2d[globe_avg_pool]/1996
        self.module_34 = py_nndct.nn.Module('nndct_shape') #VGG16_Model::VGG16_Model/1998
        self.module_35 = py_nndct.nn.Module('nndct_reshape') #VGG16_Model::VGG16_Model/2003
        self.module_36 = py_nndct.nn.Linear(in_features=512, out_features=3, bias=True) #VGG16_Model::VGG16_Model/Linear[globe_fcl]/inputs
        self.module_37 = py_nndct.nn.dequant_output() #VGG16_Model::VGG16_Model/DeQuantStub[dequant_stub]/2005

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(output_module_0)
        output_module_0 = self.module_17(output_module_0)
        output_module_0 = self.module_18(output_module_0)
        output_module_0 = self.module_19(output_module_0)
        output_module_0 = self.module_20(output_module_0)
        output_module_0 = self.module_21(output_module_0)
        output_module_0 = self.module_22(output_module_0)
        output_module_0 = self.module_23(output_module_0)
        output_module_0 = self.module_24(output_module_0)
        output_module_0 = self.module_25(output_module_0)
        output_module_0 = self.module_26(output_module_0)
        output_module_0 = self.module_27(output_module_0)
        output_module_0 = self.module_28(output_module_0)
        output_module_0 = self.module_29(output_module_0)
        output_module_0 = self.module_30(output_module_0)
        output_module_0 = self.module_31(output_module_0)
        output_module_0 = self.module_32(output_module_0)
        output_module_0 = self.module_33(output_module_0)
        output_module_34 = self.module_34(input=output_module_0, dim=0)
        output_module_35 = self.module_35(input=output_module_0, shape=[output_module_34,-1])
        output_module_35 = self.module_36(output_module_35)
        output_module_35 = self.module_37(input=output_module_35)
        return output_module_35
