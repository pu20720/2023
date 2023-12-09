import torch
from torch import nn


class VGG16_Model(nn.Module): # CNN_Model繼承nn.Module
    def __init__(self, NumClass):
        super(VGG16_Model , self).__init__() # 等價nn.Module.__init__(self)
        self.vgg16 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*64*224*224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size  = 3, stride = 1, padding = 1, bias = False), # batch_size*64*224*224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*64*112*112

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*128*112*112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*128*112*112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*128*56*56

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*256*28*28

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*28*28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*28*28 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*28*28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2 , stride = 2), # batch_size*512*14*14

            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # batch_size*512*7*7
        )
        self.NumClass = NumClass
        self.globe_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.globe_fcl = nn.Linear(512 * 1 * 1, self.NumClass)
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.vgg16(x)
        x = self.globe_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.globe_fcl(x)
        return x
    def loss(self, LossFunction, Lr):
        if LossFunction == 'A':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = Lr)
        elif LossFunction == 'S':
            self.optimizer = torch.optim.SGD(self.parameters(), lr = Lr)
    def LossCount(self, pred, lable):
        Loss = self.criterion(pred, lable)
        self.optimizer.zero_grad(set_to_none = True) # 權重梯度歸零
        Loss.backward() # 計算每個權重的loss梯度
        self.optimizer.step() # 權重更新
        return Loss
    def scheduler(self, num):
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, num, 0.1) # 通常搭配Adms使用，在特定epochs更新LR
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20) # 餘旋波，此方法所出來的結果通常較多跳動，建議epochs要剛好可以整除
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.5, patience = 15, threshold = 0.0001) # scheduler.step()要放東西給他做依據通常是驗證正確率，本方法為自適應更改不需要使用者寫這行幫助更新LR
        '''
        # 此方法是依照lambda的規則去更新
        lr_lambda = lambda epochs: 1 / (epochs + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch = -1)
        '''
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.1, last_epoch = -1) # 以指數圖形為基礎來做衰減
        return scheduler
