from PIL import Image
import torch
import numpy
from torchvision import transforms,datasets
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
import os
import argparse
import glob
from torch.backends import cudnn
from pytorch_nndct import nn as nndct_nn
from pytorch_nndct import QatProcessor
import torchvision.models as models

class Dataset(data.Dataset):
    def __init__(self, data_root, transform, Lable_Dict):
        self.transform = transform
        self.images = []
        image_path = []
        image_data_path = glob.glob(os.path.join(data_root, '*'))
        for path in image_data_path:
            image_path.extend(glob.glob(os.path.join(path, '*.png')))
        for path in image_path:
            im = Image.open(path).convert('RGB').copy()
            im = self.transform(im)
            self.images.append(im)
        self.Lables = [os.path.dirname(path) for path in image_path]
        for lab in range(len(self.Lables)):
            self.Lables[lab] = self.Lables[lab].split('/')[-1]
            self.Lables[lab] = Lable_Dict[self.Lables[lab]]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return self.images[index], self.Lables[index]
    def collate_fn(batch):
        img = list()
        cls = list()
        for data in batch:
            img.append(data[0])
            cls.append(data[1])
        cls = torch.as_tensor(cls)
        img = torch.stack(img, dim = 0)
        return img, cls

class VGG16_Model_BN(nn.Module): # CNN_Model繼承nn.Module
    def __init__(self):
        super(VGG16_Model_BN , self).__init__() # 等價nn.Module.__init__(self)
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
        self.globe_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.globe_fcl = self.fcl1=nn.Linear(512 * 1 * 1, args.out)
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()
    def forward(self, x):
        x = self.quant_stub(x)
        x = self.vgg16(x)
        x = self.globe_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.globe_fcl(x)
        x = self.dequant_stub(x)
        return x
    def loss(self, LossFunction, params_group):
        if LossFunction == 'A':
            self.optimizer = torch.optim.Adam(params_group)
        elif LossFunction == 'S':
            self.optimizer = torch.optim.SGD(params_group)
    def LossCount(self, pred, lable):
        Loss = self.criterion(pred, lable)
        self.optimizer.zero_grad(set_to_none = True) # 權重梯度歸零
        Loss.backward() # 計算每個權重的loss梯度
        self.optimizer.step() # 權重更新
        return Loss
    def scheduler(self, num):
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, num, 0.1) # 通常搭配Adms使用，在特定epochs更新LR
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20) # 餘旋波，此方法所出來的結果通常較多跳動，建議epochs要剛好可以整除
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor = 0.5, patience = 15, threshold = 0.00001) # scheduler.step()要放東西給他做依據通常是驗證正確率，本方法為自適應更改不需要使用者寫這行幫助更新LR
        # 此方法是依照lambda的規則去更新
        #lr_lambda = lambda epochs: 1 / (epochs + 1)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch = -1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.1, last_epoch = -1) # 以指數圖形為基礎來做衰減
        return scheduler

def Data(path, batch_size, train_num, num_workers):
    Transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    Lable_Dict = {'bridge' : 0, 'normal' : 1, 'sn_less' : 2}
    dataset = Dataset(path, transform = Transforms, Lable_Dict = Lable_Dict)
    train_num = round(dataset.__len__() * train_num)
    train_data, valid_data = data.random_split(dataset,[train_num, dataset.__len__() - train_num])
    train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
    valid_loader = data.DataLoader(valid_data, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
    subset_loader = data.DataLoader(valid_data, batch_size = 1, shuffle = False, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
    return  train_loader, valid_loader, subset_loader

def Data2(path, batch_size, num_workers, trained):
    Lable_Dict = {'bridge' : 0, 'normal' : 1, 'sn_less' : 2}
    if trained:
        Transforms2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), transforms.RandomVerticalFlip(1))])
        Transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = Dataset(path, transform = Transforms, Lable_Dict = Lable_Dict)
        dataset += Dataset(path, transform = Transforms2, Lable_Dict = Lable_Dict)
        dataloader = data.DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
        return dataloader
    else:
        Transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = Dataset(path, transform = Transforms, Lable_Dict = Lable_Dict)
        dataloader = data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
        return dataloader
    
def train(epochs, CNN, train_loader, device, scheduler, valid, valid_loader = None, module_name = 'Module'):
    if not valid:
        train_acc_his, train_losses_his = numpy.arange(epochs, dtype = float), numpy.arange(epochs, dtype = float)
        for i in range(0, epochs):
            print('running epoch:' + str(i + 1))
            train_correct, train_loss = 0, 0
            CNN.train()
            for _data in train_loader: # 一個batch的image、label。image：(batch_size*3*3*224)。label：(1*batch_size)
                image, label = _data
                image, label = image.to(device), label.to(device)
                pred = CNN(image) # pred：(batch_size*class) gpu
                loss = CNN.LossCount(pred, label)
                output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出) gpu
                train_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數 cpu
                train_loss += loss.item() * image.size(0) # 累加計算每一epoch的loss總和。loss.item()：(1*1)，一個batch的平均loss。image.size(0)：一個batch的訓練資料總數 cpu
            scheduler.step()
            train_acc = train_correct / len(train_loader.dataset) * 100 # 計算每一個epoch的平均訓練正確率(%)
            train_loss = train_loss / len(train_loader.dataset) # 計算每一個epoch的平均訓練loss
            print('train loss:', train_loss)
            print('train acc:', train_acc)
            train_acc_his[i] = train_acc # 累積紀錄每一個epoch的平均訓練正確率(%) (1*epochs)
            train_losses_his[i] = train_loss # 累積記錄每一個epoch的平均訓練loss (1*epochs)
            valid_acc_his = numpy.arange(epochs, dtype = float)
            valid_loss_his = numpy.arange(epochs, dtype = float)
    else:
        train_acc_his, train_losses_his, valid_acc_his, valid_loss_his = numpy.arange(epochs, dtype = float), numpy.arange(epochs, dtype = float), numpy.arange(epochs, dtype = float), numpy.arange(epochs, dtype = float)
        for i in range(0, epochs):
            print('running epoch:' + str(i + 1))
            train_correct, train_loss, valid_correct, valid_loss = 0, 0, 0, 0
            CNN.train()
            for _data in train_loader: # 一個batch的image、label。image：(batch_size*3*3*224)。label：(1*batch_size)
                image, label = _data
                image, label = image.to(device), label.to(device)
                pred = CNN(image) # pred：(batch_size*class) gpu
                loss = CNN.LossCount(pred, label)
                output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出) gpu
                train_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數 cpu
                train_loss += loss.item() * image.size(0) # 累加計算每一epoch的loss總和。loss.item()：(1*1)，一個batch的平均loss。image.size(0)：一個batch的訓練資料總數 cpu
            if i != 0:
                scheduler.step(valid_acc_his[i - 1])
            train_acc = train_correct / len(train_loader.dataset) * 100 # 計算每一個epoch的平均訓練正確率(%)
            train_loss = train_loss / len(train_loader.dataset) # 計算每一個epoch的平均訓練loss
            with torch.no_grad():
                CNN.eval()
                for _data in valid_loader: # 一個batch的image、label。image：(batch_size*3*3*224)。label：(1*batch_size)
                    image, label = _data
                    image, label = image.to(device), label.to(device)
                    pred = CNN(image) # pred：(batch_size*class)
                    loss = CNN.criterion(pred, label)
                    output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出)
                    valid_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數
                    valid_loss += loss.item() * image.size(0) # 累加計算每一epoch的loss總和。loss.item()：(1*1)，一個batch的平均loss。image.size(0)：一個batch的驗證資料總數
            valid_acc = valid_correct / len(valid_loader.dataset) * 100 # 計算每一個epoch的平均驗證正確率(%)
            valid_loss = valid_loss /  len(valid_loader.dataset)
            train_acc_his[i] = train_acc # 累積紀錄每一個epoch的平均訓練正確率(%) (1*epochs)
            train_losses_his[i] = train_loss # 累積記錄每一個epoch的平均訓練loss (1*epochs) 
            valid_acc_his[i] = valid_acc # 累積紀錄每一個epoch的平均驗證正確率(%) (1*epochs)
            valid_loss_his[i] = valid_loss
            print('train acc :', train_acc)
            print('train loss:', train_loss)
            print('valid acc :', valid_acc)
            print('valid loss:', valid_loss , '\n')
    torch.save(CNN.state_dict(), module_name)
    return train_acc_his, train_losses_his, valid_acc_his, valid_loss_his

def validate(test_loader, model, device):
  # switch to evaluate mode
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)
  with torch.no_grad():
    model.eval()
    valid_correct = 0
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        pred = model(image)
        output_id = torch.max(pred, dim=1)[1]
        valid_correct += torch.sum(label == output_id).item()
    valid_acc = valid_correct / len(test_loader.dataset) * 100 
    print(valid_acc)

def main(args):
    cudnn.benchmark = True
    Test_Loader = Data2(args.TestDataPath, 1, args.NumWorkers, False)
    Train_Loader = Data2(args.DataPath, args.batch_size, args.NumWorkers, True) 
    Valid_Loader = Data2(args.ValidPath, args.batch_size, args.NumWorkers, False)
    #Train_Loader, Valid_Loader, Subset_Loader = Data(args.DataPath, args.batch_size, args.Cut_Number, args.NumWorkers)
    Subset_Loader = Data2(args.ValidPath, 1, args.NumWorkers, False)
    
    CNN = VGG16_Model_BN()
    CNN.load_state_dict(torch.load(r'/workspace/Project/Model/SN_Finished_Training_Model'))
    CNN = CNN.to(args.device)
    
    inputs = torch.randn([args.batch_size, 3, 224, 224], dtype=torch.float32).to(args.device)
    qat_processor = QatProcessor(CNN, inputs, bitwidth=8, device=torch.device('cpu'))
    quantized_model = qat_processor.trainable_model()
    quantized_model.to(args.device)

    param_groups = [{
    'params': quantized_model.quantizer_parameters(),
    'lr': args.quantizer_lr,
    'name': 'quantizer_lr'
    }, {
    'params': quantized_model.non_quantizer_parameters(),
    'lr': args.learning_rate,
    'name': 'weight'
    }]

    quantized_model.loss('S', param_groups)
    scheduler = quantized_model.scheduler(args.scheduler)
    _, _, _, _ = train(args.epochs, quantized_model, Train_Loader, args.device, scheduler, True, Valid_Loader, args.ModuleName)
    quantized_model.load_state_dict(torch.load(args.ModuleName,'cuda'))
    deployable_model = qat_processor.to_deployable(quantized_model,args.QATDataPath)
    
    validate(Test_Loader, deployable_model, args.device)
    
    deployable_model = qat_processor.deployable_model(args.QATDataPath, used_for_xmodel=True)
    for images, _ in Subset_Loader:
       deployable_model(images)
    qat_processor.export_xmodel(args.QATDataPath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, dest = 'epochs', help = 'the number of iteration', default = 40)
    parser.add_argument('--DataPath', type = str, dest = 'DataPath', help = 'the path of train data', default = '/workspace/Project/Train')
    parser.add_argument('--ValidPath', type = str, dest = 'ValidPath', help = 'the path of valid data', default = '/workspace/Project/Valid/')
    parser.add_argument('--TestDataPath', type = str, dest = 'TestDataPath', help = 'the path of test_data', default = '/workspace/Project/Test/')
    parser.add_argument('--QATDataPath', type = str, dest = 'QATDataPath', help = 'the path of QAT result', default = '/workspace/Project/Quantization/QAT_result')
    parser.add_argument('--batch_size', type = int, dest = 'batch_size', help = 'the number of batch', default = 32)
    parser.add_argument('--out', type = int, dest = 'out', help = 'the number of class', default = 3)
    parser.add_argument('--NumWorkers', type = int, dest = 'NumWorkers', help = 'the number of CPU workers', default = 4)
    parser.add_argument('--learning_rate', type = float, dest = 'learning_rate', default = 7.5e-3)
    parser.add_argument('--quantizer_lr', type = float, dest = 'quantizer_lr', default = 1e-2)
    parser.add_argument('--scheduler', type = int, dest = 'scheduler', default = 80)
    parser.add_argument('--device', type = str, dest = 'device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-Cut_Number-', type = float, dest = 'Cut_Number', default = 0.95)
    parser.add_argument('--ModuleName', type = str, dest = 'ModuleName', default = '/workspace/Project/Quantization/QAT_result/SN_Quantization')
    args = parser.parse_args()
    main(args)
