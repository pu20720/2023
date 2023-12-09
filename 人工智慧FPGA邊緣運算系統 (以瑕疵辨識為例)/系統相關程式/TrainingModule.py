import torch
import numpy
import matplotlib.pyplot as plt
import argparse
from torch.backends import cudnn
import ModelConstruction.VGG16 as Mod
import Data
import time

def train(epochs, CNN, train_loader, device, scheduler, valid, valid_loader = None, module_name = 'Module'):
    ti = time.time()
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
            if i != 0 :scheduler.step(valid_acc_his[i - 1])
            else:scheduler.step(valid_acc_his[0])
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
    tj = time.time() - ti
    print('Train time : {:.3f}'.format(tj))
    return train_acc_his, train_losses_his, valid_acc_his, valid_loss_his

def drew(train_acc_his, train_loss_his, valid_acc_his = None, valid_loss_his = None):
    plt.figure(figsize = (15, 10))
    plt.subplot(211)
    plt.plot(train_acc_his, 'b', label = 'training accuracy')
    plt.plot(valid_acc_his, 'r', label = 'validation accuracy')
    plt.title('Accuracy(%)')
    plt.legend(loc = 'best')
    plt.subplot(212)
    plt.plot(train_loss_his, 'b', label = 'training loss')
    plt.plot(valid_loss_his, 'r', label = 'valid loss')
    plt.title('Loss')
    plt.legend(loc = 'best')
    plt.show()
    
def main(args):
    cudnn.benchmark = True
    Train_Loader = Data.Data2(args.DataPath, args.batch_size, args.NumWorkers, True)
    Valid_Loader = Data.Data2(args.ValidPath, args.batch_size, args.NumWorkers, False)
    # Train_Loader, Valid_Loader = Data.Data(args.DataPath, args.batch_size, args.Cut_Number, args.NumWorkers)
    CNN = Mod.VGG16_Model(args.out)
    CNN.to(args.device)
    CNN.loss('S', args.learning_rate)
    CNN.out = args.out
    scheduler = CNN.scheduler(args.scheduler)
    train_acc_his, train_loss_his, valid_acc_his, valid_loss_his = train(args.epochs, CNN, Train_Loader, args.device, scheduler, True, Valid_Loader, args.ModuleName)
    drew(train_acc_his, train_loss_his, valid_acc_his, valid_loss_his)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, dest = 'epochs', help = 'the number of iteration', default = 200)
    parser.add_argument('--DataPath', type = str, dest = 'DataPath', help = 'the path of data', default = r'/home/lab504/DeepLearningTesting/ImageData/Test')
    parser.add_argument('--ValidPath', type = str, dest = 'ValidPath', help = 'the path of data', default = r'/home/lab504/DeepLearningTesting/ImageData/Valid')
    parser.add_argument('--batch_size', type = int, dest = 'batch_size', help = 'the number of batch', default = 32)
    parser.add_argument('--out', type = int, dest = 'out', help = 'the number of class', default = 3)
    parser.add_argument('--NumWorkers', type = int, dest = 'NumWorkers', help = 'the number of CUP workers', default = 4)
    parser.add_argument('--learning_rate', type = float, dest = 'learning_rate', default = 1e-3)
    parser.add_argument('--scheduler', type = int, dest = 'scheduler', default = 80)
    parser.add_argument('--device', type = str, dest = 'device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-Cut_Number-', type = float, dest = 'Cut_Number', default = 0.95)
    parser.add_argument('--ModuleName', type = str, dest = 'ModuleName', default = r'/workspace/ModuleSave/SN_Finished_Training_Model')
    args = parser.parse_args()
    main(args)
