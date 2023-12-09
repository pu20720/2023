import torch
import argparse
from torch.backends import cudnn
import time
import numpy
import Data
import BatchNorm as Mod

def valid(CNN, valid_loader, device):
    CNN.eval()
    start = time.time()
    valid_correct = 0
    i = 0
    for _data in valid_loader:
        image, label = _data
        image = image.to(device)
        label = label.to(device)
        pred = CNN(image) # pred：(batch_size*class) gpu
        output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出)
        valid_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數
        i+=1
    valid_acc = valid_correct / len(valid_loader.dataset) * 100 # 計算每一個epoch的平均驗證正確率(%)
    end = time.time()
    return (end - start), (valid_acc)

def main(args):
    cudnn.benchmark = True
    DataLoader, _= Data.Data(args.DataPath, 1, 1, args.NumWorkers)
    CNN = Mod.VGG16_Model(args.out)
    CNN.out = args.out
    CNN.load_state_dict(torch.load(args.ModuleName))
    CNN.to(args.device)
    fps, valid_acc = valid(CNN, DataLoader, args.device)
    print(f'Spands Time : {fps} (s)')
    print(f'The FPS : {DataLoader.__len__() / fps}')
    print(f'The accurary : {valid_acc}')
    print(f'Data Length : {DataLoader.__len__()}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataPath', type = str, dest = 'DataPath', help = 'the path of data', default = r'/home/lab504/DeepLearningTesting/ImageData/slince') # Datas' path
    parser.add_argument('--NumWorkers', type = int, dest = 'NumWorkers', help = 'the number of CUP workers', default = 4)
    parser.add_argument('--out', type = int, dest = 'out', help = 'the number of class', default = 3) # Classs number
    parser.add_argument('--ModuleName', type = str, dest = 'ModuleName', help = 'the path of module', default = r'/home/lab504/DeepLearningTesting/ModuleSave/DigitalTestModule_BN_GAP')
    parser.add_argument('--device', type = str, dest = 'device', default = 'cuda' if torch.cuda.is_available() else 'cpu') # CUDA or CPU
    args = parser.parse_args()
    main(args)