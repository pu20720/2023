from torch.utils import data
import os
import glob
from PIL import Image
from torchvision import transforms
import torch

class Dataset(data.Dataset):
    def __init__(self, data_root, transform, Lable_Dict):
        self.transform = transform
        self.images = []
        self.name = []
        image_path = []
        image_data_path = glob.glob(os.path.join(data_root, '*'))
        for path in image_data_path:
            image_path.extend(glob.glob(os.path.join(path, '*.png')))
        for path in image_path:
            self.name.append(path)
            im = Image.open(path).convert('RGB').copy()
            im = self.transform(im)
            self.images.append(im)
        self.Lables = [os.path.dirname(path) for path in image_path]
        for lab in range(len(self.Lables)):
            self.Lables[lab] = self.Lables[lab].split('/')[-1]
            self.Lables[lab] = Lable_Dict[self.Lables[lab]]
    def __ImageName__(self):
        return self.name
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

def Data(path, batch_size, train_num, num_workers):
    Transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    Lable_Dict = {'bridge' : 0, 'normal' : 1, 'sn_less' : 2}
    dataset = Dataset(path, transform = Transforms, Lable_Dict = Lable_Dict)
    train_num = round(dataset.__len__() * train_num)
    train_data, valid_data = data.random_split(dataset,[train_num, dataset.__len__() - train_num])
    train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
    valid_loader = data.DataLoader(valid_data, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = num_workers, collate_fn = Dataset.collate_fn)
    return  train_loader, valid_loader

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