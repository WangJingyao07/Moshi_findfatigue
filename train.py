# -*- coding: utf-8 -*-
"""
本部分代码主要是根据竞赛所提供的imgs.zip中train数据集，训练一个finetune的renet34的模型；
__author__ = 'Wang Jingyao'
用visdom进行可视化输出
"""
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch as t
from torchvision  import models
import random
import visdom
import pdb
random.seed(1)
NUM_CLASSES = 10
LEARNING_RATE = 0.0001
MAX_EPOCHS = 100
use_gpu = True
BATCH_SIZE = 32
frequency_print = 100

def get_filepath(dir_root):
    file_paths = []
    for root, dirs, files in os.walk(dir_root):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class DriverDataset(data.Dataset):
    def __init__(self, data_root, transforms=None, train=True):
        self.train = train
        imgs_in = get_filepath(data_root)
        random.shuffle(imgs_in)
        imgs_num = len(imgs_in)
        
        if transforms is None:
            self.transforms = T.Compose([T.RandomResizedCrop(224), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        if self.train:
            self.imgs = imgs_in[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs_in[int(0.7*imgs_num):]
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = int(img_path.split('\\')[-2][1])
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.imgs)

model = models.resnet34(pretrained=True)

# for param in model.parameters():
#     param.require_grad = False

model.fc = nn.Linear(512, 10)#512为resnet34倒数第二层神经元的个数

if use_gpu:
    model.cuda()
    
#LOSS和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def val(model, dataloader, criterion):
    model.eval()
    acc_sum = 0
    for ii, (input, label) in enumerate(dataloader):
        val_input = input
        val_label = label
        if use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        output = model(val_input)
        acc_batch = t.mean(t.eq(t.max(output, 1)[1], val_label).float())
        acc_sum += acc_batch
        
    acc_vali = acc_sum / (ii + 1)
    model.train()
    return acc_vali


if __name__ == '__main__':

    train_data_path = '.\\data\\train'
    train_data = DriverDataset(train_data_path, train=True)
    train_dataloader = DataLoader(dataset=train_data,shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    vali_data = DriverDataset(train_data_path, train=False)
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

    vis = visdom.Visdom(env='driver')

    loss_print = []
    j = 0
    for epoch in range(MAX_EPOCHS):
        for (data_x, label) in train_dataloader:
            j += 1
            optimizer.zero_grad()
            #pdb.set_trace()
            input = data_x
            label = label
            if use_gpu:
                input = input.cuda()
                label = label.cuda()
        
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_print.append(loss)
            

            if j % frequency_print == 0:
                loss_mean = t.mean(t.Tensor(loss_print))
                loss_print = []
                print('train_loss: %f'%loss_mean)
                vis.line(X= t.Tensor([j]), Y=t.Tensor([loss_mean]), win='train loss', update='append' if j != frequency_print else None, opts=dict(title='train_loss', x_label='batch', y_label='loss'))

        acc_vali = val(model, vali_dataloader, criterion)
        print('第 %d epoch, acc_vali : %f' %(epoch,acc_vali))
        vis.line(X=t.Tensor([epoch]), Y=t.Tensor([acc_vali]), win='validation accuracy', update='append' if epoch != 0 else None, opts=dict(title='vali_acc', x_label='epoch', y_label='accuracy'))
        

        trainedmodel_path = './trained_models/%d'%epoch + '_' + '%f'%acc_vali + '.pkl'
        t.save(model, trainedmodel_path)
        vis.save(['driver'])
        
