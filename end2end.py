from torch.utils.data import Dataset, DataLoader
from torchsummaryX import summary as summaryX
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
from colorama import Fore, Back
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torchvision
import argparse
import torch
import glob
import sys
import cv2
import os

image_size = 224
image_size_3d = (320, 120)

trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransferModel(nn.Module):
    def __init__(self):
        super(TransferModel, self).__init__()
        self.resnet50 = torchvision.models.resnet50()
        self.resnet50.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self,x):
        out = self.resnet50(x)
        out = self.relu(self.bn1(out))

        out = self.relu(self.bn2(self.fc1(out)))
        out = self.relu(self.bn3(self.fc2(out)))
        out = self.relu(self.bn4(self.fc3(out)))
        out = self.fc4(out)

        return out


class ThreeDModel(nn.Module):
    def __init__(self):
        super(ThreeDModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.lstm1 = nn.LSTM(input_size=14400, hidden_size=512, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=1)

        self.bn1 = nn.BatchNorm3d(num_features=3)
        self.bn2 = nn.BatchNorm3d(num_features=64)
        self.bn3 = nn.BatchNorm3d(num_features=8)
        # self.bn4 = nn.BatchNorm1d(num_features=512)
        # self.bn5 = nn.BatchNorm1d(num_features=128)
        # self.bn6 = nn.BatchNorm1d(num_features=64)
        # self.bn7 = nn.BatchNorm1d(num_features=16)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.maxpool = nn.MaxPool3d(kernel_size=2)


    def forward(self, x):
        in_batch_size = x.size(0)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn1(self.maxpool(out)))
        out = self.relu(self.bn1(self.maxpool(out)))

        out = self.conv2(out)
        identity = out
        out = self.relu(self.bn2(out))

        out = self.bn2(self.conv3(out))
        out += identity #residual connection.
        out = self.relu(out)

        out = self.conv3(out)
        identity = out
        out = self.relu(self.bn2(out))

        out = self.bn2(self.conv3(out))
        out += identity #residual connection.
        out = self.relu(out)

        out = self.relu(self.bn3(self.conv4(out)))
        out = self.relu(self.bn3(self.conv5(out)))
        out = self.bn3(self.conv5(out))

        #https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2
        #print(out.size())
        out = self.tanh(out)
        #print(Fore.GREEN, out.size())
        batch_size, frames, C, H, W = out.size()
        out = out.view(batch_size, frames, -1)
        #print(out.size())
        out, (hidden, cell) = self.lstm1(out)
        #print(out.size())
        out = self.tanh(out)
        out, (hidden, cell) = self.lstm2(out)

        #print(out.size())
        out = out.contiguous().view(in_batch_size, -1)#(n, 1000)
        #print(out.size())
        out = self.relu(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)


        return out


#for transfer model.
class MyDataset1(Dataset):
    def __init__(self, image_path=None, steers=None, images_name=None, transform=None):
        super(MyDataset1, self).__init__()
        self.image_path = image_path
        self.steers = steers
        self.images_name = images_name
        self.transform = transform
        self.images = []
        self.load_data()

    def load_data(self):

        print(self.images_name[32], self.steers[32])
        #self.images_name = self.images_name[:1000]
        #self.steers = self.steers[:1000]
        for image_name in tqdm(self.images_name, total=len(self.images_name), ncols=100):
            image_name = self.image_path + '\\' + image_name
            #img = Image.open(image_name) # it will cause Too many open files error if do this.
            self.images.append(image_name)

    def __len__(self):
        return len(self.steers)

    def __getitem__(self, idx):
        image = self.images[idx]
        steer = self.steers[idx]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, steer


#for 3d-lstm model.
class MyDataset2(Dataset):
    def __init__(self, image_path=None, steers=None, images_name=None, transform=None, sequence=5, frames=5):
        super(MyDataset2, self).__init__()
        self.image_path = image_path
        self.steers = steers
        self.images_name = images_name
        self.transform = transform
        self.sequence = sequence
        self.frames = frames
        self.images = []
        self.load_data()

    def load_data(self):
        print(self.images_name[100], self.steers[100])
        for image_name in tqdm(self.images_name, total=len(self.images_name), ncols=100):
            image_name = self.image_path + '\\' + image_name
            self.images.append(image_name)

    def __len__(self):
        return len(self.steers) - (self.sequence * self.frames - 1)

    def __getitem__(self, idx):
        imgs = self.images[idx:idx + self.sequence * self.frames]
        '''
        imgs = [imgs_[i:i+self.frames] for i in range(0, len(imgs_), self.frames)]
        #print(len(imgs))
        #print(imgs)
        images = [[] for i in range(len(imgs))]
        for i in range(len(imgs)):
            for img in imgs[i]:
                img = cv2.imread(img, -1)
                img = cv2.resize(img, image_size_3d)
                img = img / 255.0
                if self.transform:
                    img = self.transform(img)
                images[i].append(img)
        images = np.array(images)
        images = torch.from_numpy(images)#(5,5,120,320,3)
        images = images.view(images.size(0), images.size(4), images.size(1), images.size(2), images.size(3))#(5,3,5,120,320)
        '''
        images = []
        for img in imgs:
            img = cv2.imread(img, -1)
            img = cv2.resize(img, image_size_3d)
            img = img / 255.0
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = np.array(images)
        images = torch.from_numpy(images)#(25,120,320,3)
        images = images.view(images.size(3), images.size(0), images.size(1), images.size(2))#(3,25,120,320)
        steer = self.steers[idx + self.sequence * self.frames - 1]
        return images, steer



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def train(flag='transfer', continue_train=False):
    if flag == 'transfer': # 'transfer' or '3d'.
        model = TransferModel().cuda()
    elif flag == '3d':
        model = ThreeDModel().cuda()
    if continue_train:
        model.load_state_dict(torch.load('last_{}.pth'.format(flag)))
    #summary(model,input_size=(3,224,224))
    #print(model.resnet50.__dict__)
    # freeze some layer.
    # resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    if flag == 'transfer':
        model.resnet50.layer1.requires_grad = False
        model.resnet50.layer2[0].requires_grad = False

    images_name = []
    steers = []
    train_val_ratio = 0.8
    with open(r'F:\DataSets\driving-datasets\data.txt', 'r') as f:
        for line in f.readlines():
            steer = line.split(' ')[1].split(',')[0]
            image_name = line.split(' ')[0]
            images_name.append(image_name)
            steers.append(float(steer))
    images_name = images_name[:600]
    steers = steers[:600]
    train_number = int(train_val_ratio * len(steers))
    if flag == 'transfer':
        train_dataset = MyDataset1(image_path=r'F:\DataSets\driving-datasets\data', steers=steers[:train_number],
                                   images_name=images_name[:train_number], transform=trans)
        eval_dataset = MyDataset1(image_path=r'F:\DataSets\driving-datasets\data', steers=steers[train_number:],
                                  images_name=images_name[train_number:], transform=trans)
    elif flag == '3d':
        train_dataset = MyDataset2(image_path=r'F:\DataSets\driving-datasets\data', steers=steers[:train_number],
                                   images_name=images_name[:train_number])
        eval_dataset = MyDataset2(image_path=r'F:\DataSets\driving-datasets\data', steers=steers[train_number:],
                                  images_name=images_name[train_number:])
    images, steer = eval_dataset.__getitem__(8)
    #print(images.size())
    #print(steer)
    #return
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    epochs = 300
    best_loss = 1e9 + 7
    train_loss = [] #store RMSE loss.
    eval_loss = []
    for epoch in range(epochs):
        print('epoch {}:'.format(epoch))
        print('train:')
        model.train()
        criterion = nn.MSELoss()
        total_loss = 0
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)
        for index, (images, steers) in bar:
            images = Variable(images).cuda()
            steers = Variable(steers).cuda()
            images = images.type(torch.cuda.FloatTensor)
            steers = steers.type(torch.cuda.FloatTensor)
            steers = steers.view(steers.size(0), 1)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, steers)
            total_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
        print('train loss:{}'.format(np.sqrt(total_loss / len(train_dataset))))
        train_loss.append(np.sqrt(total_loss / len(train_dataset)))
        print('eval:')
        model.eval()
        eval_criterion = nn.MSELoss()
        total_loss = 0
        bar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=100)
        for index, (images, steers) in bar:
            images = Variable(images).cuda()
            steers = Variable(steers).cuda()
            images = images.type(torch.cuda.FloatTensor)
            steers = steers.type(torch.cuda.FloatTensor)
            steers = steers.view(steers.size(0), 1)
            optimizer.zero_grad()
            out = model(images)
            loss = eval_criterion(out, steers)
            total_loss += loss.data.cpu().numpy()
        print('eval loss:{}'.format(np.sqrt(total_loss / len(eval_dataset))))
        eval_loss.append(np.sqrt(total_loss / len(eval_dataset)))
        torch.save(model.state_dict(), 'last_{}.pth'.format(flag))
        if eval_loss[-1] < best_loss:
            torch.save(model.state_dict(), 'best_{}.pth'.format(flag))
            best_loss = eval_loss[-1]

    print('train loss: ', train_loss)
    pritn('eval loss: ', eval_loss)


def test_transfer_model():
    torch.no_grad()
    model = TransferModel().cuda()
    model.load_state_dict(torch.load('best_transfer.pth'))
    img = Image.open('test.jpg')
    img = trans(img).cuda()
    img = img.unsqueeze(0)
    img = Variable(img)
    model.eval()
    out = model(img)
    print(out.data.cpu().numpy()[0][0])


def test_3d_model():
    torch.no_grad()
    model = ThreeDModel().cuda()
    summaryX(model, torch.zeros((1, 3, 25, 120, 320)).cuda())
    f = nn.MSELoss()
    out = f(torch.rand((2,3)),torch.rand((2,3)))
    print(out)
    print(torch.zeros((1, 3, 25, 120, 320)).size())


#测试resnet是否正确加载pretrained模型.
def test_resnet():
    torch.no_grad()
    resnet50 = torchvision.models.resnet50().cuda()
    resnet50.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
    img = Image.open('test-resnet.jpg')
    #img = img.resize((image_size,image_size))
    img = trans(img).cuda()
    img = img.unsqueeze(0)
    img = Variable(img)
    resnet50.eval()
    out = resnet50(img)
    index = np.argmax(out.data.cpu().numpy())# this dog is Labrador retriever:208 
    print(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #参数为完整的函数名和其参数, 如test(a).
    parser.add_argument('--method', type=str, default=None, help='full method name and parameters.')
    opt = parser.parse_args()
    eval(opt.method)
