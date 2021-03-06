import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn
from torchvision import transforms as transforms
import numpy as np
import argparse
import torch.nn as nn
import math
import random
import torch

MODEL_CHOICE=18
OPT_CHOICE=1
LEARNING_RATE=0.006
EPOCH=100
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=64
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0],isfirst=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(8192, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,isfirst=False):
        downsample = None
        if isfirst or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 3, 3, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


class Runner(object):
    def __init__(self, config):
        self.cfg = config
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.device = None
        self.cuda = torch.cuda.is_available()
        self.train_loader = None
        self.test_loader = None

    def param_report(self):
        print('--------parameter report--------')
        print('learning rate:%.5f' % self.lr)
        print('epoch:%d' % self.epochs)
        print('train batch size:%d' % self.train_batch_size)
        print('test batch size:%d' % self.test_batch_size)
        if self.cfg.model_choice == 18:
            print('model:resnet18')
        elif self.cfg.model_choice == 50:
            print('model:resnet50')

        if self.cfg.opt_choice == 0:
            print('optimizer:Adam')
        elif self.cfg.opt_choice == 1:
            print('optimizer:SGD')
        print('-------------------------------')

    def result_report(self, loss_array, train_accuracy,accuracy_array, best_accuracy):
        print('--------result report--------')
        print('best accuracy:%.3f%%' % best_accuracy)
        print('loss array:')
        for loss in loss_array:
            print('%.4f' % loss, end=',')
        print('')
        print('train accuracy array:')
        for acc in train_accuracy:
            print('%.4f' % acc, end=',')
        print('')
        print('accuracy array:')
        for accuracy in accuracy_array:
            print('%.4f' % accuracy, end=',')
        print('')
        print('-------------------------------')

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        if self.cfg.model_choice == 18:
            self.model = resnet18().to(self.device)
        elif self.cfg.model_choice == 50:
            self.model = resnet50().to(self.device)

        if self.cfg.opt_choice == 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.cfg.opt_choice == 1:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            # print('Loss: %.4f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                # print('Loss: %.4f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def run(self):
        self.param_report()
        self.load_data()
        self.load_model()
        accuracy = 0
        loss_array = []
        accuracy_array = []
        train_accuracy = []
        for epoch in range(1, self.epochs + 1):
            # self.scheduler.step(epoch)
            print("epoch: %d start" % epoch)
            train_result = self.train()
            print(train_result)
            loss_array.append(train_result[0])
            train_accuracy.append(train_result[1])
            test_result = self.test()
            accuracy_array.append(test_result[1])
            accuracy = max(accuracy, test_result[1])
            print("epoch: %d accuracy %.3f%%" % (epoch, test_result[1] * 100))
            if epoch == self.epochs:
                print("best accuracy: %.3f%%" % (accuracy * 100))

        self.result_report(loss_array,train_accuracy, accuracy_array, accuracy * 100)
        return accuracy

def testBatchSize(batchSizeArray):
    accuracyArray = []
    for batchSize in batchSizeArray:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_choice', default=MODEL_CHOICE, type=int)  # resnet18:15 resnet50:50
        parser.add_argument('--opt_choice', default=OPT_CHOICE, type=int)  # 0:adam 1:sdg
        parser.add_argument('--lr', default=LEARNING_RATE, type=float)
        parser.add_argument('--epoch', default=EPOCH, type=int)
        parser.add_argument('--trainBatchSize', default=batchSize, type=int)
        parser.add_argument('--testBatchSize', default=batchSize, type=int)
        args = parser.parse_known_args()[0]
        runner = Runner(args)
        accuracyArray.append(runner.run())
    print("--------parameter report--------")
    for i in range(len(batchSizeArray)):
        print("batchSize:%d    accuracy:%.3f%%" % (batchSizeArray[i],accuracyArray[i]))


if __name__ == '__main__':
    setup_seed(2021)
    batchSizeArray = [16]
    testBatchSize(batchSizeArray)