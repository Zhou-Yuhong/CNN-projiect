```
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
LEARNING_RATE=0.005
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
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
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


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

    def result_report(self, loss_array, accuracy_array, best_accuracy):
        print('--------result report--------')
        print('best accuracy:%.3f%%' % best_accuracy)
        print('loss array:')
        for loss in loss_array:
            print('%.4f' % loss, end=',')
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

        for epoch in range(1, self.epochs + 1):
            # self.scheduler.step(epoch)
            print("epoch: %d start" % epoch)
            train_result = self.train()
            print(train_result)
            loss_array.append(train_result[0])
            test_result = self.test()
            accuracy_array.append(test_result[1])
            accuracy = max(accuracy, test_result[1])
            print("epoch: %d accuracy %.3f%%" % (epoch, test_result[1] * 100))
            if epoch == self.epochs:
                print("best accuracy: %.3f%%" % (accuracy * 100))

        self.result_report(loss_array, accuracy_array, accuracy * 100)
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
    batchSizeArray = [16,32,64,128,256,512]
    testBatchSize(batchSizeArray)
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:16
test batch size:16
model:resnet18
optimizer:SGD
-------------------------------
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data/cifar-10-python.tar.gz
```

170499072/? [00:03<00:00, 55119878.20it/s]

```
Extracting ../data/cifar-10-python.tar.gz to ../data
Files already downloaded and verified
epoch: 1 start
train:
(4970.4259506464, 0.41114)
test:
epoch: 1 accuracy 52.470%
epoch: 2 start
train:
(3794.7032459676266, 0.56236)
test:
epoch: 2 accuracy 47.510%
epoch: 3 start
train:
(3108.6603756546974, 0.64756)
test:
epoch: 3 accuracy 68.810%
epoch: 4 start
train:
(2630.313970297575, 0.70146)
test:
epoch: 4 accuracy 70.680%
epoch: 5 start
train:
(2280.693320810795, 0.7425)
test:
epoch: 5 accuracy 73.170%
epoch: 6 start
train:
(1958.8829331547022, 0.78304)
test:
epoch: 6 accuracy 75.620%
epoch: 7 start
train:
(1714.2193242013454, 0.81012)
test:
epoch: 7 accuracy 75.040%
epoch: 8 start
train:
(1482.4069445505738, 0.83704)
test:
epoch: 8 accuracy 76.370%
epoch: 9 start
train:
(1282.3870788775384, 0.85976)
test:
epoch: 9 accuracy 79.070%
epoch: 10 start
train:
(1091.8808952029794, 0.881)
test:
epoch: 10 accuracy 78.170%
epoch: 11 start
train:
(958.0127187240869, 0.89732)
test:
epoch: 11 accuracy 76.980%
epoch: 12 start
train:
(793.1145721897483, 0.91318)
test:
epoch: 12 accuracy 78.240%
epoch: 13 start
train:
(684.3848003624007, 0.92628)
test:
epoch: 13 accuracy 79.450%
epoch: 14 start
train:
(593.3860501423478, 0.9363)
test:
epoch: 14 accuracy 79.400%
epoch: 15 start
train:
(516.4155107138213, 0.94486)
test:
epoch: 15 accuracy 79.630%
epoch: 16 start
train:
(440.2063107294962, 0.95306)
test:
epoch: 16 accuracy 79.270%
epoch: 17 start
train:
(403.8786728422856, 0.9567)
test:
epoch: 17 accuracy 80.250%
epoch: 18 start
train:
(343.46740409056656, 0.96374)
test:
epoch: 18 accuracy 80.690%
epoch: 19 start
train:
(292.92232529306784, 0.96902)
test:
epoch: 19 accuracy 80.010%
epoch: 20 start
train:
(286.4842678399873, 0.96924)
test:
epoch: 20 accuracy 79.980%
epoch: 21 start
train:
(249.02652223579935, 0.97304)
test:
epoch: 21 accuracy 80.070%
epoch: 22 start
train:
(204.56954477005638, 0.97832)
test:
epoch: 22 accuracy 80.550%
epoch: 23 start
train:
(182.63394724426325, 0.98086)
test:
epoch: 23 accuracy 81.300%
epoch: 24 start
train:
(177.0492654577538, 0.98104)
test:
epoch: 24 accuracy 81.720%
epoch: 25 start
train:
(165.3319732575328, 0.98312)
test:
epoch: 25 accuracy 81.280%
epoch: 26 start
train:
(145.26774049754022, 0.98474)
test:
epoch: 26 accuracy 81.770%
epoch: 27 start
train:
(125.5066595500175, 0.98744)
test:
epoch: 27 accuracy 82.320%
epoch: 28 start
train:
(109.30551967840438, 0.98894)
test:
epoch: 28 accuracy 82.020%
epoch: 29 start
train:
(110.44735927178408, 0.9888)
test:
epoch: 29 accuracy 81.630%
epoch: 30 start
train:
(107.57978342759452, 0.9888)
test:
epoch: 30 accuracy 81.250%
epoch: 31 start
train:
(89.89184309060511, 0.99124)
test:
epoch: 31 accuracy 81.770%
epoch: 32 start
train:
(92.67719025812403, 0.99078)
test:
epoch: 32 accuracy 81.510%
epoch: 33 start
train:
(94.34672513071564, 0.99034)
test:
epoch: 33 accuracy 81.770%
epoch: 34 start
train:
(76.15759653506029, 0.99252)
test:
epoch: 34 accuracy 81.660%
epoch: 35 start
train:
(79.55228362369598, 0.99182)
test:
epoch: 35 accuracy 81.830%
epoch: 36 start
train:
(69.57371093602342, 0.99314)
test:
epoch: 36 accuracy 82.160%
epoch: 37 start
train:
(67.59991519536561, 0.99298)
test:
epoch: 37 accuracy 81.520%
epoch: 38 start
train:
(61.11979285524285, 0.99404)
test:
epoch: 38 accuracy 82.050%
epoch: 39 start
train:
(63.256327724990115, 0.9937)
test:
epoch: 39 accuracy 81.100%
epoch: 40 start
train:
(56.68648858452798, 0.99442)
test:
epoch: 40 accuracy 81.470%
epoch: 41 start
train:
(57.487382303443155, 0.99426)
test:
epoch: 41 accuracy 82.290%
epoch: 42 start
train:
(52.99008930620039, 0.99462)
test:
epoch: 42 accuracy 82.340%
epoch: 43 start
train:
(50.420715936597844, 0.9947)
test:
epoch: 43 accuracy 82.830%
epoch: 44 start
train:
(48.07655125699057, 0.99546)
test:
epoch: 44 accuracy 82.090%
epoch: 45 start
train:
(45.7258936124781, 0.99542)
test:
epoch: 45 accuracy 81.680%
epoch: 46 start
train:
(46.45231809802408, 0.99578)
test:
epoch: 46 accuracy 82.270%
epoch: 47 start
train:
(45.614997907822726, 0.99592)
test:
epoch: 47 accuracy 82.200%
epoch: 48 start
train:
(42.617521970847065, 0.99596)
test:
epoch: 48 accuracy 82.500%
epoch: 49 start
train:
(37.12718500053052, 0.99634)
test:
epoch: 49 accuracy 81.760%
epoch: 50 start
train:
(41.00648743388774, 0.99594)
test:
epoch: 50 accuracy 82.300%
epoch: 51 start
train:
(38.13495475390209, 0.99626)
test:
epoch: 51 accuracy 81.990%
epoch: 52 start
train:
(40.33067496734566, 0.99592)
test:
epoch: 52 accuracy 82.590%
epoch: 53 start
train:
(29.88586946535088, 0.9973)
test:
epoch: 53 accuracy 82.200%
epoch: 54 start
train:
(37.69250669140547, 0.99606)
test:
epoch: 54 accuracy 82.110%
epoch: 55 start
train:
(29.587404913954742, 0.99758)
test:
epoch: 55 accuracy 81.870%
epoch: 56 start
train:
(26.144849561671435, 0.99788)
test:
epoch: 56 accuracy 82.320%
epoch: 57 start
train:
(25.745530415847497, 0.9977)
test:
epoch: 57 accuracy 82.620%
epoch: 58 start
train:
(25.972928548583695, 0.99764)
test:
epoch: 58 accuracy 81.980%
epoch: 59 start
train:
(22.074900963948494, 0.99782)
test:
epoch: 59 accuracy 82.090%
epoch: 60 start
train:
(25.177502269987144, 0.9975)
test:
epoch: 60 accuracy 82.440%
epoch: 61 start
train:
(24.147171408062604, 0.99788)
test:
epoch: 61 accuracy 82.740%
epoch: 62 start
train:
(20.911643524557803, 0.99804)
test:
epoch: 62 accuracy 82.290%
epoch: 63 start
train:
(24.233062751818125, 0.99776)
test:
epoch: 63 accuracy 81.970%
epoch: 64 start
train:
(22.685063472183174, 0.99784)
test:
epoch: 64 accuracy 82.940%
epoch: 65 start
train:
(23.95489927328208, 0.99768)
test:
epoch: 65 accuracy 82.140%
epoch: 66 start
train:
(21.608464841032855, 0.99786)
test:
epoch: 66 accuracy 82.550%
epoch: 67 start
train:
(18.36326838605555, 0.9983)
test:
epoch: 67 accuracy 82.710%
epoch: 68 start
train:
(18.716738284404528, 0.99844)
test:
epoch: 68 accuracy 82.370%
epoch: 69 start
train:
(15.872378907839447, 0.99854)
test:
epoch: 69 accuracy 82.560%
epoch: 70 start
train:
(17.239128588763606, 0.99852)
test:
epoch: 70 accuracy 83.050%
epoch: 71 start
train:
(20.555890475804517, 0.99812)
test:
epoch: 71 accuracy 82.770%
epoch: 72 start
train:
(17.636852050736252, 0.99844)
test:
epoch: 72 accuracy 82.930%
epoch: 73 start
train:
(16.053785112204423, 0.99874)
test:
epoch: 73 accuracy 82.670%
epoch: 74 start
train:
(15.832145362900974, 0.99864)
test:
epoch: 74 accuracy 82.840%
epoch: 75 start
train:
(13.078026521271568, 0.99898)
test:
epoch: 75 accuracy 82.660%
epoch: 76 start
train:
(14.603096800091407, 0.99886)
test:
epoch: 76 accuracy 82.890%
epoch: 77 start
train:
(13.715194547014335, 0.99886)
test:
epoch: 77 accuracy 83.200%
epoch: 78 start
train:
(11.79041638517242, 0.99904)
test:
epoch: 78 accuracy 82.840%
epoch: 79 start
train:
(14.96104193448059, 0.99858)
test:
epoch: 79 accuracy 82.700%
epoch: 80 start
train:
(16.781687763951368, 0.99856)
test:
epoch: 80 accuracy 82.900%
epoch: 81 start
train:
(13.181217346124185, 0.99874)
test:
epoch: 81 accuracy 82.510%
epoch: 82 start
train:
(15.109356928045372, 0.9987)
test:
epoch: 82 accuracy 82.420%
epoch: 83 start
train:
(11.616458120392508, 0.99904)
test:
epoch: 83 accuracy 82.720%
epoch: 84 start
train:
(11.73439916178313, 0.99908)
test:
epoch: 84 accuracy 82.560%
epoch: 85 start
train:
(9.399401204716014, 0.99942)
test:
epoch: 85 accuracy 82.800%
epoch: 86 start
train:
(12.591251357735246, 0.9989)
test:
epoch: 86 accuracy 83.130%
epoch: 87 start
train:
(11.27621595186622, 0.99908)
test:
epoch: 87 accuracy 82.960%
epoch: 88 start
train:
(10.422177454583107, 0.99918)
test:
epoch: 88 accuracy 83.480%
epoch: 89 start
train:
(11.171597027268035, 0.99894)
test:
epoch: 89 accuracy 83.100%
epoch: 90 start
train:
(10.683286303506065, 0.99916)
test:
epoch: 90 accuracy 83.380%
epoch: 91 start
train:
(12.588944901842297, 0.99898)
test:
epoch: 91 accuracy 82.670%
epoch: 92 start
train:
(10.997859355913192, 0.99898)
test:
epoch: 92 accuracy 82.880%
epoch: 93 start
train:
(7.782184125586696, 0.99948)
test:
epoch: 93 accuracy 83.670%
epoch: 94 start
train:
(6.061363615920072, 0.99964)
test:
epoch: 94 accuracy 83.300%
epoch: 95 start
train:
(8.621786391719752, 0.99936)
test:
epoch: 95 accuracy 83.050%
epoch: 96 start
train:
(9.182392062404688, 0.99916)
test:
epoch: 96 accuracy 83.020%
epoch: 97 start
train:
(9.080328115767202, 0.99932)
test:
epoch: 97 accuracy 82.820%
epoch: 98 start
train:
(11.2920761941823, 0.99894)
test:
epoch: 98 accuracy 82.760%
epoch: 99 start
train:
(10.370926182202993, 0.9992)
test:
epoch: 99 accuracy 82.560%
epoch: 100 start
train:
(8.659277629887356, 0.99942)
test:
epoch: 100 accuracy 82.950%
best accuracy: 83.670%
--------result report--------
best accuracy:83.670%
loss array:
4970.4260,3794.7032,3108.6604,2630.3140,2280.6933,1958.8829,1714.2193,1482.4069,1282.3871,1091.8809,958.0127,793.1146,684.3848,593.3861,516.4155,440.2063,403.8787,343.4674,292.9223,286.4843,249.0265,204.5695,182.6339,177.0493,165.3320,145.2677,125.5067,109.3055,110.4474,107.5798,89.8918,92.6772,94.3467,76.1576,79.5523,69.5737,67.5999,61.1198,63.2563,56.6865,57.4874,52.9901,50.4207,48.0766,45.7259,46.4523,45.6150,42.6175,37.1272,41.0065,38.1350,40.3307,29.8859,37.6925,29.5874,26.1448,25.7455,25.9729,22.0749,25.1775,24.1472,20.9116,24.2331,22.6851,23.9549,21.6085,18.3633,18.7167,15.8724,17.2391,20.5559,17.6369,16.0538,15.8321,13.0780,14.6031,13.7152,11.7904,14.9610,16.7817,13.1812,15.1094,11.6165,11.7344,9.3994,12.5913,11.2762,10.4222,11.1716,10.6833,12.5889,10.9979,7.7822,6.0614,8.6218,9.1824,9.0803,11.2921,10.3709,8.6593,
accuracy array:
0.5247,0.4751,0.6881,0.7068,0.7317,0.7562,0.7504,0.7637,0.7907,0.7817,0.7698,0.7824,0.7945,0.7940,0.7963,0.7927,0.8025,0.8069,0.8001,0.7998,0.8007,0.8055,0.8130,0.8172,0.8128,0.8177,0.8232,0.8202,0.8163,0.8125,0.8177,0.8151,0.8177,0.8166,0.8183,0.8216,0.8152,0.8205,0.8110,0.8147,0.8229,0.8234,0.8283,0.8209,0.8168,0.8227,0.8220,0.8250,0.8176,0.8230,0.8199,0.8259,0.8220,0.8211,0.8187,0.8232,0.8262,0.8198,0.8209,0.8244,0.8274,0.8229,0.8197,0.8294,0.8214,0.8255,0.8271,0.8237,0.8256,0.8305,0.8277,0.8293,0.8267,0.8284,0.8266,0.8289,0.8320,0.8284,0.8270,0.8290,0.8251,0.8242,0.8272,0.8256,0.8280,0.8313,0.8296,0.8348,0.8310,0.8338,0.8267,0.8288,0.8367,0.8330,0.8305,0.8302,0.8282,0.8276,0.8256,0.8295,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:32
test batch size:32
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(2595.59017932415, 0.38476)
test:
epoch: 1 accuracy 44.900%
epoch: 2 start
train:
(2036.5242186188698, 0.5271)
test:
epoch: 2 accuracy 55.280%
epoch: 3 start
train:
(1731.2227463126183, 0.6032)
test:
epoch: 3 accuracy 61.800%
epoch: 4 start
train:
(1499.7645952701569, 0.66064)
test:
epoch: 4 accuracy 57.540%
epoch: 5 start
train:
(1326.477696299553, 0.69906)
test:
epoch: 5 accuracy 64.280%
epoch: 6 start
train:
(1185.759127855301, 0.73294)
test:
epoch: 6 accuracy 67.720%
epoch: 7 start
train:
(1068.3295429348946, 0.75946)
test:
epoch: 7 accuracy 69.130%
epoch: 8 start
train:
(952.1987969726324, 0.78678)
test:
epoch: 8 accuracy 72.910%
epoch: 9 start
train:
(857.496559664607, 0.81014)
test:
epoch: 9 accuracy 73.690%
epoch: 10 start
train:
(760.0967139750719, 0.83108)
test:
epoch: 10 accuracy 71.540%
epoch: 11 start
train:
(666.2589934244752, 0.85298)
test:
epoch: 11 accuracy 74.040%
epoch: 12 start
train:
(588.15347584337, 0.87212)
test:
epoch: 12 accuracy 74.380%
epoch: 13 start
train:
(517.4040581062436, 0.88802)
test:
epoch: 13 accuracy 76.380%
epoch: 14 start
train:
(445.30758713930845, 0.90478)
test:
epoch: 14 accuracy 75.650%
epoch: 15 start
train:
(376.1168563775718, 0.91862)
test:
epoch: 15 accuracy 75.900%
epoch: 16 start
train:
(321.50277421437204, 0.93142)
test:
epoch: 16 accuracy 74.430%
epoch: 17 start
train:
(276.4356928579509, 0.9422)
test:
epoch: 17 accuracy 71.490%
epoch: 18 start
train:
(235.20449725259095, 0.95222)
test:
epoch: 18 accuracy 73.830%
epoch: 19 start
train:
(195.18188517540693, 0.96006)
test:
epoch: 19 accuracy 76.620%
epoch: 20 start
train:
(173.34695857018232, 0.96488)
test:
epoch: 20 accuracy 74.900%
epoch: 21 start
train:
(148.90658595366403, 0.96882)
test:
epoch: 21 accuracy 75.820%
epoch: 22 start
train:
(129.49768989905715, 0.97348)
test:
epoch: 22 accuracy 76.710%
epoch: 23 start
train:
(114.36646221391857, 0.9764)
test:
epoch: 23 accuracy 77.290%
epoch: 24 start
train:
(96.84281394444406, 0.9811)
test:
epoch: 24 accuracy 76.830%
epoch: 25 start
train:
(90.9916142236907, 0.98172)
test:
epoch: 25 accuracy 77.560%
epoch: 26 start
train:
(72.11494912859052, 0.98614)
test:
epoch: 26 accuracy 76.630%
epoch: 27 start
train:
(74.47532502253307, 0.98512)
test:
epoch: 27 accuracy 75.760%
epoch: 28 start
train:
(64.97742017509881, 0.98748)
test:
epoch: 28 accuracy 76.430%
epoch: 29 start
train:
(58.01612308435142, 0.98908)
test:
epoch: 29 accuracy 78.350%
epoch: 30 start
train:
(55.905487172422, 0.989)
test:
epoch: 30 accuracy 77.670%
epoch: 31 start
train:
(46.37831476109568, 0.99154)
test:
epoch: 31 accuracy 77.350%
epoch: 32 start
train:
(45.85581381531665, 0.99138)
test:
epoch: 32 accuracy 77.510%
epoch: 33 start
train:
(37.87804672907805, 0.99332)
test:
epoch: 33 accuracy 78.190%
epoch: 34 start
train:
(36.128303627017885, 0.99362)
test:
epoch: 34 accuracy 78.710%
epoch: 35 start
train:
(29.38252746121725, 0.99496)
test:
epoch: 35 accuracy 77.850%
epoch: 36 start
train:
(29.554431388358353, 0.99496)
test:
epoch: 36 accuracy 75.550%
epoch: 37 start
train:
(28.88714567106217, 0.99486)
test:
epoch: 37 accuracy 78.520%
epoch: 38 start
train:
(27.82363401389739, 0.99496)
test:
epoch: 38 accuracy 78.390%
epoch: 39 start
train:
(28.535253945243312, 0.99502)
test:
epoch: 39 accuracy 78.530%
epoch: 40 start
train:
(26.455520559858996, 0.99538)
test:
epoch: 40 accuracy 78.280%
epoch: 41 start
train:
(26.20019670651527, 0.99516)
test:
epoch: 41 accuracy 78.660%
epoch: 42 start
train:
(19.404080759821227, 0.99692)
test:
epoch: 42 accuracy 79.260%
epoch: 43 start
train:
(19.75721173234342, 0.99644)
test:
epoch: 43 accuracy 79.140%
epoch: 44 start
train:
(19.485728234241833, 0.99638)
test:
epoch: 44 accuracy 78.700%
epoch: 45 start
train:
(19.948511742637493, 0.99686)
test:
epoch: 45 accuracy 79.220%
epoch: 46 start
train:
(12.902739658515202, 0.99826)
test:
epoch: 46 accuracy 79.210%
epoch: 47 start
train:
(14.698731893506192, 0.99758)
test:
epoch: 47 accuracy 77.750%
epoch: 48 start
train:
(15.313289562807768, 0.99758)
test:
epoch: 48 accuracy 79.080%
epoch: 49 start
train:
(16.131946158740902, 0.99724)
test:
epoch: 49 accuracy 79.200%
epoch: 50 start
train:
(11.881347153204842, 0.99818)
test:
epoch: 50 accuracy 78.850%
epoch: 51 start
train:
(13.502812734484905, 0.99764)
test:
epoch: 51 accuracy 79.050%
epoch: 52 start
train:
(10.624432282191265, 0.9987)
test:
epoch: 52 accuracy 79.580%
epoch: 53 start
train:
(10.731261109664047, 0.99852)
test:
epoch: 53 accuracy 79.330%
epoch: 54 start
train:
(10.817678019240702, 0.9984)
test:
epoch: 54 accuracy 79.370%
epoch: 55 start
train:
(11.898417509073624, 0.99804)
test:
epoch: 55 accuracy 78.490%
epoch: 56 start
train:
(9.656099177467695, 0.99856)
test:
epoch: 56 accuracy 78.840%
epoch: 57 start
train:
(11.457016330143233, 0.99814)
test:
epoch: 57 accuracy 79.270%
epoch: 58 start
train:
(9.975006299609959, 0.9985)
test:
epoch: 58 accuracy 79.210%
epoch: 59 start
train:
(11.03538313340323, 0.99822)
test:
epoch: 59 accuracy 79.140%
epoch: 60 start
train:
(9.16126342390271, 0.99864)
test:
epoch: 60 accuracy 79.180%
epoch: 61 start
train:
(7.990485090729635, 0.99882)
test:
epoch: 61 accuracy 79.270%
epoch: 62 start
train:
(7.088293458145927, 0.99892)
test:
epoch: 62 accuracy 79.560%
epoch: 63 start
train:
(8.21008203522797, 0.99878)
test:
epoch: 63 accuracy 79.170%
epoch: 64 start
train:
(6.757331900378631, 0.99906)
test:
epoch: 64 accuracy 79.270%
epoch: 65 start
train:
(7.176143139153282, 0.9989)
test:
epoch: 65 accuracy 78.860%
epoch: 66 start
train:
(6.289635567592995, 0.99932)
test:
epoch: 66 accuracy 79.240%
epoch: 67 start
train:
(5.555406184434105, 0.99936)
test:
epoch: 67 accuracy 79.430%
epoch: 68 start
train:
(7.961830660231499, 0.99874)
test:
epoch: 68 accuracy 78.530%
epoch: 69 start
train:
(6.828800911778671, 0.99914)
test:
epoch: 69 accuracy 78.150%
epoch: 70 start
train:
(6.935695938216668, 0.99892)
test:
epoch: 70 accuracy 79.210%
epoch: 71 start
train:
(4.958300780193895, 0.99938)
test:
epoch: 71 accuracy 79.350%
epoch: 72 start
train:
(4.58403900889607, 0.99942)
test:
epoch: 72 accuracy 79.230%
epoch: 73 start
train:
(6.282719566914238, 0.99912)
test:
epoch: 73 accuracy 79.230%
epoch: 74 start
train:
(5.459106735223031, 0.9994)
test:
epoch: 74 accuracy 79.530%
epoch: 75 start
train:
(5.479870350678539, 0.99926)
test:
epoch: 75 accuracy 79.660%
epoch: 76 start
train:
(4.622110891115881, 0.99946)
test:
epoch: 76 accuracy 79.600%
epoch: 77 start
train:
(4.042157143627264, 0.99952)
test:
epoch: 77 accuracy 80.000%
epoch: 78 start
train:
(6.626891763180538, 0.99888)
test:
epoch: 78 accuracy 79.190%
epoch: 79 start
train:
(4.818548091920093, 0.99934)
test:
epoch: 79 accuracy 79.060%
epoch: 80 start
train:
(5.22112656583613, 0.99924)
test:
epoch: 80 accuracy 79.080%
epoch: 81 start
train:
(5.028121637773438, 0.9993)
test:
epoch: 81 accuracy 79.390%
epoch: 82 start
train:
(7.270209828035149, 0.99864)
test:
epoch: 82 accuracy 78.960%
epoch: 83 start
train:
(5.779854419564799, 0.99912)
test:
epoch: 83 accuracy 79.170%
epoch: 84 start
train:
(7.532619388524836, 0.99866)
test:
epoch: 84 accuracy 78.880%
epoch: 85 start
train:
(5.788492475194289, 0.99922)
test:
epoch: 85 accuracy 79.280%
epoch: 86 start
train:
(8.957057164927392, 0.99828)
test:
epoch: 86 accuracy 79.080%
epoch: 87 start
train:
(6.914449753990084, 0.99884)
test:
epoch: 87 accuracy 79.650%
epoch: 88 start
train:
(4.605136261247026, 0.9994)
test:
epoch: 88 accuracy 79.540%
epoch: 89 start
train:
(5.481274846839369, 0.99916)
test:
epoch: 89 accuracy 79.060%
epoch: 90 start
train:
(4.96594738175736, 0.99936)
test:
epoch: 90 accuracy 79.560%
epoch: 91 start
train:
(4.683437345478524, 0.99938)
test:
epoch: 91 accuracy 79.520%
epoch: 92 start
train:
(3.407484637495145, 0.99966)
test:
epoch: 92 accuracy 79.280%
epoch: 93 start
train:
(3.9386320935809636, 0.99936)
test:
epoch: 93 accuracy 79.240%
epoch: 94 start
train:
(3.6805854300619103, 0.99956)
test:
epoch: 94 accuracy 79.310%
epoch: 95 start
train:
(3.1890585287947033, 0.99958)
test:
epoch: 95 accuracy 78.780%
epoch: 96 start
train:
(4.187664066326761, 0.9995)
test:
epoch: 96 accuracy 78.880%
epoch: 97 start
train:
(3.813029218756128, 0.99946)
test:
epoch: 97 accuracy 79.030%
epoch: 98 start
train:
(3.0418441811525554, 0.99954)
test:
epoch: 98 accuracy 79.070%
epoch: 99 start
train:
(3.938105202243605, 0.99954)
test:
epoch: 99 accuracy 78.990%
epoch: 100 start
train:
(3.2609148042902234, 0.99958)
test:
epoch: 100 accuracy 79.050%
best accuracy: 80.000%
--------result report--------
best accuracy:80.000%
loss array:
2595.5902,2036.5242,1731.2227,1499.7646,1326.4777,1185.7591,1068.3295,952.1988,857.4966,760.0967,666.2590,588.1535,517.4041,445.3076,376.1169,321.5028,276.4357,235.2045,195.1819,173.3470,148.9066,129.4977,114.3665,96.8428,90.9916,72.1149,74.4753,64.9774,58.0161,55.9055,46.3783,45.8558,37.8780,36.1283,29.3825,29.5544,28.8871,27.8236,28.5353,26.4555,26.2002,19.4041,19.7572,19.4857,19.9485,12.9027,14.6987,15.3133,16.1319,11.8813,13.5028,10.6244,10.7313,10.8177,11.8984,9.6561,11.4570,9.9750,11.0354,9.1613,7.9905,7.0883,8.2101,6.7573,7.1761,6.2896,5.5554,7.9618,6.8288,6.9357,4.9583,4.5840,6.2827,5.4591,5.4799,4.6221,4.0422,6.6269,4.8185,5.2211,5.0281,7.2702,5.7799,7.5326,5.7885,8.9571,6.9144,4.6051,5.4813,4.9659,4.6834,3.4075,3.9386,3.6806,3.1891,4.1877,3.8130,3.0418,3.9381,3.2609,
accuracy array:
0.4490,0.5528,0.6180,0.5754,0.6428,0.6772,0.6913,0.7291,0.7369,0.7154,0.7404,0.7438,0.7638,0.7565,0.7590,0.7443,0.7149,0.7383,0.7662,0.7490,0.7582,0.7671,0.7729,0.7683,0.7756,0.7663,0.7576,0.7643,0.7835,0.7767,0.7735,0.7751,0.7819,0.7871,0.7785,0.7555,0.7852,0.7839,0.7853,0.7828,0.7866,0.7926,0.7914,0.7870,0.7922,0.7921,0.7775,0.7908,0.7920,0.7885,0.7905,0.7958,0.7933,0.7937,0.7849,0.7884,0.7927,0.7921,0.7914,0.7918,0.7927,0.7956,0.7917,0.7927,0.7886,0.7924,0.7943,0.7853,0.7815,0.7921,0.7935,0.7923,0.7923,0.7953,0.7966,0.7960,0.8000,0.7919,0.7906,0.7908,0.7939,0.7896,0.7917,0.7888,0.7928,0.7908,0.7965,0.7954,0.7906,0.7956,0.7952,0.7928,0.7924,0.7931,0.7878,0.7888,0.7903,0.7907,0.7899,0.7905,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:64
test batch size:64
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(1405.6457923650742, 0.33432)
test:
epoch: 1 accuracy 35.950%
epoch: 2 start
train:
(1145.011592745781, 0.46054)
test:
epoch: 2 accuracy 37.270%
epoch: 3 start
train:
(1019.2363626360893, 0.52354)
test:
epoch: 3 accuracy 45.770%
epoch: 4 start
train:
(926.8611734509468, 0.5693)
test:
epoch: 4 accuracy 51.970%
epoch: 5 start
train:
(844.267514526844, 0.61282)
test:
epoch: 5 accuracy 49.310%
epoch: 6 start
train:
(774.4711043834686, 0.64548)
test:
epoch: 6 accuracy 49.750%
epoch: 7 start
train:
(708.1586858034134, 0.67912)
test:
epoch: 7 accuracy 59.300%
epoch: 8 start
train:
(661.274550139904, 0.70146)
test:
epoch: 8 accuracy 56.160%
epoch: 9 start
train:
(609.2947685420513, 0.72342)
test:
epoch: 9 accuracy 54.550%
epoch: 10 start
train:
(567.3477896153927, 0.74464)
test:
epoch: 10 accuracy 56.160%
epoch: 11 start
train:
(526.563418596983, 0.7623)
test:
epoch: 11 accuracy 67.340%
epoch: 12 start
train:
(482.603760689497, 0.78476)
test:
epoch: 12 accuracy 66.720%
epoch: 13 start
train:
(444.2723713517189, 0.80336)
test:
epoch: 13 accuracy 45.640%
epoch: 14 start
train:
(407.6685175895691, 0.81966)
test:
epoch: 14 accuracy 49.950%
epoch: 15 start
train:
(366.7320205718279, 0.83894)
test:
epoch: 15 accuracy 66.240%
epoch: 16 start
train:
(334.0637405067682, 0.85546)
test:
epoch: 16 accuracy 71.530%
epoch: 17 start
train:
(298.4558124691248, 0.8707)
test:
epoch: 17 accuracy 66.180%
epoch: 18 start
train:
(265.57386896014214, 0.88604)
test:
epoch: 18 accuracy 47.770%
epoch: 19 start
train:
(237.00424731522799, 0.90006)
test:
epoch: 19 accuracy 71.380%
epoch: 20 start
train:
(206.1842200383544, 0.91332)
test:
epoch: 20 accuracy 56.200%
epoch: 21 start
train:
(178.50922556221485, 0.9266)
test:
epoch: 21 accuracy 66.820%
epoch: 22 start
train:
(153.3955520056188, 0.93896)
test:
epoch: 22 accuracy 55.640%
epoch: 23 start
train:
(135.4480801038444, 0.94612)
test:
epoch: 23 accuracy 40.360%
epoch: 24 start
train:
(112.59321635216475, 0.9569)
test:
epoch: 24 accuracy 68.070%
epoch: 25 start
train:
(97.36227472871542, 0.96246)
test:
epoch: 25 accuracy 68.600%
epoch: 26 start
train:
(79.74664371274412, 0.9713)
test:
epoch: 26 accuracy 71.600%
epoch: 27 start
train:
(66.56088784709573, 0.97676)
test:
epoch: 27 accuracy 72.100%
epoch: 28 start
train:
(58.928116073831916, 0.97924)
test:
epoch: 28 accuracy 62.180%
epoch: 29 start
train:
(49.09852936118841, 0.98386)
test:
epoch: 29 accuracy 58.880%
epoch: 30 start
train:
(46.00321146287024, 0.98382)
test:
epoch: 30 accuracy 74.250%
epoch: 31 start
train:
(35.99473876366392, 0.98876)
test:
epoch: 31 accuracy 73.670%
epoch: 32 start
train:
(30.834544917102903, 0.99108)
test:
epoch: 32 accuracy 66.040%
epoch: 33 start
train:
(28.263148745521903, 0.99144)
test:
epoch: 33 accuracy 73.460%
epoch: 34 start
train:
(25.87601809343323, 0.9923)
test:
epoch: 34 accuracy 73.580%
epoch: 35 start
train:
(20.962825229857117, 0.99412)
test:
epoch: 35 accuracy 73.150%
epoch: 36 start
train:
(19.432338969083503, 0.99496)
test:
epoch: 36 accuracy 71.660%
epoch: 37 start
train:
(19.610758491791785, 0.99434)
test:
epoch: 37 accuracy 74.180%
epoch: 38 start
train:
(14.085488222306594, 0.99672)
test:
epoch: 38 accuracy 73.920%
epoch: 39 start
train:
(14.378029803046957, 0.99622)
test:
epoch: 39 accuracy 73.940%
epoch: 40 start
train:
(12.970107580185868, 0.99686)
test:
epoch: 40 accuracy 70.390%
epoch: 41 start
train:
(12.581945493235253, 0.99678)
test:
epoch: 41 accuracy 74.490%
epoch: 42 start
train:
(10.823313224012963, 0.99744)
test:
epoch: 42 accuracy 71.660%
epoch: 43 start
train:
(11.600311509566382, 0.99698)
test:
epoch: 43 accuracy 48.910%
epoch: 44 start
train:
(11.158477646415122, 0.99706)
test:
epoch: 44 accuracy 72.760%
epoch: 45 start
train:
(9.139133406686597, 0.99806)
test:
epoch: 45 accuracy 74.220%
epoch: 46 start
train:
(9.50072784931399, 0.99748)
test:
epoch: 46 accuracy 74.630%
epoch: 47 start
train:
(7.335416405345313, 0.99848)
test:
epoch: 47 accuracy 69.400%
epoch: 48 start
train:
(8.642216329579242, 0.9979)
test:
epoch: 48 accuracy 54.270%
epoch: 49 start
train:
(9.628344272612594, 0.99764)
test:
epoch: 49 accuracy 71.120%
epoch: 50 start
train:
(6.570266124035697, 0.99852)
test:
epoch: 50 accuracy 73.520%
epoch: 51 start
train:
(5.728327285236446, 0.99886)
test:
epoch: 51 accuracy 74.940%
epoch: 52 start
train:
(4.929199917067308, 0.99916)
test:
epoch: 52 accuracy 75.210%
epoch: 53 start
train:
(5.315507088409504, 0.99888)
test:
epoch: 53 accuracy 75.080%
epoch: 54 start
train:
(3.749585808633128, 0.99968)
test:
epoch: 54 accuracy 75.590%
epoch: 55 start
train:
(4.093255859130295, 0.9993)
test:
epoch: 55 accuracy 71.980%
epoch: 56 start
train:
(4.868244690645952, 0.99902)
test:
epoch: 56 accuracy 75.560%
epoch: 57 start
train:
(3.783919032342965, 0.99932)
test:
epoch: 57 accuracy 70.010%
epoch: 58 start
train:
(4.642436808702769, 0.99892)
test:
epoch: 58 accuracy 75.460%
epoch: 59 start
train:
(3.8093464891426265, 0.99946)
test:
epoch: 59 accuracy 74.270%
epoch: 60 start
train:
(3.48642125990591, 0.99938)
test:
epoch: 60 accuracy 75.400%
epoch: 61 start
train:
(2.9788282843946945, 0.9996)
test:
epoch: 61 accuracy 74.540%
epoch: 62 start
train:
(3.262345068680588, 0.9994)
test:
epoch: 62 accuracy 75.350%
epoch: 63 start
train:
(3.861329800216481, 0.99922)
test:
epoch: 63 accuracy 72.250%
epoch: 64 start
train:
(4.589398006646661, 0.99918)
test:
epoch: 64 accuracy 69.030%
epoch: 65 start
train:
(4.9995555251371115, 0.99888)
test:
epoch: 65 accuracy 75.780%
epoch: 66 start
train:
(3.8472050702839624, 0.99908)
test:
epoch: 66 accuracy 74.340%
epoch: 67 start
train:
(3.1474736153322738, 0.99948)
test:
epoch: 67 accuracy 66.130%
epoch: 68 start
train:
(4.5067377525847405, 0.99902)
test:
epoch: 68 accuracy 73.930%
epoch: 69 start
train:
(3.5626947415585164, 0.9992)
test:
epoch: 69 accuracy 74.850%
epoch: 70 start
train:
(3.383116517070448, 0.99936)
test:
epoch: 70 accuracy 66.810%
epoch: 71 start
train:
(4.24682943712105, 0.999)
test:
epoch: 71 accuracy 75.360%
epoch: 72 start
train:
(3.33880550620961, 0.9993)
test:
epoch: 72 accuracy 71.020%
epoch: 73 start
train:
(4.480241150071379, 0.99904)
test:
epoch: 73 accuracy 75.440%
epoch: 74 start
train:
(2.8751072134036804, 0.99954)
test:
epoch: 74 accuracy 75.350%
epoch: 75 start
train:
(2.749110908684088, 0.99948)
test:
epoch: 75 accuracy 69.370%
epoch: 76 start
train:
(3.71481721336022, 0.9991)
test:
epoch: 76 accuracy 62.870%
epoch: 77 start
train:
(4.241108517089742, 0.99908)
test:
epoch: 77 accuracy 75.720%
epoch: 78 start
train:
(2.193847794260364, 0.99978)
test:
epoch: 78 accuracy 75.410%
epoch: 79 start
train:
(2.6726490476576146, 0.99942)
test:
epoch: 79 accuracy 74.980%
epoch: 80 start
train:
(2.494834269033163, 0.99958)
test:
epoch: 80 accuracy 74.600%
epoch: 81 start
train:
(1.910355812346097, 0.99984)
test:
epoch: 81 accuracy 70.550%
epoch: 82 start
train:
(2.8768377946398687, 0.99938)
test:
epoch: 82 accuracy 75.450%
epoch: 83 start
train:
(1.9761366728198482, 0.99978)
test:
epoch: 83 accuracy 74.640%
epoch: 84 start
train:
(2.086791934198118, 0.99966)
test:
epoch: 84 accuracy 65.660%
epoch: 85 start
train:
(2.7435753116878914, 0.99942)
test:
epoch: 85 accuracy 75.130%
epoch: 86 start
train:
(3.0065113253949676, 0.99938)
test:
epoch: 86 accuracy 73.760%
epoch: 87 start
train:
(2.6686131720780395, 0.99958)
test:
epoch: 87 accuracy 70.450%
epoch: 88 start
train:
(2.358556658786256, 0.9996)
test:
epoch: 88 accuracy 75.450%
epoch: 89 start
train:
(1.690843048185343, 0.99974)
test:
epoch: 89 accuracy 75.410%
epoch: 90 start
train:
(1.8955871867219685, 0.99974)
test:
epoch: 90 accuracy 56.670%
epoch: 91 start
train:
(3.079398840010981, 0.99942)
test:
epoch: 91 accuracy 75.360%
epoch: 92 start
train:
(2.204295289819129, 0.99962)
test:
epoch: 92 accuracy 58.650%
epoch: 93 start
train:
(3.0805148948857095, 0.99936)
test:
epoch: 93 accuracy 49.770%
epoch: 94 start
train:
(6.370755184412701, 0.99792)
test:
epoch: 94 accuracy 60.920%
epoch: 95 start
train:
(4.846344405421405, 0.99868)
test:
epoch: 95 accuracy 59.570%
epoch: 96 start
train:
(7.733341758328606, 0.99768)
test:
epoch: 96 accuracy 74.480%
epoch: 97 start
train:
(4.268135212449124, 0.99902)
test:
epoch: 97 accuracy 71.150%
epoch: 98 start
train:
(4.7998185505275615, 0.99854)
test:
epoch: 98 accuracy 74.160%
epoch: 99 start
train:
(3.7549727673322195, 0.99892)
test:
epoch: 99 accuracy 75.280%
epoch: 100 start
train:
(2.7626047929225024, 0.99918)
test:
epoch: 100 accuracy 75.170%
best accuracy: 75.780%
--------result report--------
best accuracy:75.780%
loss array:
1405.6458,1145.0116,1019.2364,926.8612,844.2675,774.4711,708.1587,661.2746,609.2948,567.3478,526.5634,482.6038,444.2724,407.6685,366.7320,334.0637,298.4558,265.5739,237.0042,206.1842,178.5092,153.3956,135.4481,112.5932,97.3623,79.7466,66.5609,58.9281,49.0985,46.0032,35.9947,30.8345,28.2631,25.8760,20.9628,19.4323,19.6108,14.0855,14.3780,12.9701,12.5819,10.8233,11.6003,11.1585,9.1391,9.5007,7.3354,8.6422,9.6283,6.5703,5.7283,4.9292,5.3155,3.7496,4.0933,4.8682,3.7839,4.6424,3.8093,3.4864,2.9788,3.2623,3.8613,4.5894,4.9996,3.8472,3.1475,4.5067,3.5627,3.3831,4.2468,3.3388,4.4802,2.8751,2.7491,3.7148,4.2411,2.1938,2.6726,2.4948,1.9104,2.8768,1.9761,2.0868,2.7436,3.0065,2.6686,2.3586,1.6908,1.8956,3.0794,2.2043,3.0805,6.3708,4.8463,7.7333,4.2681,4.7998,3.7550,2.7626,
accuracy array:
0.3595,0.3727,0.4577,0.5197,0.4931,0.4975,0.5930,0.5616,0.5455,0.5616,0.6734,0.6672,0.4564,0.4995,0.6624,0.7153,0.6618,0.4777,0.7138,0.5620,0.6682,0.5564,0.4036,0.6807,0.6860,0.7160,0.7210,0.6218,0.5888,0.7425,0.7367,0.6604,0.7346,0.7358,0.7315,0.7166,0.7418,0.7392,0.7394,0.7039,0.7449,0.7166,0.4891,0.7276,0.7422,0.7463,0.6940,0.5427,0.7112,0.7352,0.7494,0.7521,0.7508,0.7559,0.7198,0.7556,0.7001,0.7546,0.7427,0.7540,0.7454,0.7535,0.7225,0.6903,0.7578,0.7434,0.6613,0.7393,0.7485,0.6681,0.7536,0.7102,0.7544,0.7535,0.6937,0.6287,0.7572,0.7541,0.7498,0.7460,0.7055,0.7545,0.7464,0.6566,0.7513,0.7376,0.7045,0.7545,0.7541,0.5667,0.7536,0.5865,0.4977,0.6092,0.5957,0.7448,0.7115,0.7416,0.7528,0.7517,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:128
test batch size:128
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(756.6213338375092, 0.29332)
test:
epoch: 1 accuracy 35.070%
epoch: 2 start
train:
(626.1516859531403, 0.41074)
test:
epoch: 2 accuracy 44.820%
epoch: 3 start
train:
(566.0290722846985, 0.47028)
test:
epoch: 3 accuracy 46.790%
epoch: 4 start
train:
(525.1907571554184, 0.5117)
test:
epoch: 4 accuracy 49.510%
epoch: 5 start
train:
(494.0737110376358, 0.54176)
test:
epoch: 5 accuracy 49.540%
epoch: 6 start
train:
(466.07475131750107, 0.56748)
test:
epoch: 6 accuracy 55.740%
epoch: 7 start
train:
(442.6736940741539, 0.59258)
test:
epoch: 7 accuracy 52.670%
epoch: 8 start
train:
(419.0372961759567, 0.61514)
test:
epoch: 8 accuracy 50.130%
epoch: 9 start
train:
(397.95148199796677, 0.63508)
test:
epoch: 9 accuracy 55.820%
epoch: 10 start
train:
(377.828746676445, 0.65616)
test:
epoch: 10 accuracy 58.550%
epoch: 11 start
train:
(358.43398946523666, 0.6719)
test:
epoch: 11 accuracy 61.050%
epoch: 12 start
train:
(340.820445895195, 0.68996)
test:
epoch: 12 accuracy 58.360%
epoch: 13 start
train:
(325.35024803876877, 0.70516)
test:
epoch: 13 accuracy 64.890%
epoch: 14 start
train:
(311.46435594558716, 0.7186)
test:
epoch: 14 accuracy 62.380%
epoch: 15 start
train:
(296.46740847826004, 0.73276)
test:
epoch: 15 accuracy 63.650%
epoch: 16 start
train:
(283.07824552059174, 0.74472)
test:
epoch: 16 accuracy 66.800%
epoch: 17 start
train:
(269.3161550760269, 0.75928)
test:
epoch: 17 accuracy 66.000%
epoch: 18 start
train:
(255.79007270932198, 0.77326)
test:
epoch: 18 accuracy 65.870%
epoch: 19 start
train:
(243.405175447464, 0.78322)
test:
epoch: 19 accuracy 66.680%
epoch: 20 start
train:
(230.46873354911804, 0.79812)
test:
epoch: 20 accuracy 66.330%
epoch: 21 start
train:
(215.81407994031906, 0.81)
test:
epoch: 21 accuracy 64.130%
epoch: 22 start
train:
(202.74092215299606, 0.8222)
test:
epoch: 22 accuracy 65.040%
epoch: 23 start
train:
(190.18639186024666, 0.83422)
test:
epoch: 23 accuracy 66.470%
epoch: 24 start
train:
(177.4725096821785, 0.8476)
test:
epoch: 24 accuracy 66.040%
epoch: 25 start
train:
(163.47950984537601, 0.86074)
test:
epoch: 25 accuracy 59.790%
epoch: 26 start
train:
(152.03747148811817, 0.87096)
test:
epoch: 26 accuracy 66.760%
epoch: 27 start
train:
(139.35427071154118, 0.8839)
test:
epoch: 27 accuracy 65.120%
epoch: 28 start
train:
(126.7482550740242, 0.89558)
test:
epoch: 28 accuracy 66.930%
epoch: 29 start
train:
(114.05034171044827, 0.90882)
test:
epoch: 29 accuracy 65.450%
epoch: 30 start
train:
(102.13749958574772, 0.91882)
test:
epoch: 30 accuracy 68.030%
epoch: 31 start
train:
(91.67551813274622, 0.92896)
test:
epoch: 31 accuracy 69.420%
epoch: 32 start
train:
(81.3071354329586, 0.93906)
test:
epoch: 32 accuracy 69.320%
epoch: 33 start
train:
(70.38540463894606, 0.9481)
test:
epoch: 33 accuracy 65.180%
epoch: 34 start
train:
(61.20384855568409, 0.9577)
test:
epoch: 34 accuracy 68.210%
epoch: 35 start
train:
(54.06293799728155, 0.9624)
test:
epoch: 35 accuracy 69.610%
epoch: 36 start
train:
(46.71573527902365, 0.96996)
test:
epoch: 36 accuracy 67.240%
epoch: 37 start
train:
(38.44283473864198, 0.9772)
test:
epoch: 37 accuracy 68.390%
epoch: 38 start
train:
(33.47631385922432, 0.98136)
test:
epoch: 38 accuracy 69.680%
epoch: 39 start
train:
(27.70741480588913, 0.98526)
test:
epoch: 39 accuracy 67.180%
epoch: 40 start
train:
(24.37146505713463, 0.98764)
test:
epoch: 40 accuracy 70.340%
epoch: 41 start
train:
(20.360621536150575, 0.99024)
test:
epoch: 41 accuracy 70.880%
epoch: 42 start
train:
(17.584111098200083, 0.99292)
test:
epoch: 42 accuracy 69.080%
epoch: 43 start
train:
(14.562612928450108, 0.99456)
test:
epoch: 43 accuracy 70.460%
epoch: 44 start
train:
(12.061191864311695, 0.99588)
test:
epoch: 44 accuracy 71.120%
epoch: 45 start
train:
(11.119676992297173, 0.9966)
test:
epoch: 45 accuracy 67.970%
epoch: 46 start
train:
(10.345010668039322, 0.99694)
test:
epoch: 46 accuracy 71.410%
epoch: 47 start
train:
(9.03585640527308, 0.9974)
test:
epoch: 47 accuracy 71.320%
epoch: 48 start
train:
(8.312081805430353, 0.99772)
test:
epoch: 48 accuracy 70.470%
epoch: 49 start
train:
(7.741097501013428, 0.99786)
test:
epoch: 49 accuracy 70.900%
epoch: 50 start
train:
(6.3139992114156485, 0.9987)
test:
epoch: 50 accuracy 71.760%
epoch: 51 start
train:
(5.495253950823098, 0.99904)
test:
epoch: 51 accuracy 70.560%
epoch: 52 start
train:
(4.991608812939376, 0.99898)
test:
epoch: 52 accuracy 68.550%
epoch: 53 start
train:
(4.759762950241566, 0.99924)
test:
epoch: 53 accuracy 71.630%
epoch: 54 start
train:
(4.051357264397666, 0.99948)
test:
epoch: 54 accuracy 57.480%
epoch: 55 start
train:
(4.280162724899128, 0.9991)
test:
epoch: 55 accuracy 71.380%
epoch: 56 start
train:
(3.850126050412655, 0.99946)
test:
epoch: 56 accuracy 71.480%
epoch: 57 start
train:
(3.25629083160311, 0.9997)
test:
epoch: 57 accuracy 71.870%
epoch: 58 start
train:
(3.001191111514345, 0.99968)
test:
epoch: 58 accuracy 71.830%
epoch: 59 start
train:
(3.168139612302184, 0.99946)
test:
epoch: 59 accuracy 68.950%
epoch: 60 start
train:
(3.0910777187673375, 0.9996)
test:
epoch: 60 accuracy 71.240%
epoch: 61 start
train:
(2.7845410390291363, 0.99972)
test:
epoch: 61 accuracy 72.020%
epoch: 62 start
train:
(2.6206686950754374, 0.99968)
test:
epoch: 62 accuracy 71.760%
epoch: 63 start
train:
(2.343396790442057, 0.9998)
test:
epoch: 63 accuracy 72.060%
epoch: 64 start
train:
(2.5175656110513955, 0.99966)
test:
epoch: 64 accuracy 71.890%
epoch: 65 start
train:
(2.2099272495834157, 0.99972)
test:
epoch: 65 accuracy 69.490%
epoch: 66 start
train:
(2.041892028064467, 0.99982)
test:
epoch: 66 accuracy 71.810%
epoch: 67 start
train:
(2.00547507207375, 0.99984)
test:
epoch: 67 accuracy 71.670%
epoch: 68 start
train:
(1.7354521707165986, 0.99992)
test:
epoch: 68 accuracy 72.080%
epoch: 69 start
train:
(2.1150747812353075, 0.99974)
test:
epoch: 69 accuracy 72.060%
epoch: 70 start
train:
(1.724035908933729, 0.9999)
test:
epoch: 70 accuracy 72.070%
epoch: 71 start
train:
(1.6718573436373845, 0.99988)
test:
epoch: 71 accuracy 71.780%
epoch: 72 start
train:
(1.5964040125254542, 0.99994)
test:
epoch: 72 accuracy 72.060%
epoch: 73 start
train:
(1.8699853360885754, 0.99972)
test:
epoch: 73 accuracy 71.800%
epoch: 74 start
train:
(1.419288981705904, 0.9999)
test:
epoch: 74 accuracy 71.900%
epoch: 75 start
train:
(1.6121419850969687, 0.99986)
test:
epoch: 75 accuracy 71.380%
epoch: 76 start
train:
(1.4867232186952606, 0.99988)
test:
epoch: 76 accuracy 71.400%
epoch: 77 start
train:
(1.5225974707864225, 0.99988)
test:
epoch: 77 accuracy 72.400%
epoch: 78 start
train:
(1.3753960410831496, 0.99988)
test:
epoch: 78 accuracy 72.300%
epoch: 79 start
train:
(1.327489954594057, 0.9999)
test:
epoch: 79 accuracy 72.220%
epoch: 80 start
train:
(1.2522598807699978, 0.9999)
test:
epoch: 80 accuracy 71.550%
epoch: 81 start
train:
(1.2944153606658801, 0.9999)
test:
epoch: 81 accuracy 71.840%
epoch: 82 start
train:
(1.2619029285851866, 0.9999)
test:
epoch: 82 accuracy 72.290%
epoch: 83 start
train:
(1.0771718288306147, 0.99994)
test:
epoch: 83 accuracy 72.220%
epoch: 84 start
train:
(1.07494776614476, 0.99998)
test:
epoch: 84 accuracy 72.120%
epoch: 85 start
train:
(1.1104191646445543, 0.99994)
test:
epoch: 85 accuracy 72.020%
epoch: 86 start
train:
(1.3216143798781559, 0.99974)
test:
epoch: 86 accuracy 71.690%
epoch: 87 start
train:
(1.2244411181891337, 0.99986)
test:
epoch: 87 accuracy 71.780%
epoch: 88 start
train:
(1.1190918500069529, 0.99986)
test:
epoch: 88 accuracy 72.140%
epoch: 89 start
train:
(1.0644914847216569, 0.99984)
test:
epoch: 89 accuracy 71.970%
epoch: 90 start
train:
(0.9395060451934114, 0.99998)
test:
epoch: 90 accuracy 71.880%
epoch: 91 start
train:
(0.9682983410311863, 0.99988)
test:
epoch: 91 accuracy 70.290%
epoch: 92 start
train:
(1.0572553015663289, 0.9999)
test:
epoch: 92 accuracy 71.920%
epoch: 93 start
train:
(0.8715040243696421, 0.99996)
test:
epoch: 93 accuracy 71.640%
epoch: 94 start
train:
(1.0064131934777834, 0.99988)
test:
epoch: 94 accuracy 71.420%
epoch: 95 start
train:
(0.9556005815975368, 0.9999)
test:
epoch: 95 accuracy 72.020%
epoch: 96 start
train:
(0.9015325346845202, 0.99992)
test:
epoch: 96 accuracy 72.010%
epoch: 97 start
train:
(0.7689365273108706, 0.99996)
test:
epoch: 97 accuracy 71.780%
epoch: 98 start
train:
(0.9112347372574732, 0.99994)
test:
epoch: 98 accuracy 72.070%
epoch: 99 start
train:
(0.7669686383451335, 0.99996)
test:
epoch: 99 accuracy 71.900%
epoch: 100 start
train:
(0.9537836042582057, 0.9999)
test:
epoch: 100 accuracy 72.140%
best accuracy: 72.400%
--------result report--------
best accuracy:72.400%
loss array:
756.6213,626.1517,566.0291,525.1908,494.0737,466.0748,442.6737,419.0373,397.9515,377.8287,358.4340,340.8204,325.3502,311.4644,296.4674,283.0782,269.3162,255.7901,243.4052,230.4687,215.8141,202.7409,190.1864,177.4725,163.4795,152.0375,139.3543,126.7483,114.0503,102.1375,91.6755,81.3071,70.3854,61.2038,54.0629,46.7157,38.4428,33.4763,27.7074,24.3715,20.3606,17.5841,14.5626,12.0612,11.1197,10.3450,9.0359,8.3121,7.7411,6.3140,5.4953,4.9916,4.7598,4.0514,4.2802,3.8501,3.2563,3.0012,3.1681,3.0911,2.7845,2.6207,2.3434,2.5176,2.2099,2.0419,2.0055,1.7355,2.1151,1.7240,1.6719,1.5964,1.8700,1.4193,1.6121,1.4867,1.5226,1.3754,1.3275,1.2523,1.2944,1.2619,1.0772,1.0749,1.1104,1.3216,1.2244,1.1191,1.0645,0.9395,0.9683,1.0573,0.8715,1.0064,0.9556,0.9015,0.7689,0.9112,0.7670,0.9538,
accuracy array:
0.3507,0.4482,0.4679,0.4951,0.4954,0.5574,0.5267,0.5013,0.5582,0.5855,0.6105,0.5836,0.6489,0.6238,0.6365,0.6680,0.6600,0.6587,0.6668,0.6633,0.6413,0.6504,0.6647,0.6604,0.5979,0.6676,0.6512,0.6693,0.6545,0.6803,0.6942,0.6932,0.6518,0.6821,0.6961,0.6724,0.6839,0.6968,0.6718,0.7034,0.7088,0.6908,0.7046,0.7112,0.6797,0.7141,0.7132,0.7047,0.7090,0.7176,0.7056,0.6855,0.7163,0.5748,0.7138,0.7148,0.7187,0.7183,0.6895,0.7124,0.7202,0.7176,0.7206,0.7189,0.6949,0.7181,0.7167,0.7208,0.7206,0.7207,0.7178,0.7206,0.7180,0.7190,0.7138,0.7140,0.7240,0.7230,0.7222,0.7155,0.7184,0.7229,0.7222,0.7212,0.7202,0.7169,0.7178,0.7214,0.7197,0.7188,0.7029,0.7192,0.7164,0.7142,0.7202,0.7201,0.7178,0.7207,0.7190,0.7214,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:256
test batch size:256
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(400.06711626052856, 0.25888)
test:
epoch: 1 accuracy 32.320%
epoch: 2 start
train:
(343.1193149089813, 0.36052)
test:
epoch: 2 accuracy 37.400%
epoch: 3 start
train:
(315.058371424675, 0.40882)
test:
epoch: 3 accuracy 34.100%
epoch: 4 start
train:
(296.2430090904236, 0.44002)
test:
epoch: 4 accuracy 43.150%
epoch: 5 start
train:
(282.18069648742676, 0.47102)
test:
epoch: 5 accuracy 34.520%
epoch: 6 start
train:
(270.3126813173294, 0.49456)
test:
epoch: 6 accuracy 43.830%
epoch: 7 start
train:
(260.4829115867615, 0.51434)
test:
epoch: 7 accuracy 34.470%
epoch: 8 start
train:
(251.51732301712036, 0.53308)
test:
epoch: 8 accuracy 36.930%
epoch: 9 start
train:
(242.87635743618011, 0.55204)
test:
epoch: 9 accuracy 50.730%
epoch: 10 start
train:
(235.35691463947296, 0.56652)
test:
epoch: 10 accuracy 38.030%
epoch: 11 start
train:
(228.19358503818512, 0.5789)
test:
epoch: 11 accuracy 46.980%
epoch: 12 start
train:
(220.0450420975685, 0.5946)
test:
epoch: 12 accuracy 39.490%
epoch: 13 start
train:
(213.72775220870972, 0.60732)
test:
epoch: 13 accuracy 56.670%
epoch: 14 start
train:
(207.2636176943779, 0.61894)
test:
epoch: 14 accuracy 51.920%
epoch: 15 start
train:
(201.32581627368927, 0.63274)
test:
epoch: 15 accuracy 48.850%
epoch: 16 start
train:
(195.22593069076538, 0.64348)
test:
epoch: 16 accuracy 59.870%
epoch: 17 start
train:
(189.9012507200241, 0.65346)
test:
epoch: 17 accuracy 46.660%
epoch: 18 start
train:
(184.52424734830856, 0.66284)
test:
epoch: 18 accuracy 55.700%
epoch: 19 start
train:
(179.56814682483673, 0.6714)
test:
epoch: 19 accuracy 58.090%
epoch: 20 start
train:
(174.57815492153168, 0.68198)
test:
epoch: 20 accuracy 57.720%
epoch: 21 start
train:
(169.33614319562912, 0.69298)
test:
epoch: 21 accuracy 51.360%
epoch: 22 start
train:
(164.7777259349823, 0.70326)
test:
epoch: 22 accuracy 61.090%
epoch: 23 start
train:
(159.52058058977127, 0.71402)
test:
epoch: 23 accuracy 51.060%
epoch: 24 start
train:
(156.07135558128357, 0.7206)
test:
epoch: 24 accuracy 50.270%
epoch: 25 start
train:
(151.92317831516266, 0.72718)
test:
epoch: 25 accuracy 54.270%
epoch: 26 start
train:
(147.50033605098724, 0.73448)
test:
epoch: 26 accuracy 62.660%
epoch: 27 start
train:
(142.87829422950745, 0.7438)
test:
epoch: 27 accuracy 60.200%
epoch: 28 start
train:
(138.63536882400513, 0.75352)
test:
epoch: 28 accuracy 59.100%
epoch: 29 start
train:
(134.4796439409256, 0.7601)
test:
epoch: 29 accuracy 51.690%
epoch: 30 start
train:
(130.4102525115013, 0.77046)
test:
epoch: 30 accuracy 54.200%
epoch: 31 start
train:
(126.69834232330322, 0.77562)
test:
epoch: 31 accuracy 53.080%
epoch: 32 start
train:
(122.21167889237404, 0.78708)
test:
epoch: 32 accuracy 64.410%
epoch: 33 start
train:
(116.8282823562622, 0.79554)
test:
epoch: 33 accuracy 53.910%
epoch: 34 start
train:
(113.04568395018578, 0.80148)
test:
epoch: 34 accuracy 63.120%
epoch: 35 start
train:
(109.58741208910942, 0.8089)
test:
epoch: 35 accuracy 40.510%
epoch: 36 start
train:
(105.20749813318253, 0.81796)
test:
epoch: 36 accuracy 63.340%
epoch: 37 start
train:
(101.24443078041077, 0.82552)
test:
epoch: 37 accuracy 54.920%
epoch: 38 start
train:
(96.93209391832352, 0.83464)
test:
epoch: 38 accuracy 53.780%
epoch: 39 start
train:
(92.81272050738335, 0.84218)
test:
epoch: 39 accuracy 61.440%
epoch: 40 start
train:
(88.22125726938248, 0.85156)
test:
epoch: 40 accuracy 49.130%
epoch: 41 start
train:
(84.72832790017128, 0.8575)
test:
epoch: 41 accuracy 43.480%
epoch: 42 start
train:
(79.97132861614227, 0.86754)
test:
epoch: 42 accuracy 63.720%
epoch: 43 start
train:
(75.8602776825428, 0.87584)
test:
epoch: 43 accuracy 58.030%
epoch: 44 start
train:
(71.07586994767189, 0.88622)
test:
epoch: 44 accuracy 64.660%
epoch: 45 start
train:
(67.81467506289482, 0.89128)
test:
epoch: 45 accuracy 65.750%
epoch: 46 start
train:
(63.36375714838505, 0.89968)
test:
epoch: 46 accuracy 49.940%
epoch: 47 start
train:
(58.8474540412426, 0.90988)
test:
epoch: 47 accuracy 57.870%
epoch: 48 start
train:
(55.34148986637592, 0.91594)
test:
epoch: 48 accuracy 55.780%
epoch: 49 start
train:
(51.882428511977196, 0.9214)
test:
epoch: 49 accuracy 62.110%
epoch: 50 start
train:
(47.06409852206707, 0.93292)
test:
epoch: 50 accuracy 60.130%
epoch: 51 start
train:
(43.31921923160553, 0.93936)
test:
epoch: 51 accuracy 64.870%
epoch: 52 start
train:
(40.13957963883877, 0.9464)
test:
epoch: 52 accuracy 61.960%
epoch: 53 start
train:
(35.86238134652376, 0.95414)
test:
epoch: 53 accuracy 58.800%
epoch: 54 start
train:
(33.325652442872524, 0.95758)
test:
epoch: 54 accuracy 63.960%
epoch: 55 start
train:
(30.254864364862442, 0.96218)
test:
epoch: 55 accuracy 63.930%
epoch: 56 start
train:
(27.693159624934196, 0.96844)
test:
epoch: 56 accuracy 66.790%
epoch: 57 start
train:
(24.36362548172474, 0.97428)
test:
epoch: 57 accuracy 54.730%
epoch: 58 start
train:
(21.749178804457188, 0.97872)
test:
epoch: 58 accuracy 63.890%
epoch: 59 start
train:
(19.990308612585068, 0.98032)
test:
epoch: 59 accuracy 66.390%
epoch: 60 start
train:
(17.565654408186674, 0.98418)
test:
epoch: 60 accuracy 63.460%
epoch: 61 start
train:
(15.24970855563879, 0.98832)
test:
epoch: 61 accuracy 64.350%
epoch: 62 start
train:
(13.659278530627489, 0.99012)
test:
epoch: 62 accuracy 61.930%
epoch: 63 start
train:
(12.454283628612757, 0.99158)
test:
epoch: 63 accuracy 67.770%
epoch: 64 start
train:
(11.01064975745976, 0.99382)
test:
epoch: 64 accuracy 67.950%
epoch: 65 start
train:
(9.892815167084336, 0.99486)
test:
epoch: 65 accuracy 67.050%
epoch: 66 start
train:
(8.724291207268834, 0.99668)
test:
epoch: 66 accuracy 47.660%
epoch: 67 start
train:
(9.189347513020039, 0.9947)
test:
epoch: 67 accuracy 67.710%
epoch: 68 start
train:
(6.939485093578696, 0.99814)
test:
epoch: 68 accuracy 63.400%
epoch: 69 start
train:
(6.551137624308467, 0.99816)
test:
epoch: 69 accuracy 62.850%
epoch: 70 start
train:
(6.212482846342027, 0.99802)
test:
epoch: 70 accuracy 68.760%
epoch: 71 start
train:
(5.361541533842683, 0.9986)
test:
epoch: 71 accuracy 69.180%
epoch: 72 start
train:
(4.896827505901456, 0.99898)
test:
epoch: 72 accuracy 65.690%
epoch: 73 start
train:
(4.4510957011953, 0.99934)
test:
epoch: 73 accuracy 50.080%
epoch: 74 start
train:
(6.152660959400237, 0.99668)
test:
epoch: 74 accuracy 68.090%
epoch: 75 start
train:
(4.102192289195955, 0.99934)
test:
epoch: 75 accuracy 66.130%
epoch: 76 start
train:
(3.468895842321217, 0.9997)
test:
epoch: 76 accuracy 69.140%
epoch: 77 start
train:
(3.4853789266198874, 0.9993)
test:
epoch: 77 accuracy 68.330%
epoch: 78 start
train:
(3.2854257952421904, 0.99966)
test:
epoch: 78 accuracy 68.550%
epoch: 79 start
train:
(2.856927728280425, 0.99974)
test:
epoch: 79 accuracy 68.010%
epoch: 80 start
train:
(2.8887311331927776, 0.99966)
test:
epoch: 80 accuracy 62.770%
epoch: 81 start
train:
(2.9180214391089976, 0.99942)
test:
epoch: 81 accuracy 66.380%
epoch: 82 start
train:
(2.709244092926383, 0.9997)
test:
epoch: 82 accuracy 69.170%
epoch: 83 start
train:
(2.4876278289593756, 0.99972)
test:
epoch: 83 accuracy 68.760%
epoch: 84 start
train:
(2.340533602051437, 0.99982)
test:
epoch: 84 accuracy 61.980%
epoch: 85 start
train:
(2.508210754022002, 0.9995)
test:
epoch: 85 accuracy 69.170%
epoch: 86 start
train:
(2.1138355839066207, 0.99986)
test:
epoch: 86 accuracy 67.760%
epoch: 87 start
train:
(1.980650338344276, 0.99988)
test:
epoch: 87 accuracy 68.910%
epoch: 88 start
train:
(1.8605117346160114, 0.9999)
test:
epoch: 88 accuracy 68.600%
epoch: 89 start
train:
(1.8442798536270857, 0.99988)
test:
epoch: 89 accuracy 67.640%
epoch: 90 start
train:
(1.8286710162647069, 0.99984)
test:
epoch: 90 accuracy 66.500%
epoch: 91 start
train:
(1.8156694048084319, 0.99988)
test:
epoch: 91 accuracy 69.030%
epoch: 92 start
train:
(1.6133217504248023, 0.99996)
test:
epoch: 92 accuracy 69.330%
epoch: 93 start
train:
(1.517588515765965, 0.99996)
test:
epoch: 93 accuracy 69.410%
epoch: 94 start
train:
(1.5245234118774533, 0.99998)
test:
epoch: 94 accuracy 68.680%
epoch: 95 start
train:
(1.6842127377167344, 0.99968)
test:
epoch: 95 accuracy 68.150%
epoch: 96 start
train:
(1.3592937141656876, 0.99998)
test:
epoch: 96 accuracy 69.390%
epoch: 97 start
train:
(1.4445629159454256, 0.99988)
test:
epoch: 97 accuracy 68.980%
epoch: 98 start
train:
(1.3747423512395471, 0.99988)
test:
epoch: 98 accuracy 68.270%
epoch: 99 start
train:
(1.4172865515574813, 0.99988)
test:
epoch: 99 accuracy 68.350%
epoch: 100 start
train:
(1.2378791377414018, 0.99998)
test:
epoch: 100 accuracy 68.500%
best accuracy: 69.410%
--------result report--------
best accuracy:69.410%
loss array:
400.0671,343.1193,315.0584,296.2430,282.1807,270.3127,260.4829,251.5173,242.8764,235.3569,228.1936,220.0450,213.7278,207.2636,201.3258,195.2259,189.9013,184.5242,179.5681,174.5782,169.3361,164.7777,159.5206,156.0714,151.9232,147.5003,142.8783,138.6354,134.4796,130.4103,126.6983,122.2117,116.8283,113.0457,109.5874,105.2075,101.2444,96.9321,92.8127,88.2213,84.7283,79.9713,75.8603,71.0759,67.8147,63.3638,58.8475,55.3415,51.8824,47.0641,43.3192,40.1396,35.8624,33.3257,30.2549,27.6932,24.3636,21.7492,19.9903,17.5657,15.2497,13.6593,12.4543,11.0106,9.8928,8.7243,9.1893,6.9395,6.5511,6.2125,5.3615,4.8968,4.4511,6.1527,4.1022,3.4689,3.4854,3.2854,2.8569,2.8887,2.9180,2.7092,2.4876,2.3405,2.5082,2.1138,1.9807,1.8605,1.8443,1.8287,1.8157,1.6133,1.5176,1.5245,1.6842,1.3593,1.4446,1.3747,1.4173,1.2379,
accuracy array:
0.3232,0.3740,0.3410,0.4315,0.3452,0.4383,0.3447,0.3693,0.5073,0.3803,0.4698,0.3949,0.5667,0.5192,0.4885,0.5987,0.4666,0.5570,0.5809,0.5772,0.5136,0.6109,0.5106,0.5027,0.5427,0.6266,0.6020,0.5910,0.5169,0.5420,0.5308,0.6441,0.5391,0.6312,0.4051,0.6334,0.5492,0.5378,0.6144,0.4913,0.4348,0.6372,0.5803,0.6466,0.6575,0.4994,0.5787,0.5578,0.6211,0.6013,0.6487,0.6196,0.5880,0.6396,0.6393,0.6679,0.5473,0.6389,0.6639,0.6346,0.6435,0.6193,0.6777,0.6795,0.6705,0.4766,0.6771,0.6340,0.6285,0.6876,0.6918,0.6569,0.5008,0.6809,0.6613,0.6914,0.6833,0.6855,0.6801,0.6277,0.6638,0.6917,0.6876,0.6198,0.6917,0.6776,0.6891,0.6860,0.6764,0.6650,0.6903,0.6933,0.6941,0.6868,0.6815,0.6939,0.6898,0.6827,0.6835,0.6850,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:100
train batch size:512
test batch size:512
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(215.29911947250366, 0.20356)
test:
epoch: 1 accuracy 25.800%
epoch: 2 start
train:
(194.42408418655396, 0.29038)
test:
epoch: 2 accuracy 31.650%
epoch: 3 start
train:
(180.09761691093445, 0.33656)
test:
epoch: 3 accuracy 36.030%
epoch: 4 start
train:
(168.7870672941208, 0.37014)
test:
epoch: 4 accuracy 38.810%
epoch: 5 start
train:
(160.40372717380524, 0.39756)
test:
epoch: 5 accuracy 40.150%
epoch: 6 start
train:
(154.41951441764832, 0.42096)
test:
epoch: 6 accuracy 42.050%
epoch: 7 start
train:
(149.68647027015686, 0.43718)
test:
epoch: 7 accuracy 44.010%
epoch: 8 start
train:
(145.78765416145325, 0.45126)
test:
epoch: 8 accuracy 44.640%
epoch: 9 start
train:
(142.16310584545135, 0.46688)
test:
epoch: 9 accuracy 44.140%
epoch: 10 start
train:
(138.79884803295135, 0.4791)
test:
epoch: 10 accuracy 47.520%
epoch: 11 start
train:
(135.73249793052673, 0.49368)
test:
epoch: 11 accuracy 47.050%
epoch: 12 start
train:
(132.69774293899536, 0.50534)
test:
epoch: 12 accuracy 50.030%
epoch: 13 start
train:
(129.83782625198364, 0.51742)
test:
epoch: 13 accuracy 49.850%
epoch: 14 start
train:
(127.37378799915314, 0.53)
test:
epoch: 14 accuracy 48.810%
epoch: 15 start
train:
(124.84151148796082, 0.5375)
test:
epoch: 15 accuracy 52.350%
epoch: 16 start
train:
(122.4406179189682, 0.54702)
test:
epoch: 16 accuracy 50.470%
epoch: 17 start
train:
(119.99695670604706, 0.55632)
test:
epoch: 17 accuracy 51.020%
epoch: 18 start
train:
(118.07604777812958, 0.56364)
test:
epoch: 18 accuracy 51.860%
epoch: 19 start
train:
(115.98612630367279, 0.57062)
test:
epoch: 19 accuracy 54.640%
epoch: 20 start
train:
(114.1467113494873, 0.5784)
test:
epoch: 20 accuracy 54.620%
epoch: 21 start
train:
(112.16308879852295, 0.58742)
test:
epoch: 21 accuracy 51.980%
epoch: 22 start
train:
(109.967524766922, 0.59538)
test:
epoch: 22 accuracy 36.420%
epoch: 23 start
train:
(107.92675191164017, 0.6048)
test:
epoch: 23 accuracy 57.300%
epoch: 24 start
train:
(106.38465029001236, 0.60994)
test:
epoch: 24 accuracy 54.230%
epoch: 25 start
train:
(104.59969019889832, 0.61826)
test:
epoch: 25 accuracy 52.500%
epoch: 26 start
train:
(102.71512424945831, 0.62586)
test:
epoch: 26 accuracy 52.200%
epoch: 27 start
train:
(101.3006774187088, 0.6298)
test:
epoch: 27 accuracy 59.390%
epoch: 28 start
train:
(99.37277978658676, 0.6374)
test:
epoch: 28 accuracy 60.390%
epoch: 29 start
train:
(98.05501174926758, 0.64118)
test:
epoch: 29 accuracy 59.860%
epoch: 30 start
train:
(96.23863953351974, 0.64942)
test:
epoch: 30 accuracy 53.120%
epoch: 31 start
train:
(94.68071019649506, 0.65546)
test:
epoch: 31 accuracy 58.140%
epoch: 32 start
train:
(93.02145195007324, 0.66164)
test:
epoch: 32 accuracy 56.930%
epoch: 33 start
train:
(91.81287401914597, 0.66718)
test:
epoch: 33 accuracy 59.820%
epoch: 34 start
train:
(89.82423448562622, 0.67416)
test:
epoch: 34 accuracy 54.810%
epoch: 35 start
train:
(88.63934832811356, 0.67746)
test:
epoch: 35 accuracy 61.270%
epoch: 36 start
train:
(87.39750701189041, 0.68268)
test:
epoch: 36 accuracy 60.870%
epoch: 37 start
train:
(85.81197386980057, 0.69088)
test:
epoch: 37 accuracy 60.600%
epoch: 38 start
train:
(84.73660433292389, 0.69338)
test:
epoch: 38 accuracy 53.860%
epoch: 39 start
train:
(82.91159409284592, 0.70062)
test:
epoch: 39 accuracy 61.260%
epoch: 40 start
train:
(81.97390228509903, 0.70392)
test:
epoch: 40 accuracy 53.320%
epoch: 41 start
train:
(80.2232335805893, 0.70862)
test:
epoch: 41 accuracy 64.550%
epoch: 42 start
train:
(78.82643580436707, 0.71682)
test:
epoch: 42 accuracy 58.740%
epoch: 43 start
train:
(77.86338990926743, 0.72104)
test:
epoch: 43 accuracy 58.050%
epoch: 44 start
train:
(76.67819058895111, 0.72588)
test:
epoch: 44 accuracy 61.330%
epoch: 45 start
train:
(75.3877204656601, 0.72798)
test:
epoch: 45 accuracy 59.800%
epoch: 46 start
train:
(74.00761759281158, 0.73378)
test:
epoch: 46 accuracy 62.260%
epoch: 47 start
train:
(72.9039067029953, 0.73838)
test:
epoch: 47 accuracy 53.610%
epoch: 48 start
train:
(71.74449497461319, 0.74332)
test:
epoch: 48 accuracy 56.860%
epoch: 49 start
train:
(70.11631673574448, 0.74994)
test:
epoch: 49 accuracy 61.300%
epoch: 50 start
train:
(69.19097715616226, 0.75398)
test:
epoch: 50 accuracy 52.440%
epoch: 51 start
train:
(67.85274851322174, 0.75896)
test:
epoch: 51 accuracy 64.470%
epoch: 52 start
train:
(66.67607200145721, 0.76434)
test:
epoch: 52 accuracy 62.990%
epoch: 53 start
train:
(65.68665552139282, 0.7666)
test:
epoch: 53 accuracy 64.540%
epoch: 54 start
train:
(64.08659702539444, 0.77384)
test:
epoch: 54 accuracy 59.850%
epoch: 55 start
train:
(63.058896124362946, 0.77632)
test:
epoch: 55 accuracy 64.660%
epoch: 56 start
train:
(61.83358919620514, 0.78306)
test:
epoch: 56 accuracy 54.530%
epoch: 57 start
train:
(60.7563356757164, 0.7877)
test:
epoch: 57 accuracy 66.820%
epoch: 58 start
train:
(59.48637956380844, 0.79204)
test:
epoch: 58 accuracy 65.010%
epoch: 59 start
train:
(58.25693076848984, 0.79746)
test:
epoch: 59 accuracy 64.370%
epoch: 60 start
train:
(56.96232050657272, 0.80268)
test:
epoch: 60 accuracy 60.530%
epoch: 61 start
train:
(56.280825555324554, 0.80488)
test:
epoch: 61 accuracy 57.840%
epoch: 62 start
train:
(54.72459629178047, 0.81136)
test:
epoch: 62 accuracy 62.440%
epoch: 63 start
train:
(53.48458072543144, 0.81578)
test:
epoch: 63 accuracy 66.540%
epoch: 64 start
train:
(52.182239294052124, 0.82206)
test:
epoch: 64 accuracy 65.190%
epoch: 65 start
train:
(51.00824072957039, 0.8258)
test:
epoch: 65 accuracy 67.370%
epoch: 66 start
train:
(50.09007838368416, 0.82754)
test:
epoch: 66 accuracy 64.060%
epoch: 67 start
train:
(48.598402976989746, 0.8369)
test:
epoch: 67 accuracy 55.120%
epoch: 68 start
train:
(47.33603236079216, 0.83972)
test:
epoch: 68 accuracy 62.500%
epoch: 69 start
train:
(46.24530813097954, 0.8454)
test:
epoch: 69 accuracy 61.750%
epoch: 70 start
train:
(44.90822014212608, 0.8493)
test:
epoch: 70 accuracy 63.270%
epoch: 71 start
train:
(44.042414486408234, 0.8537)
test:
epoch: 71 accuracy 65.830%
epoch: 72 start
train:
(42.1898595392704, 0.86216)
test:
epoch: 72 accuracy 63.870%
epoch: 73 start
train:
(41.5417902469635, 0.863)
test:
epoch: 73 accuracy 63.870%
epoch: 74 start
train:
(39.85534134507179, 0.87128)
test:
epoch: 74 accuracy 66.100%
epoch: 75 start
train:
(38.88663497567177, 0.87594)
test:
epoch: 75 accuracy 66.810%
epoch: 76 start
train:
(37.32938954234123, 0.88154)
test:
epoch: 76 accuracy 65.390%
epoch: 77 start
train:
(36.496265560388565, 0.88446)
test:
epoch: 77 accuracy 65.220%
epoch: 78 start
train:
(34.87601837515831, 0.89158)
test:
epoch: 78 accuracy 66.190%
epoch: 79 start
train:
(33.361087411642075, 0.89724)
test:
epoch: 79 accuracy 56.710%
epoch: 80 start
train:
(32.66950881481171, 0.8996)
test:
epoch: 80 accuracy 62.100%
epoch: 81 start
train:
(31.399329215288162, 0.90464)
test:
epoch: 81 accuracy 66.510%
epoch: 82 start
train:
(30.157054230570793, 0.9101)
test:
epoch: 82 accuracy 65.780%
epoch: 83 start
train:
(29.20188996195793, 0.913)
test:
epoch: 83 accuracy 65.350%
epoch: 84 start
train:
(28.07621504366398, 0.91816)
test:
epoch: 84 accuracy 66.610%
epoch: 85 start
train:
(26.851683855056763, 0.9221)
test:
epoch: 85 accuracy 67.720%
epoch: 86 start
train:
(25.682600378990173, 0.9278)
test:
epoch: 86 accuracy 65.490%
epoch: 87 start
train:
(24.239226818084717, 0.9334)
test:
epoch: 87 accuracy 65.230%
epoch: 88 start
train:
(23.1670174151659, 0.9371)
test:
epoch: 88 accuracy 66.550%
epoch: 89 start
train:
(22.011658012866974, 0.94292)
test:
epoch: 89 accuracy 63.020%
epoch: 90 start
train:
(21.29241332411766, 0.94488)
test:
epoch: 90 accuracy 66.680%
epoch: 91 start
train:
(20.224969908595085, 0.94858)
test:
epoch: 91 accuracy 65.610%
epoch: 92 start
train:
(19.079633802175522, 0.9529)
test:
epoch: 92 accuracy 65.060%
epoch: 93 start
train:
(17.661130532622337, 0.95852)
test:
epoch: 93 accuracy 65.640%
epoch: 94 start
train:
(17.197376742959023, 0.95902)
test:
epoch: 94 accuracy 61.680%
epoch: 95 start
train:
(15.981868050992489, 0.9646)
test:
epoch: 95 accuracy 66.360%
epoch: 96 start
train:
(15.746334232389927, 0.9646)
test:
epoch: 96 accuracy 66.980%
epoch: 97 start
train:
(14.252664104104042, 0.97108)
test:
epoch: 97 accuracy 66.820%
epoch: 98 start
train:
(13.258388139307499, 0.9747)
test:
epoch: 98 accuracy 66.880%
epoch: 99 start
train:
(12.96216257661581, 0.9745)
test:
epoch: 99 accuracy 67.100%
epoch: 100 start
train:
(11.862829208374023, 0.97884)
test:
epoch: 100 accuracy 67.170%
best accuracy: 67.720%
--------result report--------
best accuracy:67.720%
loss array:
215.2991,194.4241,180.0976,168.7871,160.4037,154.4195,149.6865,145.7877,142.1631,138.7988,135.7325,132.6977,129.8378,127.3738,124.8415,122.4406,119.9970,118.0760,115.9861,114.1467,112.1631,109.9675,107.9268,106.3847,104.5997,102.7151,101.3007,99.3728,98.0550,96.2386,94.6807,93.0215,91.8129,89.8242,88.6393,87.3975,85.8120,84.7366,82.9116,81.9739,80.2232,78.8264,77.8634,76.6782,75.3877,74.0076,72.9039,71.7445,70.1163,69.1910,67.8527,66.6761,65.6867,64.0866,63.0589,61.8336,60.7563,59.4864,58.2569,56.9623,56.2808,54.7246,53.4846,52.1822,51.0082,50.0901,48.5984,47.3360,46.2453,44.9082,44.0424,42.1899,41.5418,39.8553,38.8866,37.3294,36.4963,34.8760,33.3611,32.6695,31.3993,30.1571,29.2019,28.0762,26.8517,25.6826,24.2392,23.1670,22.0117,21.2924,20.2250,19.0796,17.6611,17.1974,15.9819,15.7463,14.2527,13.2584,12.9622,11.8628,
accuracy array:
0.2580,0.3165,0.3603,0.3881,0.4015,0.4205,0.4401,0.4464,0.4414,0.4752,0.4705,0.5003,0.4985,0.4881,0.5235,0.5047,0.5102,0.5186,0.5464,0.5462,0.5198,0.3642,0.5730,0.5423,0.5250,0.5220,0.5939,0.6039,0.5986,0.5312,0.5814,0.5693,0.5982,0.5481,0.6127,0.6087,0.6060,0.5386,0.6126,0.5332,0.6455,0.5874,0.5805,0.6133,0.5980,0.6226,0.5361,0.5686,0.6130,0.5244,0.6447,0.6299,0.6454,0.5985,0.6466,0.5453,0.6682,0.6501,0.6437,0.6053,0.5784,0.6244,0.6654,0.6519,0.6737,0.6406,0.5512,0.6250,0.6175,0.6327,0.6583,0.6387,0.6387,0.6610,0.6681,0.6539,0.6522,0.6619,0.5671,0.6210,0.6651,0.6578,0.6535,0.6661,0.6772,0.6549,0.6523,0.6655,0.6302,0.6668,0.6561,0.6506,0.6564,0.6168,0.6636,0.6698,0.6682,0.6688,0.6710,0.6717,
-------------------------------
--------parameter report--------
batchSize:16    accuracy:0.837%
batchSize:32    accuracy:0.800%
batchSize:64    accuracy:0.758%
batchSize:128    accuracy:0.724%
batchSize:256    accuracy:0.694%
batchSize:512    accuracy:0.677%
```