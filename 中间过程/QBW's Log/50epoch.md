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
EPOCH=50
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
epoch:50
train batch size:16
test batch size:16
model:resnet18
optimizer:SGD
-------------------------------
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data/cifar-10-python.tar.gz
```

170499072/? [00:06<00:00, 32140806.81it/s]

```
Extracting ../data/cifar-10-python.tar.gz to ../data
Files already downloaded and verified
epoch: 1 start
train:
(4953.020388185978, 0.41218)
test:
epoch: 1 accuracy 52.840%
epoch: 2 start
train:
(3702.5748108923435, 0.574)
test:
epoch: 2 accuracy 61.580%
epoch: 3 start
train:
(3039.7282483279705, 0.65444)
test:
epoch: 3 accuracy 67.330%
epoch: 4 start
train:
(2563.6640931516886, 0.71076)
test:
epoch: 4 accuracy 70.180%
epoch: 5 start
train:
(2223.460085645318, 0.74968)
test:
epoch: 5 accuracy 74.710%
epoch: 6 start
train:
(1921.3995834738016, 0.78628)
test:
epoch: 6 accuracy 76.280%
epoch: 7 start
train:
(1664.1582936570048, 0.81642)
test:
epoch: 7 accuracy 76.680%
epoch: 8 start
train:
(1460.0342792831361, 0.83932)
test:
epoch: 8 accuracy 77.310%
epoch: 9 start
train:
(1258.3946554977447, 0.8624)
test:
epoch: 9 accuracy 78.140%
epoch: 10 start
train:
(1069.0180792734027, 0.88246)
test:
epoch: 10 accuracy 78.830%
epoch: 11 start
train:
(922.370321623981, 0.9)
test:
epoch: 11 accuracy 79.460%
epoch: 12 start
train:
(790.1156281260774, 0.91336)
test:
epoch: 12 accuracy 78.600%
epoch: 13 start
train:
(685.2728547528386, 0.9255)
test:
epoch: 13 accuracy 78.900%
epoch: 14 start
train:
(581.1779159749858, 0.93692)
test:
epoch: 14 accuracy 76.530%
epoch: 15 start
train:
(511.50701808393933, 0.9455)
test:
epoch: 15 accuracy 80.170%
epoch: 16 start
train:
(444.9801548917312, 0.95326)
test:
epoch: 16 accuracy 78.590%
epoch: 17 start
train:
(371.7159303963417, 0.95948)
test:
epoch: 17 accuracy 79.620%
epoch: 18 start
train:
(339.4125192766078, 0.9637)
test:
epoch: 18 accuracy 79.960%
epoch: 19 start
train:
(284.68535618443275, 0.97002)
test:
epoch: 19 accuracy 79.940%
epoch: 20 start
train:
(270.2284191357321, 0.97112)
test:
epoch: 20 accuracy 80.070%
epoch: 21 start
train:
(235.55875408946304, 0.97494)
test:
epoch: 21 accuracy 80.570%
epoch: 22 start
train:
(209.51703842988354, 0.97844)
test:
epoch: 22 accuracy 80.240%
epoch: 23 start
train:
(186.02899090907886, 0.97982)
test:
epoch: 23 accuracy 80.970%
epoch: 24 start
train:
(173.97635700920364, 0.98112)
test:
epoch: 24 accuracy 81.230%
epoch: 25 start
train:
(161.9506945602916, 0.98346)
test:
epoch: 25 accuracy 81.170%
epoch: 26 start
train:
(148.96372265512764, 0.9841)
test:
epoch: 26 accuracy 80.980%
epoch: 27 start
train:
(130.98455836022913, 0.98672)
test:
epoch: 27 accuracy 81.850%
epoch: 28 start
train:
(119.97274852315604, 0.98732)
test:
epoch: 28 accuracy 81.670%
epoch: 29 start
train:
(108.56736199256557, 0.9891)
test:
epoch: 29 accuracy 80.700%
epoch: 30 start
train:
(98.2320355700358, 0.98994)
test:
epoch: 30 accuracy 80.480%
epoch: 31 start
train:
(96.79461789560446, 0.98978)
test:
epoch: 31 accuracy 81.390%
epoch: 32 start
train:
(82.66562763466936, 0.992)
test:
epoch: 32 accuracy 81.290%
epoch: 33 start
train:
(79.32336942908296, 0.99194)
test:
epoch: 33 accuracy 81.350%
epoch: 34 start
train:
(78.02559434264913, 0.99218)
test:
epoch: 34 accuracy 80.980%
epoch: 35 start
train:
(78.36878841888392, 0.99196)
test:
epoch: 35 accuracy 81.030%
epoch: 36 start
train:
(79.18082801081982, 0.99176)
test:
epoch: 36 accuracy 81.340%
epoch: 37 start
train:
(67.05702850635862, 0.99374)
test:
epoch: 37 accuracy 81.410%
epoch: 38 start
train:
(61.24777470415211, 0.99386)
test:
epoch: 38 accuracy 81.240%
epoch: 39 start
train:
(58.92901666287071, 0.99432)
test:
epoch: 39 accuracy 81.950%
epoch: 40 start
train:
(54.40013439231916, 0.99458)
test:
epoch: 40 accuracy 81.850%
epoch: 41 start
train:
(60.915133784223144, 0.99344)
test:
epoch: 41 accuracy 81.810%
epoch: 42 start
train:
(53.66567293751723, 0.995)
test:
epoch: 42 accuracy 81.210%
epoch: 43 start
train:
(51.16242569051792, 0.99512)
test:
epoch: 43 accuracy 81.720%
epoch: 44 start
train:
(46.979205948688104, 0.99544)
test:
epoch: 44 accuracy 81.280%
epoch: 45 start
train:
(45.184690092879464, 0.99572)
test:
epoch: 45 accuracy 81.580%
epoch: 46 start
train:
(44.80510351774319, 0.99562)
test:
epoch: 46 accuracy 81.140%
epoch: 47 start
train:
(43.39022809059861, 0.9956)
test:
epoch: 47 accuracy 82.170%
epoch: 48 start
train:
(40.54680624546381, 0.9958)
test:
epoch: 48 accuracy 81.640%
epoch: 49 start
train:
(38.45034690374905, 0.9963)
test:
epoch: 49 accuracy 81.860%
epoch: 50 start
train:
(35.36515319159844, 0.99676)
test:
epoch: 50 accuracy 81.700%
best accuracy: 82.170%
--------result report--------
best accuracy:82.170%
loss array:
4953.0204,3702.5748,3039.7282,2563.6641,2223.4601,1921.3996,1664.1583,1460.0343,1258.3947,1069.0181,922.3703,790.1156,685.2729,581.1779,511.5070,444.9802,371.7159,339.4125,284.6854,270.2284,235.5588,209.5170,186.0290,173.9764,161.9507,148.9637,130.9846,119.9727,108.5674,98.2320,96.7946,82.6656,79.3234,78.0256,78.3688,79.1808,67.0570,61.2478,58.9290,54.4001,60.9151,53.6657,51.1624,46.9792,45.1847,44.8051,43.3902,40.5468,38.4503,35.3652,
accuracy array:
0.5284,0.6158,0.6733,0.7018,0.7471,0.7628,0.7668,0.7731,0.7814,0.7883,0.7946,0.7860,0.7890,0.7653,0.8017,0.7859,0.7962,0.7996,0.7994,0.8007,0.8057,0.8024,0.8097,0.8123,0.8117,0.8098,0.8185,0.8167,0.8070,0.8048,0.8139,0.8129,0.8135,0.8098,0.8103,0.8134,0.8141,0.8124,0.8195,0.8185,0.8181,0.8121,0.8172,0.8128,0.8158,0.8114,0.8217,0.8164,0.8186,0.8170,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:50
train batch size:32
test batch size:32
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(2590.222452044487, 0.38426)
test:
epoch: 1 accuracy 45.760%
epoch: 2 start
train:
(2025.6651899814606, 0.52974)
test:
epoch: 2 accuracy 51.250%
epoch: 3 start
train:
(1734.8704310059547, 0.60184)
test:
epoch: 3 accuracy 61.290%
epoch: 4 start
train:
(1501.4597491025925, 0.65608)
test:
epoch: 4 accuracy 61.500%
epoch: 5 start
train:
(1332.9417690634727, 0.69722)
test:
epoch: 5 accuracy 67.000%
epoch: 6 start
train:
(1195.0263379514217, 0.73026)
test:
epoch: 6 accuracy 66.810%
epoch: 7 start
train:
(1070.275891751051, 0.76074)
test:
epoch: 7 accuracy 71.790%
epoch: 8 start
train:
(960.5827174633741, 0.78582)
test:
epoch: 8 accuracy 71.610%
epoch: 9 start
train:
(851.5714209377766, 0.81008)
test:
epoch: 9 accuracy 70.510%
epoch: 10 start
train:
(750.3914395570755, 0.83326)
test:
epoch: 10 accuracy 75.750%
epoch: 11 start
train:
(665.5195095911622, 0.85358)
test:
epoch: 11 accuracy 73.790%
epoch: 12 start
train:
(584.2203751131892, 0.87252)
test:
epoch: 12 accuracy 67.420%
epoch: 13 start
train:
(505.9128138720989, 0.89038)
test:
epoch: 13 accuracy 75.070%
epoch: 14 start
train:
(434.8397917859256, 0.90694)
test:
epoch: 14 accuracy 73.600%
epoch: 15 start
train:
(373.11408723145723, 0.92122)
test:
epoch: 15 accuracy 75.860%
epoch: 16 start
train:
(321.4977234452963, 0.93142)
test:
epoch: 16 accuracy 73.810%
epoch: 17 start
train:
(272.5033566392958, 0.94254)
test:
epoch: 17 accuracy 75.880%
epoch: 18 start
train:
(233.24001355282962, 0.95158)
test:
epoch: 18 accuracy 74.480%
epoch: 19 start
train:
(190.89394827838987, 0.96074)
test:
epoch: 19 accuracy 74.650%
epoch: 20 start
train:
(172.83763330616057, 0.96332)
test:
epoch: 20 accuracy 76.940%
epoch: 21 start
train:
(151.11369728017598, 0.97004)
test:
epoch: 21 accuracy 76.810%
epoch: 22 start
train:
(128.1190370679833, 0.9743)
test:
epoch: 22 accuracy 75.810%
epoch: 23 start
train:
(110.97784209111705, 0.97792)
test:
epoch: 23 accuracy 76.450%
epoch: 24 start
train:
(95.35588819114491, 0.98108)
test:
epoch: 24 accuracy 77.770%
epoch: 25 start
train:
(89.03270327299833, 0.98186)
test:
epoch: 25 accuracy 77.370%
epoch: 26 start
train:
(81.96316916344222, 0.98392)
test:
epoch: 26 accuracy 76.860%
epoch: 27 start
train:
(73.05804994213395, 0.98482)
test:
epoch: 27 accuracy 76.370%
epoch: 28 start
train:
(63.48610810912214, 0.988)
test:
epoch: 28 accuracy 77.900%
epoch: 29 start
train:
(54.971677188412286, 0.9897)
test:
epoch: 29 accuracy 78.220%
epoch: 30 start
train:
(54.27743763587205, 0.9894)
test:
epoch: 30 accuracy 77.580%
epoch: 31 start
train:
(49.904704299231526, 0.99068)
test:
epoch: 31 accuracy 78.540%
epoch: 32 start
train:
(41.76290879014414, 0.9922)
test:
epoch: 32 accuracy 77.920%
epoch: 33 start
train:
(41.33572558875312, 0.99214)
test:
epoch: 33 accuracy 77.910%
epoch: 34 start
train:
(42.86214778551948, 0.99198)
test:
epoch: 34 accuracy 77.670%
epoch: 35 start
train:
(39.28637173800962, 0.9927)
test:
epoch: 35 accuracy 78.300%
epoch: 36 start
train:
(32.08222300777561, 0.99398)
test:
epoch: 36 accuracy 78.360%
epoch: 37 start
train:
(37.067737216100795, 0.99294)
test:
epoch: 37 accuracy 78.660%
epoch: 38 start
train:
(27.300935710896738, 0.995)
test:
epoch: 38 accuracy 78.850%
epoch: 39 start
train:
(25.517807067881222, 0.99558)
test:
epoch: 39 accuracy 78.620%
epoch: 40 start
train:
(26.11051811586367, 0.9954)
test:
epoch: 40 accuracy 78.280%
epoch: 41 start
train:
(22.603558516711928, 0.99628)
test:
epoch: 41 accuracy 77.790%
epoch: 42 start
train:
(25.26344618794974, 0.99544)
test:
epoch: 42 accuracy 77.620%
epoch: 43 start
train:
(22.176400598182227, 0.99598)
test:
epoch: 43 accuracy 78.830%
epoch: 44 start
train:
(24.50093790848041, 0.99522)
test:
epoch: 44 accuracy 78.100%
epoch: 45 start
train:
(18.163795748812845, 0.9973)
test:
epoch: 45 accuracy 78.880%
epoch: 46 start
train:
(19.36822090978967, 0.99688)
test:
epoch: 46 accuracy 79.120%
epoch: 47 start
train:
(19.408092615703936, 0.99654)
test:
epoch: 47 accuracy 78.720%
epoch: 48 start
train:
(19.12410774921591, 0.99622)
test:
epoch: 48 accuracy 77.410%
epoch: 49 start
train:
(17.67883340708795, 0.9969)
test:
epoch: 49 accuracy 78.520%
epoch: 50 start
train:
(18.44579045955834, 0.99682)
test:
epoch: 50 accuracy 79.070%
best accuracy: 79.120%
--------result report--------
best accuracy:79.120%
loss array:
2590.2225,2025.6652,1734.8704,1501.4597,1332.9418,1195.0263,1070.2759,960.5827,851.5714,750.3914,665.5195,584.2204,505.9128,434.8398,373.1141,321.4977,272.5034,233.2400,190.8939,172.8376,151.1137,128.1190,110.9778,95.3559,89.0327,81.9632,73.0580,63.4861,54.9717,54.2774,49.9047,41.7629,41.3357,42.8621,39.2864,32.0822,37.0677,27.3009,25.5178,26.1105,22.6036,25.2634,22.1764,24.5009,18.1638,19.3682,19.4081,19.1241,17.6788,18.4458,
accuracy array:
0.4576,0.5125,0.6129,0.6150,0.6700,0.6681,0.7179,0.7161,0.7051,0.7575,0.7379,0.6742,0.7507,0.7360,0.7586,0.7381,0.7588,0.7448,0.7465,0.7694,0.7681,0.7581,0.7645,0.7777,0.7737,0.7686,0.7637,0.7790,0.7822,0.7758,0.7854,0.7792,0.7791,0.7767,0.7830,0.7836,0.7866,0.7885,0.7862,0.7828,0.7779,0.7762,0.7883,0.7810,0.7888,0.7912,0.7872,0.7741,0.7852,0.7907,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:50
train batch size:64
test batch size:64
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(1398.8535752296448, 0.3413)
test:
epoch: 1 accuracy 38.970%
epoch: 2 start
train:
(1132.4188570976257, 0.46878)
test:
epoch: 2 accuracy 41.230%
epoch: 3 start
train:
(1010.0802658200264, 0.52982)
test:
epoch: 3 accuracy 50.100%
epoch: 4 start
train:
(912.3487519025803, 0.58062)
test:
epoch: 4 accuracy 54.710%
epoch: 5 start
train:
(829.8427313566208, 0.61914)
test:
epoch: 5 accuracy 45.940%
epoch: 6 start
train:
(760.42197650671, 0.65372)
test:
epoch: 6 accuracy 53.220%
epoch: 7 start
train:
(699.112896323204, 0.68372)
test:
epoch: 7 accuracy 60.040%
epoch: 8 start
train:
(647.285721719265, 0.70976)
test:
epoch: 8 accuracy 61.720%
epoch: 9 start
train:
(600.3490529954433, 0.7286)
test:
epoch: 9 accuracy 59.720%
epoch: 10 start
train:
(559.993859231472, 0.74902)
test:
epoch: 10 accuracy 60.130%
epoch: 11 start
train:
(515.2444635331631, 0.76742)
test:
epoch: 11 accuracy 54.060%
epoch: 12 start
train:
(479.2447063624859, 0.785)
test:
epoch: 12 accuracy 64.220%
epoch: 13 start
train:
(440.630118265748, 0.8058)
test:
epoch: 13 accuracy 68.440%
epoch: 14 start
train:
(402.0613238066435, 0.8227)
test:
epoch: 14 accuracy 61.740%
epoch: 15 start
train:
(367.188589155674, 0.8379)
test:
epoch: 15 accuracy 60.130%
epoch: 16 start
train:
(335.0370138734579, 0.85428)
test:
epoch: 16 accuracy 68.730%
epoch: 17 start
train:
(293.61622509360313, 0.8723)
test:
epoch: 17 accuracy 57.710%
epoch: 18 start
train:
(264.61614026129246, 0.88798)
test:
epoch: 18 accuracy 70.510%
epoch: 19 start
train:
(233.2314631268382, 0.90158)
test:
epoch: 19 accuracy 69.850%
epoch: 20 start
train:
(208.07088751345873, 0.91266)
test:
epoch: 20 accuracy 71.540%
epoch: 21 start
train:
(176.38965360075235, 0.9262)
test:
epoch: 21 accuracy 72.630%
epoch: 22 start
train:
(152.70060413330793, 0.9381)
test:
epoch: 22 accuracy 68.830%
epoch: 23 start
train:
(131.1207368262112, 0.94818)
test:
epoch: 23 accuracy 67.880%
epoch: 24 start
train:
(111.16313840448856, 0.95662)
test:
epoch: 24 accuracy 54.990%
epoch: 25 start
train:
(95.44256990216672, 0.96302)
test:
epoch: 25 accuracy 43.190%
epoch: 26 start
train:
(79.7576100435108, 0.97036)
test:
epoch: 26 accuracy 66.900%
epoch: 27 start
train:
(70.15928808227181, 0.9745)
test:
epoch: 27 accuracy 67.460%
epoch: 28 start
train:
(58.85909108631313, 0.97954)
test:
epoch: 28 accuracy 71.460%
epoch: 29 start
train:
(48.262854668311775, 0.98324)
test:
epoch: 29 accuracy 71.760%
epoch: 30 start
train:
(39.98544542212039, 0.98698)
test:
epoch: 30 accuracy 63.800%
epoch: 31 start
train:
(35.73263159673661, 0.98898)
test:
epoch: 31 accuracy 59.120%
epoch: 32 start
train:
(30.393140118103474, 0.99092)
test:
epoch: 32 accuracy 69.410%
epoch: 33 start
train:
(28.6019876874052, 0.9908)
test:
epoch: 33 accuracy 74.210%
epoch: 34 start
train:
(24.943911496549845, 0.99202)
test:
epoch: 34 accuracy 73.200%
epoch: 35 start
train:
(22.95255656214431, 0.99344)
test:
epoch: 35 accuracy 73.780%
epoch: 36 start
train:
(18.524661789881065, 0.99484)
test:
epoch: 36 accuracy 67.670%
epoch: 37 start
train:
(18.28441329440102, 0.99486)
test:
epoch: 37 accuracy 41.440%
epoch: 38 start
train:
(18.51223706966266, 0.9946)
test:
epoch: 38 accuracy 68.390%
epoch: 39 start
train:
(15.958623410901055, 0.99582)
test:
epoch: 39 accuracy 74.070%
epoch: 40 start
train:
(15.106540702050552, 0.99562)
test:
epoch: 40 accuracy 75.210%
epoch: 41 start
train:
(12.076440613134764, 0.9968)
test:
epoch: 41 accuracy 73.320%
epoch: 42 start
train:
(11.116505866288207, 0.99714)
test:
epoch: 42 accuracy 72.320%
epoch: 43 start
train:
(10.569189717061818, 0.99764)
test:
epoch: 43 accuracy 61.030%
epoch: 44 start
train:
(13.352942104218528, 0.99604)
test:
epoch: 44 accuracy 42.820%
epoch: 45 start
train:
(13.445744073251262, 0.99614)
test:
epoch: 45 accuracy 72.920%
epoch: 46 start
train:
(8.985479096358176, 0.99798)
test:
epoch: 46 accuracy 71.880%
epoch: 47 start
train:
(8.569246592000127, 0.99786)
test:
epoch: 47 accuracy 73.540%
epoch: 48 start
train:
(6.976069675933104, 0.99856)
test:
epoch: 48 accuracy 74.180%
epoch: 49 start
train:
(6.499934921681415, 0.99888)
test:
epoch: 49 accuracy 74.130%
epoch: 50 start
train:
(5.9637253725668415, 0.99876)
test:
epoch: 50 accuracy 74.480%
best accuracy: 75.210%
--------result report--------
best accuracy:75.210%
loss array:
1398.8536,1132.4189,1010.0803,912.3488,829.8427,760.4220,699.1129,647.2857,600.3491,559.9939,515.2445,479.2447,440.6301,402.0613,367.1886,335.0370,293.6162,264.6161,233.2315,208.0709,176.3897,152.7006,131.1207,111.1631,95.4426,79.7576,70.1593,58.8591,48.2629,39.9854,35.7326,30.3931,28.6020,24.9439,22.9526,18.5247,18.2844,18.5122,15.9586,15.1065,12.0764,11.1165,10.5692,13.3529,13.4457,8.9855,8.5692,6.9761,6.4999,5.9637,
accuracy array:
0.3897,0.4123,0.5010,0.5471,0.4594,0.5322,0.6004,0.6172,0.5972,0.6013,0.5406,0.6422,0.6844,0.6174,0.6013,0.6873,0.5771,0.7051,0.6985,0.7154,0.7263,0.6883,0.6788,0.5499,0.4319,0.6690,0.6746,0.7146,0.7176,0.6380,0.5912,0.6941,0.7421,0.7320,0.7378,0.6767,0.4144,0.6839,0.7407,0.7521,0.7332,0.7232,0.6103,0.4282,0.7292,0.7188,0.7354,0.7418,0.7413,0.7448,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:50
train batch size:128
test batch size:128
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(757.7458477020264, 0.29298)
test:
epoch: 1 accuracy 37.170%
epoch: 2 start
train:
(631.4178479909897, 0.404)
test:
epoch: 2 accuracy 36.700%
epoch: 3 start
train:
(574.0772995948792, 0.45774)
test:
epoch: 3 accuracy 34.790%
epoch: 4 start
train:
(532.2437393665314, 0.49926)
test:
epoch: 4 accuracy 48.810%
epoch: 5 start
train:
(501.90221428871155, 0.53264)
test:
epoch: 5 accuracy 31.730%
epoch: 6 start
train:
(472.29758858680725, 0.56324)
test:
epoch: 6 accuracy 46.960%
epoch: 7 start
train:
(447.63430643081665, 0.58572)
test:
epoch: 7 accuracy 55.540%
epoch: 8 start
train:
(423.9368763566017, 0.61162)
test:
epoch: 8 accuracy 53.650%
epoch: 9 start
train:
(402.61761808395386, 0.63048)
test:
epoch: 9 accuracy 59.310%
epoch: 10 start
train:
(383.48556476831436, 0.64932)
test:
epoch: 10 accuracy 60.570%
epoch: 11 start
train:
(363.6402772665024, 0.66788)
test:
epoch: 11 accuracy 48.470%
epoch: 12 start
train:
(347.25647896528244, 0.6838)
test:
epoch: 12 accuracy 60.730%
epoch: 13 start
train:
(332.4280751347542, 0.69804)
test:
epoch: 13 accuracy 59.830%
epoch: 14 start
train:
(316.13328605890274, 0.7149)
test:
epoch: 14 accuracy 62.100%
epoch: 15 start
train:
(301.73201325535774, 0.72796)
test:
epoch: 15 accuracy 62.900%
epoch: 16 start
train:
(287.1237065792084, 0.74196)
test:
epoch: 16 accuracy 64.530%
epoch: 17 start
train:
(273.9092334806919, 0.75422)
test:
epoch: 17 accuracy 58.180%
epoch: 18 start
train:
(259.9080834686756, 0.76812)
test:
epoch: 18 accuracy 67.340%
epoch: 19 start
train:
(246.57711419463158, 0.7806)
test:
epoch: 19 accuracy 65.290%
epoch: 20 start
train:
(234.16861802339554, 0.7907)
test:
epoch: 20 accuracy 62.530%
epoch: 21 start
train:
(219.80236288905144, 0.80744)
test:
epoch: 21 accuracy 66.090%
epoch: 22 start
train:
(208.9698920249939, 0.8158)
test:
epoch: 22 accuracy 67.800%
epoch: 23 start
train:
(193.8908085823059, 0.83012)
test:
epoch: 23 accuracy 65.020%
epoch: 24 start
train:
(181.7547868192196, 0.84256)
test:
epoch: 24 accuracy 67.160%
epoch: 25 start
train:
(167.2878989726305, 0.85514)
test:
epoch: 25 accuracy 67.700%
epoch: 26 start
train:
(155.99454288184643, 0.86692)
test:
epoch: 26 accuracy 69.150%
epoch: 27 start
train:
(142.34709987044334, 0.8829)
test:
epoch: 27 accuracy 66.170%
epoch: 28 start
train:
(130.05608043074608, 0.89252)
test:
epoch: 28 accuracy 69.550%
epoch: 29 start
train:
(119.15856729447842, 0.9005)
test:
epoch: 29 accuracy 64.530%
epoch: 30 start
train:
(106.44704128801823, 0.91686)
test:
epoch: 30 accuracy 65.360%
epoch: 31 start
train:
(96.27952013909817, 0.92346)
test:
epoch: 31 accuracy 69.570%
epoch: 32 start
train:
(84.72804003953934, 0.93568)
test:
epoch: 32 accuracy 64.830%
epoch: 33 start
train:
(73.96120127290487, 0.94456)
test:
epoch: 33 accuracy 68.790%
epoch: 34 start
train:
(64.41662388294935, 0.95428)
test:
epoch: 34 accuracy 65.270%
epoch: 35 start
train:
(54.65476320683956, 0.96244)
test:
epoch: 35 accuracy 66.870%
epoch: 36 start
train:
(47.057315081357956, 0.97082)
test:
epoch: 36 accuracy 68.540%
epoch: 37 start
train:
(40.147724106907845, 0.97582)
test:
epoch: 37 accuracy 68.210%
epoch: 38 start
train:
(36.55504812672734, 0.97774)
test:
epoch: 38 accuracy 68.050%
epoch: 39 start
train:
(30.061956839635968, 0.98352)
test:
epoch: 39 accuracy 69.560%
epoch: 40 start
train:
(26.101611383259296, 0.9866)
test:
epoch: 40 accuracy 68.540%
epoch: 41 start
train:
(22.82481485232711, 0.98862)
test:
epoch: 41 accuracy 69.350%
epoch: 42 start
train:
(18.968581028282642, 0.99192)
test:
epoch: 42 accuracy 70.220%
epoch: 43 start
train:
(17.2987367156893, 0.99244)
test:
epoch: 43 accuracy 69.680%
epoch: 44 start
train:
(14.021510798484087, 0.99498)
test:
epoch: 44 accuracy 69.020%
epoch: 45 start
train:
(11.597212088294327, 0.99622)
test:
epoch: 45 accuracy 70.600%
epoch: 46 start
train:
(10.282411647029221, 0.99722)
test:
epoch: 46 accuracy 69.850%
epoch: 47 start
train:
(9.200088620651513, 0.99754)
test:
epoch: 47 accuracy 69.000%
epoch: 48 start
train:
(8.010945090092719, 0.9981)
test:
epoch: 48 accuracy 68.630%
epoch: 49 start
train:
(7.70052554178983, 0.99826)
test:
epoch: 49 accuracy 69.720%
epoch: 50 start
train:
(6.380768559407443, 0.99884)
test:
epoch: 50 accuracy 70.440%
best accuracy: 70.600%
--------result report--------
best accuracy:70.600%
loss array:
757.7458,631.4178,574.0773,532.2437,501.9022,472.2976,447.6343,423.9369,402.6176,383.4856,363.6403,347.2565,332.4281,316.1333,301.7320,287.1237,273.9092,259.9081,246.5771,234.1686,219.8024,208.9699,193.8908,181.7548,167.2879,155.9945,142.3471,130.0561,119.1586,106.4470,96.2795,84.7280,73.9612,64.4166,54.6548,47.0573,40.1477,36.5550,30.0620,26.1016,22.8248,18.9686,17.2987,14.0215,11.5972,10.2824,9.2001,8.0109,7.7005,6.3808,
accuracy array:
0.3717,0.3670,0.3479,0.4881,0.3173,0.4696,0.5554,0.5365,0.5931,0.6057,0.4847,0.6073,0.5983,0.6210,0.6290,0.6453,0.5818,0.6734,0.6529,0.6253,0.6609,0.6780,0.6502,0.6716,0.6770,0.6915,0.6617,0.6955,0.6453,0.6536,0.6957,0.6483,0.6879,0.6527,0.6687,0.6854,0.6821,0.6805,0.6956,0.6854,0.6935,0.7022,0.6968,0.6902,0.7060,0.6985,0.6900,0.6863,0.6972,0.7044,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:50
train batch size:256
test batch size:256
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(409.2187705039978, 0.24012)
test:
epoch: 1 accuracy 30.480%
epoch: 2 start
train:
(351.2524175643921, 0.34072)
test:
epoch: 2 accuracy 36.600%
epoch: 3 start
train:
(322.6464533805847, 0.39218)
test:
epoch: 3 accuracy 41.620%
epoch: 4 start
train:
(301.9124631881714, 0.43026)
test:
epoch: 4 accuracy 42.910%
epoch: 5 start
train:
(286.6763845682144, 0.46248)
test:
epoch: 5 accuracy 47.190%
epoch: 6 start
train:
(273.4162929058075, 0.48816)
test:
epoch: 6 accuracy 47.970%
epoch: 7 start
train:
(262.0954737663269, 0.50888)
test:
epoch: 7 accuracy 46.130%
epoch: 8 start
train:
(251.7060376405716, 0.53092)
test:
epoch: 8 accuracy 39.150%
epoch: 9 start
train:
(242.66067671775818, 0.54948)
test:
epoch: 9 accuracy 43.070%
epoch: 10 start
train:
(234.21150016784668, 0.56532)
test:
epoch: 10 accuracy 52.530%
epoch: 11 start
train:
(226.7552306652069, 0.58004)
test:
epoch: 11 accuracy 51.480%
epoch: 12 start
train:
(218.9807552099228, 0.59524)
test:
epoch: 12 accuracy 49.050%
epoch: 13 start
train:
(212.61563217639923, 0.60654)
test:
epoch: 13 accuracy 50.250%
epoch: 14 start
train:
(206.23634320497513, 0.6198)
test:
epoch: 14 accuracy 53.930%
epoch: 15 start
train:
(199.94434970617294, 0.63084)
test:
epoch: 15 accuracy 56.670%
epoch: 16 start
train:
(193.5938537120819, 0.64524)
test:
epoch: 16 accuracy 48.900%
epoch: 17 start
train:
(188.42466562986374, 0.65488)
test:
epoch: 17 accuracy 51.730%
epoch: 18 start
train:
(182.30044144392014, 0.66712)
test:
epoch: 18 accuracy 57.340%
epoch: 19 start
train:
(177.38018721342087, 0.6783)
test:
epoch: 19 accuracy 53.780%
epoch: 20 start
train:
(172.1100376844406, 0.6874)
test:
epoch: 20 accuracy 58.790%
epoch: 21 start
train:
(167.14743250608444, 0.6966)
test:
epoch: 21 accuracy 62.950%
epoch: 22 start
train:
(162.49486649036407, 0.70736)
test:
epoch: 22 accuracy 56.880%
epoch: 23 start
train:
(157.4505756497383, 0.71388)
test:
epoch: 23 accuracy 62.290%
epoch: 24 start
train:
(153.44541734457016, 0.72232)
test:
epoch: 24 accuracy 60.990%
epoch: 25 start
train:
(148.02797371149063, 0.7322)
test:
epoch: 25 accuracy 58.720%
epoch: 26 start
train:
(143.27257364988327, 0.7411)
test:
epoch: 26 accuracy 54.100%
epoch: 27 start
train:
(138.7636239528656, 0.7505)
test:
epoch: 27 accuracy 55.890%
epoch: 28 start
train:
(135.2546803355217, 0.75624)
test:
epoch: 28 accuracy 56.900%
epoch: 29 start
train:
(130.81194001436234, 0.76784)
test:
epoch: 29 accuracy 64.700%
epoch: 30 start
train:
(127.77117329835892, 0.7709)
test:
epoch: 30 accuracy 59.320%
epoch: 31 start
train:
(122.78007790446281, 0.7809)
test:
epoch: 31 accuracy 65.570%
epoch: 32 start
train:
(118.4275381565094, 0.79048)
test:
epoch: 32 accuracy 65.330%
epoch: 33 start
train:
(114.55383294820786, 0.7975)
test:
epoch: 33 accuracy 62.650%
epoch: 34 start
train:
(110.711843252182, 0.80608)
test:
epoch: 34 accuracy 59.270%
epoch: 35 start
train:
(105.89175868034363, 0.81672)
test:
epoch: 35 accuracy 32.770%
epoch: 36 start
train:
(102.20003405213356, 0.82208)
test:
epoch: 36 accuracy 63.810%
epoch: 37 start
train:
(98.07990998029709, 0.8314)
test:
epoch: 37 accuracy 52.850%
epoch: 38 start
train:
(93.17555323243141, 0.83944)
test:
epoch: 38 accuracy 69.490%
epoch: 39 start
train:
(89.8103714287281, 0.84668)
test:
epoch: 39 accuracy 63.820%
epoch: 40 start
train:
(85.22634238004684, 0.85694)
test:
epoch: 40 accuracy 59.420%
epoch: 41 start
train:
(80.67662572860718, 0.86548)
test:
epoch: 41 accuracy 51.180%
epoch: 42 start
train:
(76.60209834575653, 0.8731)
test:
epoch: 42 accuracy 61.530%
epoch: 43 start
train:
(73.6467157304287, 0.88012)
test:
epoch: 43 accuracy 63.210%
epoch: 44 start
train:
(69.41196502745152, 0.88722)
test:
epoch: 44 accuracy 61.380%
epoch: 45 start
train:
(65.19970300793648, 0.896)
test:
epoch: 45 accuracy 67.570%
epoch: 46 start
train:
(60.6176670640707, 0.90444)
test:
epoch: 46 accuracy 64.160%
epoch: 47 start
train:
(56.37509082257748, 0.91374)
test:
epoch: 47 accuracy 59.320%
epoch: 48 start
train:
(53.010171204805374, 0.9203)
test:
epoch: 48 accuracy 68.050%
epoch: 49 start
train:
(49.34080982208252, 0.92794)
test:
epoch: 49 accuracy 67.800%
epoch: 50 start
train:
(45.10558395087719, 0.93678)
test:
epoch: 50 accuracy 60.280%
best accuracy: 69.490%
--------result report--------
best accuracy:69.490%
loss array:
409.2188,351.2524,322.6465,301.9125,286.6764,273.4163,262.0955,251.7060,242.6607,234.2115,226.7552,218.9808,212.6156,206.2363,199.9443,193.5939,188.4247,182.3004,177.3802,172.1100,167.1474,162.4949,157.4506,153.4454,148.0280,143.2726,138.7636,135.2547,130.8119,127.7712,122.7801,118.4275,114.5538,110.7118,105.8918,102.2000,98.0799,93.1756,89.8104,85.2263,80.6766,76.6021,73.6467,69.4120,65.1997,60.6177,56.3751,53.0102,49.3408,45.1056,
accuracy array:
0.3048,0.3660,0.4162,0.4291,0.4719,0.4797,0.4613,0.3915,0.4307,0.5253,0.5148,0.4905,0.5025,0.5393,0.5667,0.4890,0.5173,0.5734,0.5378,0.5879,0.6295,0.5688,0.6229,0.6099,0.5872,0.5410,0.5589,0.5690,0.6470,0.5932,0.6557,0.6533,0.6265,0.5927,0.3277,0.6381,0.5285,0.6949,0.6382,0.5942,0.5118,0.6153,0.6321,0.6138,0.6757,0.6416,0.5932,0.6805,0.6780,0.6028,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:50
train batch size:512
test batch size:512
model:resnet18
optimizer:SGD
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(213.39185309410095, 0.21442)
test:
epoch: 1 accuracy 27.060%
epoch: 2 start
train:
(190.5841507911682, 0.29822)
test:
epoch: 2 accuracy 32.540%
epoch: 3 start
train:
(176.68066465854645, 0.34434)
test:
epoch: 3 accuracy 36.450%
epoch: 4 start
train:
(167.62262284755707, 0.3761)
test:
epoch: 4 accuracy 37.500%
epoch: 5 start
train:
(160.41727566719055, 0.40074)
test:
epoch: 5 accuracy 40.880%
epoch: 6 start
train:
(154.73649954795837, 0.41744)
test:
epoch: 6 accuracy 42.840%
epoch: 7 start
train:
(150.02175641059875, 0.43534)
test:
epoch: 7 accuracy 43.460%
epoch: 8 start
train:
(146.03376579284668, 0.45188)
test:
epoch: 8 accuracy 43.420%
epoch: 9 start
train:
(142.54206442832947, 0.46632)
test:
epoch: 9 accuracy 44.990%
epoch: 10 start
train:
(139.27254140377045, 0.48224)
test:
epoch: 10 accuracy 46.240%
epoch: 11 start
train:
(136.33427941799164, 0.49264)
test:
epoch: 11 accuracy 48.530%
epoch: 12 start
train:
(133.5731781721115, 0.50398)
test:
epoch: 12 accuracy 46.970%
epoch: 13 start
train:
(130.64439725875854, 0.513)
test:
epoch: 13 accuracy 44.780%
epoch: 14 start
train:
(128.1196504831314, 0.5244)
test:
epoch: 14 accuracy 51.120%
epoch: 15 start
train:
(125.69917094707489, 0.53436)
test:
epoch: 15 accuracy 46.060%
epoch: 16 start
train:
(123.36222624778748, 0.54404)
test:
epoch: 16 accuracy 50.270%
epoch: 17 start
train:
(120.78892743587494, 0.55412)
test:
epoch: 17 accuracy 52.540%
epoch: 18 start
train:
(119.13649213314056, 0.55988)
test:
epoch: 18 accuracy 43.540%
epoch: 19 start
train:
(116.74616301059723, 0.57036)
test:
epoch: 19 accuracy 54.990%
epoch: 20 start
train:
(114.89335477352142, 0.57674)
test:
epoch: 20 accuracy 46.200%
epoch: 21 start
train:
(112.79396688938141, 0.58466)
test:
epoch: 21 accuracy 54.690%
epoch: 22 start
train:
(110.98675274848938, 0.59102)
test:
epoch: 22 accuracy 53.190%
epoch: 23 start
train:
(109.1376440525055, 0.60002)
test:
epoch: 23 accuracy 52.850%
epoch: 24 start
train:
(107.3652902841568, 0.60664)
test:
epoch: 24 accuracy 56.000%
epoch: 25 start
train:
(105.73686450719833, 0.61336)
test:
epoch: 25 accuracy 42.750%
epoch: 26 start
train:
(103.85436654090881, 0.61854)
test:
epoch: 26 accuracy 57.680%
epoch: 27 start
train:
(102.02662128210068, 0.626)
test:
epoch: 27 accuracy 51.650%
epoch: 28 start
train:
(100.50111681222916, 0.63364)
test:
epoch: 28 accuracy 58.020%
epoch: 29 start
train:
(98.97845888137817, 0.6389)
test:
epoch: 29 accuracy 59.220%
epoch: 30 start
train:
(97.52763599157333, 0.6452)
test:
epoch: 30 accuracy 49.550%
epoch: 31 start
train:
(96.0795909166336, 0.64998)
test:
epoch: 31 accuracy 54.590%
epoch: 32 start
train:
(94.30924415588379, 0.6557)
test:
epoch: 32 accuracy 43.420%
epoch: 33 start
train:
(93.31880575418472, 0.65908)
test:
epoch: 33 accuracy 54.150%
epoch: 34 start
train:
(91.1816241145134, 0.66796)
test:
epoch: 34 accuracy 50.860%
epoch: 35 start
train:
(90.10587960481644, 0.67176)
test:
epoch: 35 accuracy 58.540%
epoch: 36 start
train:
(88.41709041595459, 0.6788)
test:
epoch: 36 accuracy 53.740%
epoch: 37 start
train:
(87.40828895568848, 0.68356)
test:
epoch: 37 accuracy 58.570%
epoch: 38 start
train:
(85.76456987857819, 0.69028)
test:
epoch: 38 accuracy 59.460%
epoch: 39 start
train:
(83.94307392835617, 0.69822)
test:
epoch: 39 accuracy 57.540%
epoch: 40 start
train:
(82.9261366724968, 0.70196)
test:
epoch: 40 accuracy 63.440%
epoch: 41 start
train:
(81.57933741807938, 0.70576)
test:
epoch: 41 accuracy 62.080%
epoch: 42 start
train:
(80.11114156246185, 0.7113)
test:
epoch: 42 accuracy 58.970%
epoch: 43 start
train:
(78.70782917737961, 0.71708)
test:
epoch: 43 accuracy 62.840%
epoch: 44 start
train:
(77.29959052801132, 0.72232)
test:
epoch: 44 accuracy 60.830%
epoch: 45 start
train:
(76.11567950248718, 0.72568)
test:
epoch: 45 accuracy 59.130%
epoch: 46 start
train:
(74.73731434345245, 0.73236)
test:
epoch: 46 accuracy 65.170%
epoch: 47 start
train:
(73.42758506536484, 0.73652)
test:
epoch: 47 accuracy 62.340%
epoch: 48 start
train:
(72.05184370279312, 0.74194)
test:
epoch: 48 accuracy 62.130%
epoch: 49 start
train:
(70.82155239582062, 0.74838)
test:
epoch: 49 accuracy 63.200%
epoch: 50 start
train:
(69.55633842945099, 0.75334)
test:
epoch: 50 accuracy 64.840%
best accuracy: 65.170%
--------result report--------
best accuracy:65.170%
loss array:
213.3919,190.5842,176.6807,167.6226,160.4173,154.7365,150.0218,146.0338,142.5421,139.2725,136.3343,133.5732,130.6444,128.1197,125.6992,123.3622,120.7889,119.1365,116.7462,114.8934,112.7940,110.9868,109.1376,107.3653,105.7369,103.8544,102.0266,100.5011,98.9785,97.5276,96.0796,94.3092,93.3188,91.1816,90.1059,88.4171,87.4083,85.7646,83.9431,82.9261,81.5793,80.1111,78.7078,77.2996,76.1157,74.7373,73.4276,72.0518,70.8216,69.5563,
accuracy array:
0.2706,0.3254,0.3645,0.3750,0.4088,0.4284,0.4346,0.4342,0.4499,0.4624,0.4853,0.4697,0.4478,0.5112,0.4606,0.5027,0.5254,0.4354,0.5499,0.4620,0.5469,0.5319,0.5285,0.5600,0.4275,0.5768,0.5165,0.5802,0.5922,0.4955,0.5459,0.4342,0.5415,0.5086,0.5854,0.5374,0.5857,0.5946,0.5754,0.6344,0.6208,0.5897,0.6284,0.6083,0.5913,0.6517,0.6234,0.6213,0.6320,0.6484,
-------------------------------
--------parameter report--------
batchSize:16    accuracy:0.822%
batchSize:32    accuracy:0.791%
batchSize:64    accuracy:0.752%
batchSize:128    accuracy:0.706%
batchSize:256    accuracy:0.695%
batchSize:512    accuracy:0.652%
```