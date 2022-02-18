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

MODEL_CHOICE = 18
OPT_CHOICE = 0
LEARNING_RATE = 0.005
EPOCH = 30
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 64


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

    def result_report(self, loss_array, train_accuracy, accuracy_array, best_accuracy, train_accuracy_array):
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
        print('test accuracy array:')
        for accuracy in accuracy_array:
            print('%.4f' % accuracy, end=',')
        print('')
        print('train accuracy array:')
        for accuracy in train_accuracy_array:
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

        self.result_report(loss_array, train_accuracy, accuracy_array, accuracy * 100, train_accuracy)
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
        print("batchSize:%d    accuracy:%.3f%%" % (batchSizeArray[i], accuracyArray[i]))


def testLearningRate(LearningRateArray):
    accuracyArray = []
    for lrr in LearningRateArray:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_choice', default=MODEL_CHOICE, type=int)  # resnet18:15 resnet50:50
        parser.add_argument('--opt_choice', default=OPT_CHOICE, type=int)  # 0:adam 1:sdg
        parser.add_argument('--lr', default=lrr, type=float)
        parser.add_argument('--epoch', default=EPOCH, type=int)
        parser.add_argument('--trainBatchSize', default=16, type=int)
        parser.add_argument('--testBatchSize', default=16, type=int)
        args = parser.parse_known_args()[0]
        runner = Runner(args)
        accuracyArray.append(runner.run())
    print("--------parameter report--------")
    for i in range(len(LearningRateArray)):
        print("batchSize:%d    accuracy:%.3f%%" % (LearningRateArray[i], accuracyArray[i]))


if __name__ == '__main__':
    setup_seed(2021)
    batchSizeArray = [16]
    learning_array = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015]
    testLearningRate(learning_array)
--------parameter report--------
learning rate:0.00100
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data/cifar-10-python.tar.gz
```

170499072/? [00:02<00:00, 86186880.52it/s]

```
Extracting ../data/cifar-10-python.tar.gz to ../data
Files already downloaded and verified
epoch: 1 start
train:
(4072.6030228435993, 0.52854)
test:
epoch: 1 accuracy 69.910%
epoch: 2 start
train:
(2480.832460127771, 0.7257)
test:
epoch: 2 accuracy 74.600%
epoch: 3 start
train:
(1897.190456353128, 0.79148)
test:
epoch: 3 accuracy 81.100%
epoch: 4 start
train:
(1532.4618175663054, 0.83006)
test:
epoch: 4 accuracy 82.060%
epoch: 5 start
train:
(1251.7539138831198, 0.86092)
test:
epoch: 5 accuracy 83.310%
epoch: 6 start
train:
(1039.269017253071, 0.88662)
test:
epoch: 6 accuracy 84.720%
epoch: 7 start
train:
(851.0100557813421, 0.90548)
test:
epoch: 7 accuracy 84.770%
epoch: 8 start
train:
(700.3544449836481, 0.92118)
test:
epoch: 8 accuracy 84.950%
epoch: 9 start
train:
(573.5181853717659, 0.9368)
test:
epoch: 9 accuracy 86.550%
epoch: 10 start
train:
(475.2054639029084, 0.9469)
test:
epoch: 10 accuracy 85.730%
epoch: 11 start
train:
(404.69667302264133, 0.9555)
test:
epoch: 11 accuracy 87.060%
epoch: 12 start
train:
(334.7783113003825, 0.96358)
test:
epoch: 12 accuracy 86.260%
epoch: 13 start
train:
(292.09846760187065, 0.96814)
test:
epoch: 13 accuracy 86.980%
epoch: 14 start
train:
(263.4442046478507, 0.9705)
test:
epoch: 14 accuracy 87.020%
epoch: 15 start
train:
(226.36496022211213, 0.97498)
test:
epoch: 15 accuracy 87.070%
epoch: 16 start
train:
(209.50593062534244, 0.97742)
test:
epoch: 16 accuracy 87.560%
epoch: 17 start
train:
(196.87640327409463, 0.97802)
test:
epoch: 17 accuracy 86.780%
epoch: 18 start
train:
(173.18329803231245, 0.98006)
test:
epoch: 18 accuracy 87.240%
epoch: 19 start
train:
(155.71705841027506, 0.98298)
test:
epoch: 19 accuracy 86.770%
epoch: 20 start
train:
(151.36621992136133, 0.9835)
test:
epoch: 20 accuracy 87.180%
epoch: 21 start
train:
(143.5355668622651, 0.98358)
test:
epoch: 21 accuracy 87.670%
epoch: 22 start
train:
(128.48387404052846, 0.98532)
test:
epoch: 22 accuracy 87.500%
epoch: 23 start
train:
(119.5805448367355, 0.9869)
test:
epoch: 23 accuracy 87.220%
epoch: 24 start
train:
(123.38957253522403, 0.9867)
test:
epoch: 24 accuracy 87.460%
epoch: 25 start
train:
(118.92495001961106, 0.9878)
test:
epoch: 25 accuracy 87.860%
epoch: 26 start
train:
(100.22067127862829, 0.98886)
test:
epoch: 26 accuracy 87.490%
epoch: 27 start
train:
(105.34496373846468, 0.98856)
test:
epoch: 27 accuracy 86.900%
epoch: 28 start
train:
(94.59033510547215, 0.98966)
test:
epoch: 28 accuracy 87.170%
epoch: 29 start
train:
(91.72410005936581, 0.99024)
test:
epoch: 29 accuracy 87.890%
epoch: 30 start
train:
(91.9385421360414, 0.99014)
test:
epoch: 30 accuracy 88.070%
best accuracy: 88.070%
--------result report--------
best accuracy:88.070%
loss array:
4072.6030,2480.8325,1897.1905,1532.4618,1251.7539,1039.2690,851.0101,700.3544,573.5182,475.2055,404.6967,334.7783,292.0985,263.4442,226.3650,209.5059,196.8764,173.1833,155.7171,151.3662,143.5356,128.4839,119.5805,123.3896,118.9250,100.2207,105.3450,94.5903,91.7241,91.9385,
train accuracy array:
0.5285,0.7257,0.7915,0.8301,0.8609,0.8866,0.9055,0.9212,0.9368,0.9469,0.9555,0.9636,0.9681,0.9705,0.9750,0.9774,0.9780,0.9801,0.9830,0.9835,0.9836,0.9853,0.9869,0.9867,0.9878,0.9889,0.9886,0.9897,0.9902,0.9901,
test accuracy array:
0.6991,0.7460,0.8110,0.8206,0.8331,0.8472,0.8477,0.8495,0.8655,0.8573,0.8706,0.8626,0.8698,0.8702,0.8707,0.8756,0.8678,0.8724,0.8677,0.8718,0.8767,0.8750,0.8722,0.8746,0.8786,0.8749,0.8690,0.8717,0.8789,0.8807,
train accuracy array:
0.5285,0.7257,0.7915,0.8301,0.8609,0.8866,0.9055,0.9212,0.9368,0.9469,0.9555,0.9636,0.9681,0.9705,0.9750,0.9774,0.9780,0.9801,0.9830,0.9835,0.9836,0.9853,0.9869,0.9867,0.9878,0.9889,0.9886,0.9897,0.9902,0.9901,
-------------------------------
--------parameter report--------
learning rate:0.00300
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(4682.598452925682, 0.44772)
test:
epoch: 1 accuracy 61.340%
epoch: 2 start
train:
(2895.4091205596924, 0.67218)
test:
epoch: 2 accuracy 73.700%
epoch: 3 start
train:
(2088.1189586520195, 0.7679)
test:
epoch: 3 accuracy 79.030%
epoch: 4 start
train:
(1649.6819696351886, 0.81676)
test:
epoch: 4 accuracy 82.080%
epoch: 5 start
train:
(1337.9036655202508, 0.85086)
test:
epoch: 5 accuracy 84.240%
epoch: 6 start
train:
(1105.6461857380345, 0.8775)
test:
epoch: 6 accuracy 82.230%
epoch: 7 start
train:
(906.7477691122331, 0.89978)
test:
epoch: 7 accuracy 85.010%
epoch: 8 start
train:
(736.3759299837984, 0.91858)
test:
epoch: 8 accuracy 85.360%
epoch: 9 start
train:
(623.2628240189515, 0.93012)
test:
epoch: 9 accuracy 85.620%
epoch: 10 start
train:
(525.1311561047914, 0.94028)
test:
epoch: 10 accuracy 86.180%
epoch: 11 start
train:
(437.8281685950933, 0.95166)
test:
epoch: 11 accuracy 86.340%
epoch: 12 start
train:
(367.61460046225693, 0.95956)
test:
epoch: 12 accuracy 86.580%
epoch: 13 start
train:
(324.9625959985424, 0.96404)
test:
epoch: 13 accuracy 86.330%
epoch: 14 start
train:
(284.795241744092, 0.9674)
test:
epoch: 14 accuracy 87.100%
epoch: 15 start
train:
(263.3640628416615, 0.97092)
test:
epoch: 15 accuracy 86.210%
epoch: 16 start
train:
(236.26560779994543, 0.97424)
test:
epoch: 16 accuracy 85.790%
epoch: 17 start
train:
(215.2101700849489, 0.97612)
test:
epoch: 17 accuracy 86.070%
epoch: 18 start
train:
(211.82989279020876, 0.97664)
test:
epoch: 18 accuracy 86.440%
epoch: 19 start
train:
(188.55765176172645, 0.98016)
test:
epoch: 19 accuracy 86.280%
epoch: 20 start
train:
(192.51937554203505, 0.97902)
test:
epoch: 20 accuracy 86.800%
epoch: 21 start
train:
(166.2009142851557, 0.98198)
test:
epoch: 21 accuracy 86.680%
epoch: 22 start
train:
(162.1048474617155, 0.98244)
test:
epoch: 22 accuracy 86.610%
epoch: 23 start
train:
(158.70136582170926, 0.9828)
test:
epoch: 23 accuracy 86.930%
epoch: 24 start
train:
(142.26245852816555, 0.98442)
test:
epoch: 24 accuracy 86.710%
epoch: 25 start
train:
(142.44347298539287, 0.98432)
test:
epoch: 25 accuracy 86.550%
epoch: 26 start
train:
(137.30315851632508, 0.98502)
test:
epoch: 26 accuracy 87.000%
epoch: 27 start
train:
(125.93446748225233, 0.9862)
test:
epoch: 27 accuracy 87.590%
epoch: 28 start
train:
(123.51730779083175, 0.98642)
test:
epoch: 28 accuracy 86.830%
epoch: 29 start
train:
(118.63477062218635, 0.98672)
test:
epoch: 29 accuracy 86.390%
epoch: 30 start
train:
(120.19528929283166, 0.98642)
test:
epoch: 30 accuracy 86.630%
best accuracy: 87.590%
--------result report--------
best accuracy:87.590%
loss array:
4682.5985,2895.4091,2088.1190,1649.6820,1337.9037,1105.6462,906.7478,736.3759,623.2628,525.1312,437.8282,367.6146,324.9626,284.7952,263.3641,236.2656,215.2102,211.8299,188.5577,192.5194,166.2009,162.1048,158.7014,142.2625,142.4435,137.3032,125.9345,123.5173,118.6348,120.1953,
train accuracy array:
0.4477,0.6722,0.7679,0.8168,0.8509,0.8775,0.8998,0.9186,0.9301,0.9403,0.9517,0.9596,0.9640,0.9674,0.9709,0.9742,0.9761,0.9766,0.9802,0.9790,0.9820,0.9824,0.9828,0.9844,0.9843,0.9850,0.9862,0.9864,0.9867,0.9864,
test accuracy array:
0.6134,0.7370,0.7903,0.8208,0.8424,0.8223,0.8501,0.8536,0.8562,0.8618,0.8634,0.8658,0.8633,0.8710,0.8621,0.8579,0.8607,0.8644,0.8628,0.8680,0.8668,0.8661,0.8693,0.8671,0.8655,0.8700,0.8759,0.8683,0.8639,0.8663,
train accuracy array:
0.4477,0.6722,0.7679,0.8168,0.8509,0.8775,0.8998,0.9186,0.9301,0.9403,0.9517,0.9596,0.9640,0.9674,0.9709,0.9742,0.9761,0.9766,0.9802,0.9790,0.9820,0.9824,0.9828,0.9844,0.9843,0.9850,0.9862,0.9864,0.9867,0.9864,
-------------------------------
--------parameter report--------
learning rate:0.00500
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(5102.632133126259, 0.3995)
test:
epoch: 1 accuracy 55.560%
epoch: 2 start
train:
(3123.9335125535727, 0.6423)
test:
epoch: 2 accuracy 69.990%
epoch: 3 start
train:
(2293.6283514127135, 0.74448)
test:
epoch: 3 accuracy 77.840%
epoch: 4 start
train:
(1751.0996870808303, 0.80542)
test:
epoch: 4 accuracy 81.090%
epoch: 5 start
train:
(1426.8324277848005, 0.84212)
test:
epoch: 5 accuracy 82.610%
epoch: 6 start
train:
(1173.5340673374012, 0.86844)
test:
epoch: 6 accuracy 83.100%
epoch: 7 start
train:
(984.6653680773452, 0.8892)
test:
epoch: 7 accuracy 85.020%
epoch: 8 start
train:
(806.4253955001477, 0.91154)
test:
epoch: 8 accuracy 85.750%
epoch: 9 start
train:
(678.3162749445764, 0.92556)
test:
epoch: 9 accuracy 86.090%
epoch: 10 start
train:
(589.6148727007676, 0.93478)
test:
epoch: 10 accuracy 86.480%
epoch: 11 start
train:
(500.55643589486135, 0.94444)
test:
epoch: 11 accuracy 85.870%
epoch: 12 start
train:
(439.22447300425847, 0.95182)
test:
epoch: 12 accuracy 84.820%
epoch: 13 start
train:
(383.36867430456914, 0.95758)
test:
epoch: 13 accuracy 86.000%
epoch: 14 start
train:
(322.91393382613387, 0.96362)
test:
epoch: 14 accuracy 85.660%
epoch: 15 start
train:
(309.41473503714224, 0.96562)
test:
epoch: 15 accuracy 86.680%
epoch: 16 start
train:
(286.51505142185124, 0.96784)
test:
epoch: 16 accuracy 86.950%
epoch: 17 start
train:
(267.82896544468895, 0.97002)
test:
epoch: 17 accuracy 86.180%
epoch: 18 start
train:
(247.504379523456, 0.97238)
test:
epoch: 18 accuracy 85.980%
epoch: 19 start
train:
(223.3055403262806, 0.97492)
test:
epoch: 19 accuracy 85.880%
epoch: 20 start
train:
(218.431904216799, 0.97594)
test:
epoch: 20 accuracy 86.010%
epoch: 21 start
train:
(204.14425245160237, 0.97748)
test:
epoch: 21 accuracy 86.370%
epoch: 22 start
train:
(205.2660832386905, 0.97802)
test:
epoch: 22 accuracy 86.040%
epoch: 23 start
train:
(188.68713807987842, 0.97906)
test:
epoch: 23 accuracy 86.280%
epoch: 24 start
train:
(171.604982129154, 0.9814)
test:
epoch: 24 accuracy 86.930%
epoch: 25 start
train:
(170.60255183146955, 0.98196)
test:
epoch: 25 accuracy 86.420%
epoch: 26 start
train:
(180.91248244903272, 0.98054)
test:
epoch: 26 accuracy 85.910%
epoch: 27 start
train:
(150.3657073820915, 0.98396)
test:
epoch: 27 accuracy 86.180%
epoch: 28 start
train:
(167.89701172065315, 0.98172)
test:
epoch: 28 accuracy 86.830%
epoch: 29 start
train:
(152.467139207429, 0.98376)
test:
epoch: 29 accuracy 86.460%
epoch: 30 start
train:
(157.23925346665715, 0.98314)
test:
epoch: 30 accuracy 86.210%
best accuracy: 86.950%
--------result report--------
best accuracy:86.950%
loss array:
5102.6321,3123.9335,2293.6284,1751.0997,1426.8324,1173.5341,984.6654,806.4254,678.3163,589.6149,500.5564,439.2245,383.3687,322.9139,309.4147,286.5151,267.8290,247.5044,223.3055,218.4319,204.1443,205.2661,188.6871,171.6050,170.6026,180.9125,150.3657,167.8970,152.4671,157.2393,
train accuracy array:
0.3995,0.6423,0.7445,0.8054,0.8421,0.8684,0.8892,0.9115,0.9256,0.9348,0.9444,0.9518,0.9576,0.9636,0.9656,0.9678,0.9700,0.9724,0.9749,0.9759,0.9775,0.9780,0.9791,0.9814,0.9820,0.9805,0.9840,0.9817,0.9838,0.9831,
test accuracy array:
0.5556,0.6999,0.7784,0.8109,0.8261,0.8310,0.8502,0.8575,0.8609,0.8648,0.8587,0.8482,0.8600,0.8566,0.8668,0.8695,0.8618,0.8598,0.8588,0.8601,0.8637,0.8604,0.8628,0.8693,0.8642,0.8591,0.8618,0.8683,0.8646,0.8621,
train accuracy array:
0.3995,0.6423,0.7445,0.8054,0.8421,0.8684,0.8892,0.9115,0.9256,0.9348,0.9444,0.9518,0.9576,0.9636,0.9656,0.9678,0.9700,0.9724,0.9749,0.9759,0.9775,0.9780,0.9791,0.9814,0.9820,0.9805,0.9840,0.9817,0.9838,0.9831,
-------------------------------
--------parameter report--------
learning rate:0.00700
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(5179.30052536726, 0.3948)
test:
epoch: 1 accuracy 55.200%
epoch: 2 start
train:
(3141.450041115284, 0.64488)
test:
epoch: 2 accuracy 71.780%
epoch: 3 start
train:
(2191.1988438665867, 0.75672)
test:
epoch: 3 accuracy 78.310%
epoch: 4 start
train:
(1737.4292402341962, 0.80678)
test:
epoch: 4 accuracy 80.870%
epoch: 5 start
train:
(1450.8914065808058, 0.83964)
test:
epoch: 5 accuracy 82.140%
epoch: 6 start
train:
(1206.9402554184198, 0.86664)
test:
epoch: 6 accuracy 82.450%
epoch: 7 start
train:
(1020.3974699927494, 0.88758)
test:
epoch: 7 accuracy 83.250%
epoch: 8 start
train:
(870.6220804518089, 0.9033)
test:
epoch: 8 accuracy 84.170%
epoch: 9 start
train:
(749.8027533625718, 0.91768)
test:
epoch: 9 accuracy 84.490%
epoch: 10 start
train:
(645.5848020047415, 0.92738)
test:
epoch: 10 accuracy 84.980%
epoch: 11 start
train:
(556.6275544884265, 0.9381)
test:
epoch: 11 accuracy 85.400%
epoch: 12 start
train:
(481.4865565949294, 0.94634)
test:
epoch: 12 accuracy 84.980%
epoch: 13 start
train:
(428.28288975440955, 0.95346)
test:
epoch: 13 accuracy 85.210%
epoch: 14 start
train:
(393.99217134842183, 0.95678)
test:
epoch: 14 accuracy 85.480%
epoch: 15 start
train:
(355.37251846084837, 0.96116)
test:
epoch: 15 accuracy 84.890%
epoch: 16 start
train:
(333.36507589415123, 0.96406)
test:
epoch: 16 accuracy 84.770%
epoch: 17 start
train:
(304.7762139798906, 0.96648)
test:
epoch: 17 accuracy 85.860%
epoch: 18 start
train:
(292.681889410258, 0.96758)
test:
epoch: 18 accuracy 84.600%
epoch: 19 start
train:
(268.3826265888738, 0.97116)
test:
epoch: 19 accuracy 85.460%
epoch: 20 start
train:
(259.78358507735084, 0.97142)
test:
epoch: 20 accuracy 85.950%
epoch: 21 start
train:
(244.3288772199121, 0.97288)
test:
epoch: 21 accuracy 84.750%
epoch: 22 start
train:
(222.62555926712525, 0.97516)
test:
epoch: 22 accuracy 85.870%
epoch: 23 start
train:
(226.58418384475272, 0.97566)
test:
epoch: 23 accuracy 85.220%
epoch: 24 start
train:
(207.31264277165792, 0.97826)
test:
epoch: 24 accuracy 86.220%
epoch: 25 start
train:
(213.51084670384625, 0.97702)
test:
epoch: 25 accuracy 85.250%
epoch: 26 start
train:
(210.42552051352004, 0.97804)
test:
epoch: 26 accuracy 85.290%
epoch: 27 start
train:
(193.14889319763233, 0.97882)
test:
epoch: 27 accuracy 85.870%
epoch: 28 start
train:
(186.1594397107906, 0.98048)
test:
epoch: 28 accuracy 85.700%
epoch: 29 start
train:
(186.02200015645417, 0.9801)
test:
epoch: 29 accuracy 85.710%
epoch: 30 start
train:
(179.4927923081225, 0.98106)
test:
epoch: 30 accuracy 85.840%
best accuracy: 86.220%
--------result report--------
best accuracy:86.220%
loss array:
5179.3005,3141.4500,2191.1988,1737.4292,1450.8914,1206.9403,1020.3975,870.6221,749.8028,645.5848,556.6276,481.4866,428.2829,393.9922,355.3725,333.3651,304.7762,292.6819,268.3826,259.7836,244.3289,222.6256,226.5842,207.3126,213.5108,210.4255,193.1489,186.1594,186.0220,179.4928,
train accuracy array:
0.3948,0.6449,0.7567,0.8068,0.8396,0.8666,0.8876,0.9033,0.9177,0.9274,0.9381,0.9463,0.9535,0.9568,0.9612,0.9641,0.9665,0.9676,0.9712,0.9714,0.9729,0.9752,0.9757,0.9783,0.9770,0.9780,0.9788,0.9805,0.9801,0.9811,
test accuracy array:
0.5520,0.7178,0.7831,0.8087,0.8214,0.8245,0.8325,0.8417,0.8449,0.8498,0.8540,0.8498,0.8521,0.8548,0.8489,0.8477,0.8586,0.8460,0.8546,0.8595,0.8475,0.8587,0.8522,0.8622,0.8525,0.8529,0.8587,0.8570,0.8571,0.8584,
train accuracy array:
0.3948,0.6449,0.7567,0.8068,0.8396,0.8666,0.8876,0.9033,0.9177,0.9274,0.9381,0.9463,0.9535,0.9568,0.9612,0.9641,0.9665,0.9676,0.9712,0.9714,0.9729,0.9752,0.9757,0.9783,0.9770,0.9780,0.9788,0.9805,0.9801,0.9811,
-------------------------------
--------parameter report--------
learning rate:0.01000
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(5207.515186846256, 0.38564)
test:
epoch: 1 accuracy 56.350%
epoch: 2 start
train:
(3241.6993516236544, 0.63306)
test:
epoch: 2 accuracy 67.830%
epoch: 3 start
train:
(2354.45840318501, 0.73768)
test:
epoch: 3 accuracy 77.550%
epoch: 4 start
train:
(1876.2941528856754, 0.79288)
test:
epoch: 4 accuracy 78.850%
epoch: 5 start
train:
(1593.4007045403123, 0.82428)
test:
epoch: 5 accuracy 82.460%
epoch: 6 start
train:
(1366.411079019308, 0.84842)
test:
epoch: 6 accuracy 82.320%
epoch: 7 start
train:
(1175.733315677382, 0.86998)
test:
epoch: 7 accuracy 82.440%
epoch: 8 start
train:
(1027.9974778089672, 0.88614)
test:
epoch: 8 accuracy 84.400%
epoch: 9 start
train:
(889.9181870832108, 0.9005)
test:
epoch: 9 accuracy 85.220%
epoch: 10 start
train:
(779.5481574570294, 0.91384)
test:
epoch: 10 accuracy 84.400%
epoch: 11 start
train:
(681.7461409941316, 0.92364)
test:
epoch: 11 accuracy 85.530%
epoch: 12 start
train:
(607.2566911962349, 0.9325)
test:
epoch: 12 accuracy 84.900%
epoch: 13 start
train:
(549.4555212748528, 0.93874)
test:
epoch: 13 accuracy 84.870%
epoch: 14 start
train:
(488.4100357236239, 0.94552)
test:
epoch: 14 accuracy 85.620%
epoch: 15 start
train:
(467.7797349431967, 0.94882)
test:
epoch: 15 accuracy 83.690%
epoch: 16 start
train:
(403.2130071061256, 0.95498)
test:
epoch: 16 accuracy 85.080%
epoch: 17 start
train:
(383.86473639017277, 0.9587)
test:
epoch: 17 accuracy 84.850%
epoch: 18 start
train:
(351.43306679508714, 0.96152)
test:
epoch: 18 accuracy 85.570%
epoch: 19 start
train:
(328.3181989848199, 0.96418)
test:
epoch: 19 accuracy 84.560%
epoch: 20 start
train:
(323.31669413853524, 0.96474)
test:
epoch: 20 accuracy 85.160%
epoch: 21 start
train:
(307.13597274504355, 0.96748)
test:
epoch: 21 accuracy 85.030%
epoch: 22 start
train:
(277.74518278500136, 0.97018)
test:
epoch: 22 accuracy 85.320%
epoch: 23 start
train:
(278.4385936936792, 0.97006)
test:
epoch: 23 accuracy 85.640%
epoch: 24 start
train:
(254.1247672090035, 0.97306)
test:
epoch: 24 accuracy 86.040%
epoch: 25 start
train:
(246.5933898167641, 0.97354)
test:
epoch: 25 accuracy 85.730%
epoch: 26 start
train:
(244.19121972401354, 0.97396)
test:
epoch: 26 accuracy 86.010%
epoch: 27 start
train:
(230.72513679508938, 0.9753)
test:
epoch: 27 accuracy 85.800%
epoch: 28 start
train:
(237.76230271751365, 0.97386)
test:
epoch: 28 accuracy 86.330%
epoch: 29 start
train:
(212.0920404658309, 0.97694)
test:
epoch: 29 accuracy 86.060%
epoch: 30 start
train:
(204.6535327193983, 0.97746)
test:
epoch: 30 accuracy 84.570%
best accuracy: 86.330%
--------result report--------
best accuracy:86.330%
loss array:
5207.5152,3241.6994,2354.4584,1876.2942,1593.4007,1366.4111,1175.7333,1027.9975,889.9182,779.5482,681.7461,607.2567,549.4555,488.4100,467.7797,403.2130,383.8647,351.4331,328.3182,323.3167,307.1360,277.7452,278.4386,254.1248,246.5934,244.1912,230.7251,237.7623,212.0920,204.6535,
train accuracy array:
0.3856,0.6331,0.7377,0.7929,0.8243,0.8484,0.8700,0.8861,0.9005,0.9138,0.9236,0.9325,0.9387,0.9455,0.9488,0.9550,0.9587,0.9615,0.9642,0.9647,0.9675,0.9702,0.9701,0.9731,0.9735,0.9740,0.9753,0.9739,0.9769,0.9775,
test accuracy array:
0.5635,0.6783,0.7755,0.7885,0.8246,0.8232,0.8244,0.8440,0.8522,0.8440,0.8553,0.8490,0.8487,0.8562,0.8369,0.8508,0.8485,0.8557,0.8456,0.8516,0.8503,0.8532,0.8564,0.8604,0.8573,0.8601,0.8580,0.8633,0.8606,0.8457,
train accuracy array:
0.3856,0.6331,0.7377,0.7929,0.8243,0.8484,0.8700,0.8861,0.9005,0.9138,0.9236,0.9325,0.9387,0.9455,0.9488,0.9550,0.9587,0.9615,0.9642,0.9647,0.9675,0.9702,0.9701,0.9731,0.9735,0.9740,0.9753,0.9739,0.9769,0.9775,
-------------------------------
--------parameter report--------
learning rate:0.01500
epoch:30
train batch size:16
test batch size:16
model:resnet18
optimizer:Adam
-------------------------------
Files already downloaded and verified
Files already downloaded and verified
epoch: 1 start
train:
(5028.00432485342, 0.4143)
test:
epoch: 1 accuracy 56.700%
epoch: 2 start
train:
(3154.79661783576, 0.64284)
test:
epoch: 2 accuracy 67.110%
epoch: 3 start
train:
(2395.1668832600117, 0.73028)
test:
epoch: 3 accuracy 77.080%
epoch: 4 start
train:
(1938.6153901405632, 0.78572)
test:
epoch: 4 accuracy 77.500%
epoch: 5 start
train:
(1666.613046810031, 0.81578)
test:
epoch: 5 accuracy 80.780%
epoch: 6 start
train:
(1446.5718289837241, 0.84106)
test:
epoch: 6 accuracy 81.430%
epoch: 7 start
train:
(1275.460980284959, 0.86054)
test:
epoch: 7 accuracy 82.260%
epoch: 8 start
train:
(1128.8491272269748, 0.87472)
test:
epoch: 8 accuracy 81.970%
epoch: 9 start
train:
(995.9478093739599, 0.89012)
test:
epoch: 9 accuracy 83.510%
epoch: 10 start
train:
(878.3974379845895, 0.90288)
test:
epoch: 10 accuracy 83.680%
epoch: 11 start
train:
(786.128561709309, 0.91266)
test:
epoch: 11 accuracy 84.420%
epoch: 12 start
train:
(706.9773844396695, 0.92166)
test:
epoch: 12 accuracy 83.910%
epoch: 13 start
train:
(626.6114132773364, 0.93036)
test:
epoch: 13 accuracy 84.520%
epoch: 14 start
train:
(570.5483252233826, 0.93758)
test:
epoch: 14 accuracy 84.280%
epoch: 15 start
train:
(525.3479153265653, 0.94214)
test:
epoch: 15 accuracy 84.910%
epoch: 16 start
train:
(481.78365521185333, 0.94778)
test:
epoch: 16 accuracy 85.030%
epoch: 17 start
train:
(449.1916739825974, 0.95078)
test:
epoch: 17 accuracy 85.110%
epoch: 18 start
train:
(415.00350348916254, 0.9543)
test:
epoch: 18 accuracy 84.950%
epoch: 19 start
train:
(382.5326712545648, 0.95852)
test:
epoch: 19 accuracy 85.650%
epoch: 20 start
train:
(364.00468379238737, 0.9601)
test:
epoch: 20 accuracy 85.360%
epoch: 21 start
train:
(342.02667966360605, 0.96148)
test:
epoch: 21 accuracy 85.290%
epoch: 22 start
train:
(317.1484526036984, 0.96456)
test:
epoch: 22 accuracy 85.860%
epoch: 23 start
train:
(300.07296969742674, 0.96736)
test:
epoch: 23 accuracy 84.810%
epoch: 24 start
train:
(301.9608946536173, 0.96696)
test:
epoch: 24 accuracy 85.610%
epoch: 25 start
train:
(286.3885473472001, 0.9692)
test:
epoch: 25 accuracy 85.220%
epoch: 26 start
train:
(263.7404302908476, 0.97154)
test:
epoch: 26 accuracy 85.640%
epoch: 27 start
train:
(254.6305298304419, 0.97226)
test:
epoch: 27 accuracy 85.650%
epoch: 28 start
train:
(258.6396557338103, 0.97242)
test:
epoch: 28 accuracy 86.020%
epoch: 29 start
train:
(240.7975643770733, 0.97394)
test:
epoch: 29 accuracy 85.090%
epoch: 30 start
train:
(223.23623024679682, 0.97584)
test:
epoch: 30 accuracy 85.210%
best accuracy: 86.020%
--------result report--------
best accuracy:86.020%
loss array:
5028.0043,3154.7966,2395.1669,1938.6154,1666.6130,1446.5718,1275.4610,1128.8491,995.9478,878.3974,786.1286,706.9774,626.6114,570.5483,525.3479,481.7837,449.1917,415.0035,382.5327,364.0047,342.0267,317.1485,300.0730,301.9609,286.3885,263.7404,254.6305,258.6397,240.7976,223.2362,
train accuracy array:
0.4143,0.6428,0.7303,0.7857,0.8158,0.8411,0.8605,0.8747,0.8901,0.9029,0.9127,0.9217,0.9304,0.9376,0.9421,0.9478,0.9508,0.9543,0.9585,0.9601,0.9615,0.9646,0.9674,0.9670,0.9692,0.9715,0.9723,0.9724,0.9739,0.9758,
test accuracy array:
0.5670,0.6711,0.7708,0.7750,0.8078,0.8143,0.8226,0.8197,0.8351,0.8368,0.8442,0.8391,0.8452,0.8428,0.8491,0.8503,0.8511,0.8495,0.8565,0.8536,0.8529,0.8586,0.8481,0.8561,0.8522,0.8564,0.8565,0.8602,0.8509,0.8521,
train accuracy array:
0.4143,0.6428,0.7303,0.7857,0.8158,0.8411,0.8605,0.8747,0.8901,0.9029,0.9127,0.9217,0.9304,0.9376,0.9421,0.9478,0.9508,0.9543,0.9585,0.9601,0.9615,0.9646,0.9674,0.9670,0.9692,0.9715,0.9723,0.9724,0.9739,0.9758,
-------------------------------
--------parameter report--------
batchSize:0    accuracy:0.881%
batchSize:0    accuracy:0.876%
batchSize:0    accuracy:0.870%
batchSize:0    accuracy:0.862%
batchSize:0    accuracy:0.863%
batchSize:0    accuracy:0.860%
```