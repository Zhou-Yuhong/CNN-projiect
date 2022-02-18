```
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

if __name__ == '__main__':

    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    #  load
    train_dataset = torchvision.datasets.CIFAR10(root='cifar10origin', train=True, download=True,
                                                 transform=myTransforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    test_dataset = torchvision.datasets.CIFAR10(root='cifar10origin', train=False, download=True,
                                                transform=myTransforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 定义模型
    myModel = torchvision.models.resnet18(pretrained=False)
    # 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
    inchannel = myModel.fc.in_features
    myModel.fc = nn.Linear(inchannel, 10)

    # 损失函数及优化器
    # GPU加速
    myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(myDevice)

    learning_rate = 0.001
    myOptimzier = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
    myLoss = torch.nn.CrossEntropyLoss()

    for _epoch in range(100):
        begin_time = datetime.datetime.now()
        training_loss = 0.0
        for _step, input_data in enumerate(train_loader):
            image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  # GPU加速
            predict_label = myModel.forward(image)

            loss = myLoss(predict_label, label)



            myOptimzier.zero_grad()
            loss.backward()
            myOptimzier.step()

            training_loss = training_loss + loss.item()
            if _step % 100 == 0:
                print(
                    '[iteration - %3d] training loss: %.3f' % (_epoch * len(train_loader) + _step, training_loss / 10))
                training_loss = 0.0
                print()
        correct = 0
        total = 0
        # torch.save(myModel, 'Resnet50_Own.pkl') # 保存整个模型
        myModel.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # GPU加速
                images = images.to(myDevice)
                labels = labels.to(myDevice)
                outputs = myModel(images)  # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
                numbers, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()\

        end_time = datetime.datetime.now()
        print('epoch: %d  seconds:%d' % (_epoch, (end_time-begin_time).seconds))
        print('Testing Accuracy : %.3f %%' % (100 * correct / total))
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to cifar10origin/cifar-10-python.tar.gz
```

170499072/? [00:02<00:00, 88242712.99it/s]

```
Extracting cifar10origin/cifar-10-python.tar.gz to cifar10origin
Files already downloaded and verified
[iteration -   0] training loss: 0.244

[iteration - 100] training loss: 21.171

[iteration - 200] training loss: 18.623

[iteration - 300] training loss: 17.408

[iteration - 400] training loss: 16.587

[iteration - 500] training loss: 16.089

[iteration - 600] training loss: 15.587

[iteration - 700] training loss: 15.075

epoch: 0  seconds:186
Testing Accuracy : 46.480 %
[iteration - 782] training loss: 0.130

[iteration - 882] training loss: 15.967

[iteration - 982] training loss: 15.482

[iteration - 1082] training loss: 14.604

[iteration - 1182] training loss: 14.393

[iteration - 1282] training loss: 13.895

[iteration - 1382] training loss: 14.048

[iteration - 1482] training loss: 13.610

epoch: 1  seconds:179
Testing Accuracy : 52.420 %
[iteration - 1564] training loss: 0.121

[iteration - 1664] training loss: 13.067

[iteration - 1764] training loss: 12.797

[iteration - 1864] training loss: 12.670

[iteration - 1964] training loss: 12.305

[iteration - 2064] training loss: 12.195

[iteration - 2164] training loss: 12.132

[iteration - 2264] training loss: 11.995

epoch: 2  seconds:179
Testing Accuracy : 58.200 %
[iteration - 2346] training loss: 0.105

[iteration - 2446] training loss: 11.553

[iteration - 2546] training loss: 11.252

[iteration - 2646] training loss: 11.275

[iteration - 2746] training loss: 11.108

[iteration - 2846] training loss: 11.453

[iteration - 2946] training loss: 10.846

[iteration - 3046] training loss: 10.730

epoch: 3  seconds:179
Testing Accuracy : 61.190 %
[iteration - 3128] training loss: 0.102

[iteration - 3228] training loss: 10.359

[iteration - 3328] training loss: 10.272

[iteration - 3428] training loss: 10.339

[iteration - 3528] training loss: 10.361

[iteration - 3628] training loss: 10.048

[iteration - 3728] training loss: 9.720

[iteration - 3828] training loss: 9.545

epoch: 4  seconds:179
Testing Accuracy : 64.840 %
[iteration - 3910] training loss: 0.091

[iteration - 4010] training loss: 9.744

[iteration - 4110] training loss: 9.043

[iteration - 4210] training loss: 9.150

[iteration - 4310] training loss: 9.193

[iteration - 4410] training loss: 9.168

[iteration - 4510] training loss: 9.122

[iteration - 4610] training loss: 8.820

epoch: 5  seconds:179
Testing Accuracy : 65.610 %
[iteration - 4692] training loss: 0.096

[iteration - 4792] training loss: 8.871

[iteration - 4892] training loss: 8.555

[iteration - 4992] training loss: 8.198

[iteration - 5092] training loss: 8.758

[iteration - 5192] training loss: 8.126

[iteration - 5292] training loss: 8.530

[iteration - 5392] training loss: 8.423

epoch: 6  seconds:179
Testing Accuracy : 65.910 %
[iteration - 5474] training loss: 0.077

[iteration - 5574] training loss: 7.865

[iteration - 5674] training loss: 7.823

[iteration - 5774] training loss: 8.098

[iteration - 5874] training loss: 7.774

[iteration - 5974] training loss: 7.576

[iteration - 6074] training loss: 7.689

[iteration - 6174] training loss: 7.661

epoch: 7  seconds:179
Testing Accuracy : 72.430 %
[iteration - 6256] training loss: 0.062

[iteration - 6356] training loss: 7.498

[iteration - 6456] training loss: 7.528

[iteration - 6556] training loss: 7.193

[iteration - 6656] training loss: 7.143

[iteration - 6756] training loss: 7.280

[iteration - 6856] training loss: 7.100

[iteration - 6956] training loss: 6.841

epoch: 8  seconds:179
Testing Accuracy : 70.530 %
[iteration - 7038] training loss: 0.068

[iteration - 7138] training loss: 6.676

[iteration - 7238] training loss: 6.679

[iteration - 7338] training loss: 6.789

[iteration - 7438] training loss: 6.830

[iteration - 7538] training loss: 6.764

[iteration - 7638] training loss: 6.743

[iteration - 7738] training loss: 6.604

epoch: 9  seconds:179
Testing Accuracy : 74.610 %
[iteration - 7820] training loss: 0.052

[iteration - 7920] training loss: 6.206

[iteration - 8020] training loss: 6.408

[iteration - 8120] training loss: 6.194

[iteration - 8220] training loss: 6.564

[iteration - 8320] training loss: 6.307

[iteration - 8420] training loss: 6.305

[iteration - 8520] training loss: 6.178

epoch: 10  seconds:179
Testing Accuracy : 73.550 %
[iteration - 8602] training loss: 0.084

[iteration - 8702] training loss: 5.771

[iteration - 8802] training loss: 5.850

[iteration - 8902] training loss: 6.078

[iteration - 9002] training loss: 5.702

[iteration - 9102] training loss: 5.854

[iteration - 9202] training loss: 5.831

[iteration - 9302] training loss: 5.933

epoch: 11  seconds:179
Testing Accuracy : 78.220 %
[iteration - 9384] training loss: 0.038

[iteration - 9484] training loss: 5.611

[iteration - 9584] training loss: 5.313

[iteration - 9684] training loss: 5.561

[iteration - 9784] training loss: 5.566

[iteration - 9884] training loss: 5.403

[iteration - 9984] training loss: 5.420

[iteration - 10084] training loss: 5.294

epoch: 12  seconds:178
Testing Accuracy : 75.090 %
[iteration - 10166] training loss: 0.073

[iteration - 10266] training loss: 4.945

[iteration - 10366] training loss: 4.803

[iteration - 10466] training loss: 5.078

[iteration - 10566] training loss: 5.029

[iteration - 10666] training loss: 5.008

[iteration - 10766] training loss: 5.166

[iteration - 10866] training loss: 5.041

epoch: 13  seconds:180
Testing Accuracy : 78.510 %
[iteration - 10948] training loss: 0.037

[iteration - 11048] training loss: 4.970

[iteration - 11148] training loss: 4.726

[iteration - 11248] training loss: 4.769

[iteration - 11348] training loss: 4.323

[iteration - 11448] training loss: 4.748

[iteration - 11548] training loss: 4.674

[iteration - 11648] training loss: 4.364

epoch: 14  seconds:179
Testing Accuracy : 78.780 %
[iteration - 11730] training loss: 0.050

[iteration - 11830] training loss: 4.030

[iteration - 11930] training loss: 4.264

[iteration - 12030] training loss: 4.234

[iteration - 12130] training loss: 4.273

[iteration - 12230] training loss: 4.327

[iteration - 12330] training loss: 4.538

[iteration - 12430] training loss: 4.268

epoch: 15  seconds:179
Testing Accuracy : 74.680 %
[iteration - 12512] training loss: 0.050

[iteration - 12612] training loss: 3.945

[iteration - 12712] training loss: 4.056

[iteration - 12812] training loss: 3.929

[iteration - 12912] training loss: 3.742

[iteration - 13012] training loss: 3.959

[iteration - 13112] training loss: 3.658

[iteration - 13212] training loss: 4.029

epoch: 16  seconds:179
Testing Accuracy : 76.860 %
[iteration - 13294] training loss: 0.031

[iteration - 13394] training loss: 3.433

[iteration - 13494] training loss: 3.643

[iteration - 13594] training loss: 3.623

[iteration - 13694] training loss: 3.699

[iteration - 13794] training loss: 3.872

[iteration - 13894] training loss: 3.557

[iteration - 13994] training loss: 3.691

epoch: 17  seconds:179
Testing Accuracy : 80.500 %
[iteration - 14076] training loss: 0.025

[iteration - 14176] training loss: 3.085

[iteration - 14276] training loss: 3.207

[iteration - 14376] training loss: 3.638

[iteration - 14476] training loss: 3.431

[iteration - 14576] training loss: 3.383

[iteration - 14676] training loss: 3.501

[iteration - 14776] training loss: 3.675

epoch: 18  seconds:179
Testing Accuracy : 78.020 %
[iteration - 14858] training loss: 0.035

[iteration - 14958] training loss: 2.936

[iteration - 15058] training loss: 3.126

[iteration - 15158] training loss: 2.829

[iteration - 15258] training loss: 3.075

[iteration - 15358] training loss: 3.107

[iteration - 15458] training loss: 3.220

[iteration - 15558] training loss: 3.062

epoch: 19  seconds:178
Testing Accuracy : 80.500 %
[iteration - 15640] training loss: 0.019

[iteration - 15740] training loss: 3.001

[iteration - 15840] training loss: 2.448

[iteration - 15940] training loss: 3.081

[iteration - 16040] training loss: 2.865

[iteration - 16140] training loss: 2.565

[iteration - 16240] training loss: 2.808

[iteration - 16340] training loss: 2.898

epoch: 20  seconds:179
Testing Accuracy : 80.100 %
[iteration - 16422] training loss: 0.017

[iteration - 16522] training loss: 2.769

[iteration - 16622] training loss: 2.296

[iteration - 16722] training loss: 2.609

[iteration - 16822] training loss: 2.239

[iteration - 16922] training loss: 2.406

[iteration - 17022] training loss: 2.629

[iteration - 17122] training loss: 2.529

epoch: 21  seconds:180
Testing Accuracy : 80.100 %
[iteration - 17204] training loss: 0.024

[iteration - 17304] training loss: 2.262

[iteration - 17404] training loss: 2.373

[iteration - 17504] training loss: 2.025

[iteration - 17604] training loss: 2.293

[iteration - 17704] training loss: 2.168

[iteration - 17804] training loss: 2.144

[iteration - 17904] training loss: 2.382

epoch: 22  seconds:179
Testing Accuracy : 82.210 %
[iteration - 17986] training loss: 0.011

[iteration - 18086] training loss: 1.768

[iteration - 18186] training loss: 1.681

[iteration - 18286] training loss: 2.065

[iteration - 18386] training loss: 2.079

[iteration - 18486] training loss: 2.217

[iteration - 18586] training loss: 2.107

[iteration - 18686] training loss: 2.174

epoch: 23  seconds:179
Testing Accuracy : 79.290 %
[iteration - 18768] training loss: 0.010

[iteration - 18868] training loss: 1.939

[iteration - 18968] training loss: 1.626

[iteration - 19068] training loss: 1.822

[iteration - 19168] training loss: 1.525

[iteration - 19268] training loss: 1.804

[iteration - 19368] training loss: 2.022

[iteration - 19468] training loss: 1.834

epoch: 24  seconds:180
Testing Accuracy : 81.010 %
[iteration - 19550] training loss: 0.025

[iteration - 19650] training loss: 1.399

[iteration - 19750] training loss: 1.602

[iteration - 19850] training loss: 1.617

[iteration - 19950] training loss: 1.879

[iteration - 20050] training loss: 1.655

[iteration - 20150] training loss: 1.483

[iteration - 20250] training loss: 1.860

epoch: 25  seconds:179
Testing Accuracy : 80.840 %
[iteration - 20332] training loss: 0.013

[iteration - 20432] training loss: 1.528

[iteration - 20532] training loss: 1.362

[iteration - 20632] training loss: 1.495

[iteration - 20732] training loss: 1.569

[iteration - 20832] training loss: 1.723

[iteration - 20932] training loss: 1.298

[iteration - 21032] training loss: 1.400

epoch: 26  seconds:180
Testing Accuracy : 79.210 %
[iteration - 21114] training loss: 0.018

[iteration - 21214] training loss: 1.157

[iteration - 21314] training loss: 1.447

[iteration - 21414] training loss: 1.209

[iteration - 21514] training loss: 1.096

[iteration - 21614] training loss: 1.132

[iteration - 21714] training loss: 1.224

[iteration - 21814] training loss: 1.260

epoch: 27  seconds:179
Testing Accuracy : 81.770 %
[iteration - 21896] training loss: 0.010

[iteration - 21996] training loss: 1.084

[iteration - 22096] training loss: 1.190

[iteration - 22196] training loss: 1.204

[iteration - 22296] training loss: 1.010

[iteration - 22396] training loss: 1.131

[iteration - 22496] training loss: 1.358

[iteration - 22596] training loss: 1.453

epoch: 28  seconds:180
Testing Accuracy : 81.300 %
[iteration - 22678] training loss: 0.019

[iteration - 22778] training loss: 1.155

[iteration - 22878] training loss: 1.007

[iteration - 22978] training loss: 0.872

[iteration - 23078] training loss: 1.188

[iteration - 23178] training loss: 1.087

[iteration - 23278] training loss: 0.889

[iteration - 23378] training loss: 0.927

epoch: 29  seconds:179
Testing Accuracy : 81.480 %
[iteration - 23460] training loss: 0.004

[iteration - 23560] training loss: 1.159

[iteration - 23660] training loss: 0.550

[iteration - 23760] training loss: 1.294

[iteration - 23860] training loss: 0.995

[iteration - 23960] training loss: 0.883

[iteration - 24060] training loss: 0.956

[iteration - 24160] training loss: 1.023

epoch: 30  seconds:180
Testing Accuracy : 81.710 %
[iteration - 24242] training loss: 0.002

[iteration - 24342] training loss: 0.688

[iteration - 24442] training loss: 0.870

[iteration - 24542] training loss: 0.769

[iteration - 24642] training loss: 0.730

[iteration - 24742] training loss: 0.806

[iteration - 24842] training loss: 0.725

[iteration - 24942] training loss: 0.809

epoch: 31  seconds:180
Testing Accuracy : 79.550 %
[iteration - 25024] training loss: 0.020

[iteration - 25124] training loss: 0.886

[iteration - 25224] training loss: 0.839

[iteration - 25324] training loss: 0.881

[iteration - 25424] training loss: 0.759

[iteration - 25524] training loss: 0.517

[iteration - 25624] training loss: 0.706

[iteration - 25724] training loss: 0.811

epoch: 32  seconds:180
Testing Accuracy : 81.900 %
[iteration - 25806] training loss: 0.001

[iteration - 25906] training loss: 0.615

[iteration - 26006] training loss: 0.479

[iteration - 26106] training loss: 1.019

[iteration - 26206] training loss: 0.673

[iteration - 26306] training loss: 0.744

[iteration - 26406] training loss: 0.654

[iteration - 26506] training loss: 1.061

epoch: 33  seconds:179
Testing Accuracy : 81.750 %
[iteration - 26588] training loss: 0.004

[iteration - 26688] training loss: 0.481

[iteration - 26788] training loss: 0.405

[iteration - 26888] training loss: 0.763

[iteration - 26988] training loss: 0.734

[iteration - 27088] training loss: 0.562

[iteration - 27188] training loss: 0.571

[iteration - 27288] training loss: 0.944

epoch: 34  seconds:180
Testing Accuracy : 81.880 %
[iteration - 27370] training loss: 0.001

[iteration - 27470] training loss: 0.412

[iteration - 27570] training loss: 0.520

[iteration - 27670] training loss: 0.648

[iteration - 27770] training loss: 0.571

[iteration - 27870] training loss: 0.695

[iteration - 27970] training loss: 0.804

[iteration - 28070] training loss: 0.683

epoch: 35  seconds:179
Testing Accuracy : 81.510 %
[iteration - 28152] training loss: 0.003

[iteration - 28252] training loss: 0.500

[iteration - 28352] training loss: 0.409

[iteration - 28452] training loss: 0.534

[iteration - 28552] training loss: 0.397

[iteration - 28652] training loss: 0.388

[iteration - 28752] training loss: 0.448

[iteration - 28852] training loss: 1.057

epoch: 36  seconds:179
Testing Accuracy : 81.670 %
[iteration - 28934] training loss: 0.003

[iteration - 29034] training loss: 0.350

[iteration - 29134] training loss: 0.357

[iteration - 29234] training loss: 0.492

[iteration - 29334] training loss: 0.552

[iteration - 29434] training loss: 0.437

[iteration - 29534] training loss: 0.790

[iteration - 29634] training loss: 0.390

epoch: 37  seconds:180
Testing Accuracy : 81.560 %
[iteration - 29716] training loss: 0.002

[iteration - 29816] training loss: 0.966

[iteration - 29916] training loss: 0.510

[iteration - 30016] training loss: 0.331

[iteration - 30116] training loss: 0.298

[iteration - 30216] training loss: 0.308

[iteration - 30316] training loss: 0.488

[iteration - 30416] training loss: 0.436

epoch: 38  seconds:179
Testing Accuracy : 79.620 %
[iteration - 30498] training loss: 0.006

[iteration - 30598] training loss: 0.767

[iteration - 30698] training loss: 0.315

[iteration - 30798] training loss: 0.504

[iteration - 30898] training loss: 0.502

[iteration - 30998] training loss: 0.267

[iteration - 31098] training loss: 0.651

[iteration - 31198] training loss: 0.479

epoch: 39  seconds:180
Testing Accuracy : 81.120 %
[iteration - 31280] training loss: 0.004

[iteration - 31380] training loss: 0.270

[iteration - 31480] training loss: 0.309

[iteration - 31580] training loss: 0.377

[iteration - 31680] training loss: 0.288

[iteration - 31780] training loss: 0.332

[iteration - 31880] training loss: 0.634

[iteration - 31980] training loss: 0.303

epoch: 40  seconds:179
Testing Accuracy : 82.300 %
[iteration - 32062] training loss: 0.002

[iteration - 32162] training loss: 0.312

[iteration - 32262] training loss: 0.210

[iteration - 32362] training loss: 0.400

[iteration - 32462] training loss: 0.335

[iteration - 32562] training loss: 0.420

[iteration - 32662] training loss: 0.454

[iteration - 32762] training loss: 0.319

epoch: 41  seconds:179
Testing Accuracy : 82.510 %
[iteration - 32844] training loss: 0.001

[iteration - 32944] training loss: 0.656

[iteration - 33044] training loss: 0.346

[iteration - 33144] training loss: 0.424

[iteration - 33244] training loss: 0.507

[iteration - 33344] training loss: 0.588

[iteration - 33444] training loss: 0.550

[iteration - 33544] training loss: 0.405

epoch: 42  seconds:179
Testing Accuracy : 82.610 %
[iteration - 33626] training loss: 0.002

[iteration - 33726] training loss: 0.328

[iteration - 33826] training loss: 0.324

[iteration - 33926] training loss: 0.235

[iteration - 34026] training loss: 0.251

[iteration - 34126] training loss: 0.256

[iteration - 34226] training loss: 0.180

[iteration - 34326] training loss: 0.542

epoch: 43  seconds:180
Testing Accuracy : 81.340 %
[iteration - 34408] training loss: 0.013

[iteration - 34508] training loss: 0.404

[iteration - 34608] training loss: 0.234

[iteration - 34708] training loss: 0.432

[iteration - 34808] training loss: 0.327

[iteration - 34908] training loss: 0.442

[iteration - 35008] training loss: 0.278

[iteration - 35108] training loss: 0.408

epoch: 44  seconds:180
Testing Accuracy : 81.740 %
[iteration - 35190] training loss: 0.005

[iteration - 35290] training loss: 0.216

[iteration - 35390] training loss: 0.195

[iteration - 35490] training loss: 0.154

[iteration - 35590] training loss: 0.250

[iteration - 35690] training loss: 0.244

[iteration - 35790] training loss: 0.503

[iteration - 35890] training loss: 0.397

epoch: 45  seconds:180
Testing Accuracy : 83.580 %
[iteration - 35972] training loss: 0.001

[iteration - 36072] training loss: 0.230

[iteration - 36172] training loss: 0.121

[iteration - 36272] training loss: 0.101

[iteration - 36372] training loss: 0.153

[iteration - 36472] training loss: 0.246

[iteration - 36572] training loss: 0.270

[iteration - 36672] training loss: 0.360

epoch: 46  seconds:180
Testing Accuracy : 82.870 %
[iteration - 36754] training loss: 0.001

[iteration - 36854] training loss: 0.090

[iteration - 36954] training loss: 0.082

[iteration - 37054] training loss: 0.167

[iteration - 37154] training loss: 0.145

[iteration - 37254] training loss: 0.310

[iteration - 37354] training loss: 0.198

[iteration - 37454] training loss: 0.289

epoch: 47  seconds:180
Testing Accuracy : 81.570 %
[iteration - 37536] training loss: 0.007

[iteration - 37636] training loss: 0.469

[iteration - 37736] training loss: 0.455

[iteration - 37836] training loss: 0.146

[iteration - 37936] training loss: 0.142

[iteration - 38036] training loss: 0.153

[iteration - 38136] training loss: 0.171

[iteration - 38236] training loss: 0.288

epoch: 48  seconds:179
Testing Accuracy : 81.410 %
[iteration - 38318] training loss: 0.005

[iteration - 38418] training loss: 0.355

[iteration - 38518] training loss: 0.495

[iteration - 38618] training loss: 0.214

[iteration - 38718] training loss: 0.269

[iteration - 38818] training loss: 0.157

[iteration - 38918] training loss: 0.192

[iteration - 39018] training loss: 0.211

epoch: 49  seconds:179
Testing Accuracy : 81.420 %
[iteration - 39100] training loss: 0.002

[iteration - 39200] training loss: 0.557

[iteration - 39300] training loss: 0.085

[iteration - 39400] training loss: 0.123

[iteration - 39500] training loss: 0.235

[iteration - 39600] training loss: 0.225

[iteration - 39700] training loss: 0.595

[iteration - 39800] training loss: 0.399

epoch: 50  seconds:181
Testing Accuracy : 83.600 %
[iteration - 39882] training loss: 0.000

[iteration - 39982] training loss: 0.160

[iteration - 40082] training loss: 0.070

[iteration - 40182] training loss: 0.030

[iteration - 40282] training loss: 0.030

[iteration - 40382] training loss: 0.148

[iteration - 40482] training loss: 0.089

[iteration - 40582] training loss: 0.103

epoch: 51  seconds:180
Testing Accuracy : 80.440 %
[iteration - 40664] training loss: 0.005

[iteration - 40764] training loss: 0.199

[iteration - 40864] training loss: 0.267

[iteration - 40964] training loss: 0.078

[iteration - 41064] training loss: 0.217

[iteration - 41164] training loss: 0.246

[iteration - 41264] training loss: 0.466

[iteration - 41364] training loss: 0.243

epoch: 52  seconds:179
Testing Accuracy : 82.330 %
[iteration - 41446] training loss: 0.001

[iteration - 41546] training loss: 0.302

[iteration - 41646] training loss: 0.182

[iteration - 41746] training loss: 0.078

[iteration - 41846] training loss: 0.100

[iteration - 41946] training loss: 0.067

[iteration - 42046] training loss: 0.266

[iteration - 42146] training loss: 0.214

epoch: 53  seconds:180
Testing Accuracy : 83.220 %
[iteration - 42228] training loss: 0.002

[iteration - 42328] training loss: 0.100

[iteration - 42428] training loss: 0.123

[iteration - 42528] training loss: 0.045

[iteration - 42628] training loss: 0.101

[iteration - 42728] training loss: 0.062

[iteration - 42828] training loss: 0.061

[iteration - 42928] training loss: 0.036

epoch: 54  seconds:179
Testing Accuracy : 81.920 %
[iteration - 43010] training loss: 0.002

[iteration - 43110] training loss: 0.168

[iteration - 43210] training loss: 0.090

[iteration - 43310] training loss: 0.242

[iteration - 43410] training loss: 0.349

[iteration - 43510] training loss: 0.276

[iteration - 43610] training loss: 0.216

[iteration - 43710] training loss: 0.144

epoch: 55  seconds:180
Testing Accuracy : 83.310 %
[iteration - 43792] training loss: 0.001

[iteration - 43892] training loss: 0.129

[iteration - 43992] training loss: 0.176

[iteration - 44092] training loss: 0.140

[iteration - 44192] training loss: 0.171

[iteration - 44292] training loss: 0.205

[iteration - 44392] training loss: 0.423

[iteration - 44492] training loss: 0.203

epoch: 56  seconds:180
Testing Accuracy : 83.050 %
[iteration - 44574] training loss: 0.000

[iteration - 44674] training loss: 0.099

[iteration - 44774] training loss: 0.053

[iteration - 44874] training loss: 0.075

[iteration - 44974] training loss: 0.146

[iteration - 45074] training loss: 0.161

[iteration - 45174] training loss: 0.237

[iteration - 45274] training loss: 0.169

epoch: 57  seconds:180
Testing Accuracy : 82.800 %
[iteration - 45356] training loss: 0.002

[iteration - 45456] training loss: 0.099

[iteration - 45556] training loss: 0.221

[iteration - 45656] training loss: 0.174

[iteration - 45756] training loss: 0.196

[iteration - 45856] training loss: 0.100

[iteration - 45956] training loss: 0.181

[iteration - 46056] training loss: 0.199

epoch: 58  seconds:181
Testing Accuracy : 83.100 %
[iteration - 46138] training loss: 0.000

[iteration - 46238] training loss: 0.387

[iteration - 46338] training loss: 0.060

[iteration - 46438] training loss: 0.028

[iteration - 46538] training loss: 0.080

[iteration - 46638] training loss: 0.059

[iteration - 46738] training loss: 0.060

[iteration - 46838] training loss: 0.072

epoch: 59  seconds:180
Testing Accuracy : 83.370 %
[iteration - 46920] training loss: 0.000

[iteration - 47020] training loss: 0.127

[iteration - 47120] training loss: 0.164

[iteration - 47220] training loss: 0.131

[iteration - 47320] training loss: 0.183

[iteration - 47420] training loss: 0.111

[iteration - 47520] training loss: 0.212

[iteration - 47620] training loss: 0.152

epoch: 60  seconds:179
Testing Accuracy : 83.900 %
[iteration - 47702] training loss: 0.001

[iteration - 47802] training loss: 0.024

[iteration - 47902] training loss: 0.046

[iteration - 48002] training loss: 0.110

[iteration - 48102] training loss: 0.258

[iteration - 48202] training loss: 0.326

[iteration - 48302] training loss: 0.198

[iteration - 48402] training loss: 0.209

epoch: 61  seconds:179
Testing Accuracy : 83.360 %
[iteration - 48484] training loss: 0.000

[iteration - 48584] training loss: 0.057

[iteration - 48684] training loss: 0.036

[iteration - 48784] training loss: 0.103

[iteration - 48884] training loss: 0.238

[iteration - 48984] training loss: 0.123

[iteration - 49084] training loss: 0.091

[iteration - 49184] training loss: 0.181

epoch: 62  seconds:179
Testing Accuracy : 82.400 %
[iteration - 49266] training loss: 0.009

[iteration - 49366] training loss: 0.528

[iteration - 49466] training loss: 0.200

[iteration - 49566] training loss: 0.077

[iteration - 49666] training loss: 0.099

[iteration - 49766] training loss: 0.153

[iteration - 49866] training loss: 0.057

[iteration - 49966] training loss: 0.131

epoch: 63  seconds:179
Testing Accuracy : 82.750 %
[iteration - 50048] training loss: 0.000

[iteration - 50148] training loss: 0.149

[iteration - 50248] training loss: 0.165

[iteration - 50348] training loss: 0.054

[iteration - 50448] training loss: 0.048

[iteration - 50548] training loss: 0.118

[iteration - 50648] training loss: 0.062

[iteration - 50748] training loss: 0.124

epoch: 64  seconds:180
Testing Accuracy : 83.180 %
[iteration - 50830] training loss: 0.000

[iteration - 50930] training loss: 0.169

[iteration - 51030] training loss: 0.056

[iteration - 51130] training loss: 0.284

[iteration - 51230] training loss: 0.207

[iteration - 51330] training loss: 0.060

[iteration - 51430] training loss: 0.129

[iteration - 51530] training loss: 0.306

epoch: 65  seconds:180
Testing Accuracy : 83.930 %
[iteration - 51612] training loss: 0.004

[iteration - 51712] training loss: 0.061

[iteration - 51812] training loss: 0.124

[iteration - 51912] training loss: 0.119

[iteration - 52012] training loss: 0.068

[iteration - 52112] training loss: 0.431

[iteration - 52212] training loss: 0.348

[iteration - 52312] training loss: 0.238

epoch: 66  seconds:180
Testing Accuracy : 83.920 %
[iteration - 52394] training loss: 0.000

[iteration - 52494] training loss: 0.040

[iteration - 52594] training loss: 0.018

[iteration - 52694] training loss: 0.010

[iteration - 52794] training loss: 0.006

[iteration - 52894] training loss: 0.022

[iteration - 52994] training loss: 0.150

[iteration - 53094] training loss: 0.168

epoch: 67  seconds:180
Testing Accuracy : 82.170 %
[iteration - 53176] training loss: 0.002

[iteration - 53276] training loss: 0.045

[iteration - 53376] training loss: 0.023

[iteration - 53476] training loss: 0.109

[iteration - 53576] training loss: 0.039

[iteration - 53676] training loss: 0.034

[iteration - 53776] training loss: 0.088

[iteration - 53876] training loss: 0.157

epoch: 68  seconds:180
Testing Accuracy : 83.720 %
[iteration - 53958] training loss: 0.000

[iteration - 54058] training loss: 0.105

[iteration - 54158] training loss: 0.106

[iteration - 54258] training loss: 0.052

[iteration - 54358] training loss: 0.010

[iteration - 54458] training loss: 0.017

[iteration - 54558] training loss: 0.032

[iteration - 54658] training loss: 0.030

epoch: 69  seconds:180
Testing Accuracy : 82.920 %
[iteration - 54740] training loss: 0.004

[iteration - 54840] training loss: 0.022

[iteration - 54940] training loss: 0.043

[iteration - 55040] training loss: 0.007

[iteration - 55140] training loss: 0.008

[iteration - 55240] training loss: 0.038

[iteration - 55340] training loss: 0.418

[iteration - 55440] training loss: 0.112

epoch: 70  seconds:180
Testing Accuracy : 82.500 %
[iteration - 55522] training loss: 0.004

[iteration - 55622] training loss: 0.575

[iteration - 55722] training loss: 0.115

[iteration - 55822] training loss: 0.099

[iteration - 55922] training loss: 0.122

[iteration - 56022] training loss: 0.161

[iteration - 56122] training loss: 0.041

[iteration - 56222] training loss: 0.065

epoch: 71  seconds:180
Testing Accuracy : 84.480 %
[iteration - 56304] training loss: 0.000

[iteration - 56404] training loss: 0.011

[iteration - 56504] training loss: 0.013

[iteration - 56604] training loss: 0.007

[iteration - 56704] training loss: 0.004

[iteration - 56804] training loss: 0.003

[iteration - 56904] training loss: 0.010

[iteration - 57004] training loss: 0.003

epoch: 72  seconds:180
Testing Accuracy : 84.960 %
[iteration - 57086] training loss: 0.000

[iteration - 57186] training loss: 0.019

[iteration - 57286] training loss: 0.014

[iteration - 57386] training loss: 0.019

[iteration - 57486] training loss: 0.006

[iteration - 57586] training loss: 0.010

[iteration - 57686] training loss: 0.004

[iteration - 57786] training loss: 0.003

epoch: 73  seconds:180
Testing Accuracy : 84.810 %
[iteration - 57868] training loss: 0.000

[iteration - 57968] training loss: 0.002

[iteration - 58068] training loss: 0.002

[iteration - 58168] training loss: 0.002

[iteration - 58268] training loss: 0.001

[iteration - 58368] training loss: 0.001

[iteration - 58468] training loss: 0.001

[iteration - 58568] training loss: 0.002

epoch: 74  seconds:179
Testing Accuracy : 84.980 %
[iteration - 58650] training loss: 0.000

[iteration - 58750] training loss: 0.001

[iteration - 58850] training loss: 0.001

[iteration - 58950] training loss: 0.001

[iteration - 59050] training loss: 0.001

[iteration - 59150] training loss: 0.001

[iteration - 59250] training loss: 0.001

[iteration - 59350] training loss: 0.001

epoch: 75  seconds:180
Testing Accuracy : 84.670 %
[iteration - 59432] training loss: 0.000

[iteration - 59532] training loss: 0.001

[iteration - 59632] training loss: 0.001

[iteration - 59732] training loss: 0.000

[iteration - 59832] training loss: 0.000

[iteration - 59932] training loss: 0.000

[iteration - 60032] training loss: 0.000

[iteration - 60132] training loss: 0.001

epoch: 76  seconds:180
Testing Accuracy : 85.240 %
[iteration - 60214] training loss: 0.000

[iteration - 60314] training loss: 0.000

[iteration - 60414] training loss: 0.000

[iteration - 60514] training loss: 0.000

[iteration - 60614] training loss: 0.000

[iteration - 60714] training loss: 0.000

[iteration - 60814] training loss: 0.000

[iteration - 60914] training loss: 0.000

epoch: 77  seconds:179
Testing Accuracy : 85.040 %
[iteration - 60996] training loss: 0.000

[iteration - 61096] training loss: 0.000

[iteration - 61196] training loss: 0.000

[iteration - 61296] training loss: 0.000

[iteration - 61396] training loss: 0.000

[iteration - 61496] training loss: 0.000

[iteration - 61596] training loss: 0.000

[iteration - 61696] training loss: 0.000

epoch: 78  seconds:180
Testing Accuracy : 84.990 %
[iteration - 61778] training loss: 0.000

[iteration - 61878] training loss: 0.000

[iteration - 61978] training loss: 0.000

[iteration - 62078] training loss: 0.000

[iteration - 62178] training loss: 0.000

[iteration - 62278] training loss: 0.000

[iteration - 62378] training loss: 0.000

[iteration - 62478] training loss: 0.000

epoch: 79  seconds:181
Testing Accuracy : 84.990 %
[iteration - 62560] training loss: 0.000

[iteration - 62660] training loss: 0.000

[iteration - 62760] training loss: 0.000

[iteration - 62860] training loss: 0.000

[iteration - 62960] training loss: 0.000

[iteration - 63060] training loss: 0.000

[iteration - 63160] training loss: 0.000

[iteration - 63260] training loss: 0.000

epoch: 80  seconds:179
Testing Accuracy : 85.130 %
[iteration - 63342] training loss: 0.000

[iteration - 63442] training loss: 0.000

[iteration - 63542] training loss: 0.000

[iteration - 63642] training loss: 0.000

[iteration - 63742] training loss: 0.000

[iteration - 63842] training loss: 0.000

[iteration - 63942] training loss: 0.000

[iteration - 64042] training loss: 0.000

epoch: 81  seconds:179
Testing Accuracy : 84.950 %
[iteration - 64124] training loss: 0.000

[iteration - 64224] training loss: 0.000

[iteration - 64324] training loss: 0.000

[iteration - 64424] training loss: 0.000

[iteration - 64524] training loss: 0.000

[iteration - 64624] training loss: 0.000

[iteration - 64724] training loss: 0.000

[iteration - 64824] training loss: 0.000

epoch: 82  seconds:179
Testing Accuracy : 85.310 %
[iteration - 64906] training loss: 0.000

[iteration - 65006] training loss: 0.000

[iteration - 65106] training loss: 0.000

[iteration - 65206] training loss: 0.000

[iteration - 65306] training loss: 0.000

[iteration - 65406] training loss: 0.000

[iteration - 65506] training loss: 0.000

[iteration - 65606] training loss: 0.000

epoch: 83  seconds:178
Testing Accuracy : 85.020 %
[iteration - 65688] training loss: 0.000

[iteration - 65788] training loss: 0.000

[iteration - 65888] training loss: 0.000

[iteration - 65988] training loss: 0.000

[iteration - 66088] training loss: 0.000

[iteration - 66188] training loss: 0.000

[iteration - 66288] training loss: 0.000

[iteration - 66388] training loss: 0.000

epoch: 84  seconds:178
Testing Accuracy : 85.280 %
[iteration - 66470] training loss: 0.000

[iteration - 66570] training loss: 0.000

[iteration - 66670] training loss: 0.000

[iteration - 66770] training loss: 0.000

[iteration - 66870] training loss: 0.000

[iteration - 66970] training loss: 0.000

[iteration - 67070] training loss: 0.000

[iteration - 67170] training loss: 0.000

epoch: 85  seconds:179
Testing Accuracy : 84.770 %
[iteration - 67252] training loss: 0.000

[iteration - 67352] training loss: 0.000

[iteration - 67452] training loss: 0.000

[iteration - 67552] training loss: 0.000

[iteration - 67652] training loss: 0.000

[iteration - 67752] training loss: 0.000

[iteration - 67852] training loss: 0.000

[iteration - 67952] training loss: 0.000

epoch: 86  seconds:179
Testing Accuracy : 85.040 %
[iteration - 68034] training loss: 0.000

[iteration - 68134] training loss: 0.000

[iteration - 68234] training loss: 0.000

[iteration - 68334] training loss: 0.000

[iteration - 68434] training loss: 0.000

[iteration - 68534] training loss: 0.000

[iteration - 68634] training loss: 0.000

[iteration - 68734] training loss: 0.000

epoch: 87  seconds:179
Testing Accuracy : 85.110 %
[iteration - 68816] training loss: 0.000

[iteration - 68916] training loss: 0.000

[iteration - 69016] training loss: 0.000

[iteration - 69116] training loss: 0.000

[iteration - 69216] training loss: 0.000

[iteration - 69316] training loss: 0.000

[iteration - 69416] training loss: 0.000

[iteration - 69516] training loss: 0.000

epoch: 88  seconds:178
Testing Accuracy : 84.890 %
[iteration - 69598] training loss: 0.000

[iteration - 69698] training loss: 0.000

[iteration - 69798] training loss: 0.000

[iteration - 69898] training loss: 0.000

[iteration - 69998] training loss: 0.000

[iteration - 70098] training loss: 0.000

[iteration - 70198] training loss: 0.000

[iteration - 70298] training loss: 0.000

epoch: 89  seconds:179
Testing Accuracy : 85.100 %
[iteration - 70380] training loss: 0.000

[iteration - 70480] training loss: 0.000

[iteration - 70580] training loss: 0.000

[iteration - 70680] training loss: 0.000

[iteration - 70780] training loss: 0.000

[iteration - 70880] training loss: 0.000

[iteration - 70980] training loss: 0.000

[iteration - 71080] training loss: 0.000

epoch: 90  seconds:179
Testing Accuracy : 85.110 %
[iteration - 71162] training loss: 0.000

[iteration - 71262] training loss: 0.000

[iteration - 71362] training loss: 0.000

[iteration - 71462] training loss: 0.000

[iteration - 71562] training loss: 0.000

[iteration - 71662] training loss: 0.000

[iteration - 71762] training loss: 0.000

[iteration - 71862] training loss: 0.000

epoch: 91  seconds:178
Testing Accuracy : 84.820 %
[iteration - 71944] training loss: 0.000

[iteration - 72044] training loss: 0.000

[iteration - 72144] training loss: 0.000

[iteration - 72244] training loss: 0.000

[iteration - 72344] training loss: 0.000

[iteration - 72444] training loss: 0.000

[iteration - 72544] training loss: 0.000

[iteration - 72644] training loss: 0.000

epoch: 92  seconds:178
Testing Accuracy : 85.250 %
[iteration - 72726] training loss: 0.000

[iteration - 72826] training loss: 0.000

[iteration - 72926] training loss: 0.000

[iteration - 73026] training loss: 0.000

[iteration - 73126] training loss: 0.000

[iteration - 73226] training loss: 0.000

[iteration - 73326] training loss: 0.000

[iteration - 73426] training loss: 0.000

epoch: 93  seconds:179
Testing Accuracy : 84.840 %
[iteration - 73508] training loss: 0.000

[iteration - 73608] training loss: 0.000

[iteration - 73708] training loss: 0.000

[iteration - 73808] training loss: 0.000

[iteration - 73908] training loss: 0.000

[iteration - 74008] training loss: 0.000

[iteration - 74108] training loss: 0.000

[iteration - 74208] training loss: 0.000

epoch: 94  seconds:179
Testing Accuracy : 85.220 %
[iteration - 74290] training loss: 0.000

[iteration - 74390] training loss: 0.000

[iteration - 74490] training loss: 0.000

[iteration - 74590] training loss: 0.000

[iteration - 74690] training loss: 0.000

[iteration - 74790] training loss: 0.000

[iteration - 74890] training loss: 0.000

[iteration - 74990] training loss: 0.000

epoch: 95  seconds:181
Testing Accuracy : 84.950 %
[iteration - 75072] training loss: 0.000

[iteration - 75172] training loss: 0.000

[iteration - 75272] training loss: 0.000

[iteration - 75372] training loss: 0.000

[iteration - 75472] training loss: 0.000

[iteration - 75572] training loss: 0.000

[iteration - 75672] training loss: 0.000

[iteration - 75772] training loss: 0.000

epoch: 96  seconds:179
Testing Accuracy : 84.970 %
[iteration - 75854] training loss: 0.000

[iteration - 75954] training loss: 0.000

[iteration - 76054] training loss: 0.000

[iteration - 76154] training loss: 0.000

[iteration - 76254] training loss: 0.000

[iteration - 76354] training loss: 0.000

[iteration - 76454] training loss: 0.000

[iteration - 76554] training loss: 0.000

epoch: 97  seconds:180
Testing Accuracy : 84.990 %
[iteration - 76636] training loss: 0.000

[iteration - 76736] training loss: 0.000

[iteration - 76836] training loss: 0.000

[iteration - 76936] training loss: 0.000

[iteration - 77036] training loss: 0.000

[iteration - 77136] training loss: 0.000

[iteration - 77236] training loss: 0.000

[iteration - 77336] training loss: 0.000

epoch: 98  seconds:180
Testing Accuracy : 85.250 %
[iteration - 77418] training loss: 0.000

[iteration - 77518] training loss: 0.000

[iteration - 77618] training loss: 0.000

[iteration - 77718] training loss: 0.000

[iteration - 77818] training loss: 0.000

[iteration - 77918] training loss: 0.000

[iteration - 78018] training loss: 0.000

[iteration - 78118] training loss: 0.000

epoch: 99  seconds:179
Testing Accuracy : 85.130 %
```