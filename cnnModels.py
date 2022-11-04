#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# from non_local_embedded_gaussian import NONLocalBlock2D, NONLocalBlock1D
from torch.distributions import Normal
# %%
class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
class Conv1d(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, stride=1, dilation=1,
                        paddingType = 'same'):
        super(Conv1d, self).__init__()
        self.net = nn.Sequential()
        # assymetrical padding like tensorflow
        if paddingType == 'same':
            inpShape = (None, inpChan, length) #currentInpShape[2]//stride[0])
            outShape = (None, outChan, length//stride)
            C_p = outShape[2]
            C = inpShape[2]
            Pc = (C_p - 1) * stride + ( kernel - 1) * dilation + 1 - C
            right = Pc - Pc//2
            left = Pc//2
            self.net.add_module('zp',nn.ZeroPad2d((left, right,0,0)))
        self.net.add_module('conv1d', nn.Conv1d(inpChan, outChan, kernel, stride ,padding=0, dilation=dilation, bias=True))

        torch.nn.init.xavier_uniform_(
        self.net.conv1d.weight, gain=torch.nn.init.calculate_gain('linear'))
        # self.outputShape = length // stride
    def forward(self, x):
        # print(self.pad(x).shape)
        return self.net(x)

class ConvUp1d(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, scale=1, dilation=1, mode = 'nearest'):
        super(ConvUp1d, self).__init__()
        # assymetrical padding like tensorflow
        self.net = nn.Sequential()
        if scale > 1:
            self.net.add_module('upsample', Upsample(scale_factor = scale,mode = mode))
        self.net.add_module('conv', Conv1d(inpChan, outChan, length = scale*length, kernel=kernel, stride=1, dilation=dilation,
                                paddingType = 'same'))
    def forward(self, x):
        return self.net(x)
class ConvBlock(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, stride=1, dilation=1,
                  norm = False, activation = None, pooling = None, poolingKernel = 2):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential()
        if pooling is not None:
            stride = 1
        self.net.add_module('conv', Conv1d(inpChan, outChan, length, kernel, stride , dilation=dilation, paddingType='same'))  
        self.outLength = length // stride

        if norm is True:
            self.net.add_module('batchNorm', nn.BatchNorm1d(outChan))
        if activation == 'lrelu':
            self.net.add_module('activation', nn.LeakyReLU(inplace=True))
        elif activation == 'relu':
            self.net.add_module('activation', nn.ReLU(inplace=True))
        if pooling == 'max':
            self.net.add_module('pooling', nn.MaxPool1d(poolingKernel))
            self.outLength = self.outLength // poolingKernel
    def forward(self, x):
        return self.net(x)


class MlpBlock(nn.Module):
    def __init__(self, inpChan, outChan, norm = False, activation = None, dropout = 0.0):
        super(MlpBlock, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('conv', nn.Linear(inpChan, outChan))  
        
        if norm is True:
            self.net.add_module('batchNorm', nn.BatchNorm1d(outChan))
        if activation == 'lrelu':
            self.net.add_module('activation', nn.LeakyReLU(inplace=True))
        elif activation == 'relu':
            self.net.add_module('activation', nn.ReLU(inplace=True))
        if dropout > 0.0:
            self.net.add_module('dropout', nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class ConvUpBlock(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, scale=1, dilation=1,
                  norm = False, activation = None, mode = 'nearest'):
        super(ConvUpBlock, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('convUp', ConvUp1d(inpChan, outChan, length, kernel, scale , dilation=dilation, mode=mode))  
        self.outLength = length * scale

        if norm is True:
            self.net.add_module('batchNorm', nn.BatchNorm1d(outChan))
        if activation == 'lrelu':
            self.net.add_module('activation', nn.LeakyReLU(inplace=True))
        elif activation == 'relu':
            self.net.add_module('activation', nn.ReLU(inplace=True))
        # if pooling == 'max':
        #     self.net.add_module('pooling', nn.MaxPool1d(poolingKernel))
        #     self.outLength = self.outLength // poolingKernel
    def forward(self, x):
        return self.net(x)

class ResDownBlock(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, stride=1, dilation=1,
                  norm = False, activation = 'lrelu', pooling = None, poolingKernel = 2):
        super(ResDownBlock, self).__init__()
        
        btlnkChan = outChan//2
        if btlnkChan == 0:
            btlnkChan = 1
        
        self.outLength = length
        self.conv1 = ConvBlock(inpChan, btlnkChan, length, 1, 1, 1, norm, activation, None)
        self.outLength = self.conv1.outLength
        self.conv2 = ConvBlock(btlnkChan, btlnkChan, self.outLength, kernel, stride, dilation, norm, activation, pooling, poolingKernel)
        self.outLength = self.conv2.outLength
        self.conv3 = ConvBlock(btlnkChan, outChan, self.outLength, 1, 1, 1, norm, activation, None)
        self.outLength = self.conv3.outLength

        self.skip = ConvBlock(inpChan, outChan, length, 1, stride, 1, norm, None, pooling, poolingKernel)
        # if pooling == 'average':
        #     self.AvePool = nn.AvgPool2d(scale)
        # else:
        #     self.AvePool = nn.MaxPool2d(scale)
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(x+skip)
        return x

class ResUpBlock(nn.Module):
    def __init__(self, inpChan, outChan, length = 512, kernel=3, scale=1, dilation=1, finalActivation = 'lrelu',
                  norm = False, activation = 'lrelu', mode='nearest', bottleneck = True):
        super(ResUpBlock, self).__init__()
        # scale has the role that stride had in ResDownBlock
        # we used stride to reduce the spatial size (instead of pooling)
        if bottleneck == True:
            btlnkChan = outChan//2
            if btlnkChan == 0:
                btlnkChan = 1
        else:
            btlnkChan = outChan

        self.outLength = length
        self.conv1 = ConvUpBlock(inpChan, btlnkChan, length, 1, 1, 1, norm, activation, mode)
        self.outLength = self.conv1.outLength
        self.conv2 = ConvUpBlock(btlnkChan, btlnkChan, self.outLength, kernel, scale, dilation, norm, activation, mode)
        self.outLength = self.conv2.outLength
        self.conv3 = ConvUpBlock(btlnkChan, outChan, self.outLength, 1, 1, 1, norm, activation, mode)
        self.outLength = self.conv3.outLength

        self.finalActivation = finalActivation
        self.skip = ConvUpBlock(inpChan, outChan, length, 1, scale, 1, norm, mode)
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        # print(x.shape)
        # print(self.skip)
        skip = self.skip(x)
        # print(skip.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.finalActivation is not None:
            x = self.act(x+skip)
        else: 
            x = x+skip
        return x

#%%
class CurveVAEDil(nn.Module):
    def __init__(self, device, inpChan, baseChan, zDims=128, layers = 6, kernel=3, nSamples = 512, decoderDilation=True, encoderDilation=True, finalActivation = 'lrelu',
                    mode='nearest', bottleneck=True, encoderType = 'CNN', decoderType = 'CNN',norm = True, pooling=None, activation = 'lrelu', dropout = 0.0):
        super(CurveVAEDil, self).__init__()
        self.nSamples = nSamples
        self.device = device
        self.zDims = zDims
        self.inpChan = inpChan
        self.baseChan = baseChan
        self.dropout = dropout
        inpFlatSize = inpChan * nSamples
        self.encoderType = encoderType
        self.decoderType = decoderType
        self.encoder = nn.Sequential()

        if encoderType == 'CNN':
            # layers = 7
            currentChan = inpChan
            currentLength = nSamples
            kernel = kernel
            # self.decoderList = nn.ModuleList()
            strides = [2]*layers
            for i in range(layers):
                if encoderDilation is True:
                    currentDil = 2**i
                else:
                    currentDil = 2**0
                # currentDil = 2**i
                outChan = baseChan * 2**i
                resBlock =  ResDownBlock(currentChan, outChan, length = currentLength, kernel=kernel, stride=strides[i], dilation=currentDil,
                    norm = norm, activation = activation, pooling = pooling, poolingKernel = strides[i])
                self.encoder.add_module(f'layer{i}', resBlock)
                print(f"encoder layer {i} inpC {currentChan} outC {outChan} dil {currentDil}")
                # resUpBlock =  ResUpBlock(outChan, currentChan, length = currentLength, kernel=kernel, scale=strides[i], dilation=currentDil,
                #       norm = norm, activation = activation, pooling = pooling, poolingKernel = strides[i])
                # self.decoderList.append()
                currentChan = outChan
                currentLength = resBlock.outLength
            # print(currentLength)
            self.mu = Conv1d(currentChan,zDims,length=currentLength,
                                                    kernel=currentLength, paddingType = None)
            self.std = Conv1d(currentChan,zDims,length=currentLength,
                                                    kernel=currentLength, paddingType = None)
        elif encoderType == 'MLP':
            if layers == 2:
                self.encoder.add_module('layer1_mlp', MlpBlock(inpFlatSize, 512, norm = norm, activation = activation, dropout=dropout))
                self.encoder.add_module('layer3_mlp', MlpBlock(512, 128, norm = norm, activation = activation, dropout=dropout))
            else:
                self.encoder.add_module('layer1_mlp', MlpBlock(inpFlatSize, 512, norm = norm, activation = activation, dropout=dropout))
                self.encoder.add_module('layer2_mlp', MlpBlock(512, 256, norm = norm, activation = activation, dropout=dropout))

                self.encoder.add_module('layer3_mlp', MlpBlock(256, 128, norm = norm, activation = activation, dropout=dropout))
            self.mu = nn.Linear(128, zDims)
            self.std = nn.Linear(128, zDims)

            # currentChan = baseChan * 2**(layers-1)

        self.decoder = nn.Sequential()

        if decoderType == 'CNN':
            
            if encoderType == 'CNN':
                self.upsample = Upsample(scale_factor=currentLength)
                self.conv11 = nn.Conv1d(zDims, currentChan,1,1,0,1)
            
            # currentChan = zDims
            scales = [2]*layers
            for i in range(layers):
                currentAct = activation
                if i == layers-1:
                    currentAct = finalActivation
                if decoderDilation is True:
                    currentDil = 2**i
                else:
                    currentDil = 2**0
                if i < layers - 1:
                    outChan = baseChan * 2**(layers-i-2)
                elif i == layers - 1:
                    outChan = inpChan
                resUpBlock =  ResUpBlock(currentChan, outChan, length = currentLength, kernel=kernel, scale=scales[i], dilation=currentDil,
                    norm = norm, activation = activation, mode=mode, bottleneck=bottleneck, finalActivation = currentAct )
                
                self.decoder.add_module(f'layerUp{i}', resUpBlock)
                print(f"decoder layer {i} inpC {currentChan} outC {outChan} dil {currentDil}")

                # resUpBlock =  ResUpBlock(outChan, currentChan, length = currentLength, kernel=kernel, scale=strides[i], dilation=currentDil,
                #       norm = norm, activation = activation, pooling = pooling, poolingKernel = strides[i])
                # self.decoderList.append()
                currentChan = outChan
                currentLength = resBlock.outLength
        
        elif decoderType == 'MLP':
            if layers == 2:
                self.decoder.add_module('layer1_mlp', MlpBlock(zDims, 512, norm = norm, activation = activation,dropout=dropout))
                self.decoder.add_module('layer3_mlp', MlpBlock(512, inpFlatSize, norm = False, activation = finalActivation))

            else:
                self.decoder.add_module('layer1_mlp', MlpBlock(zDims, 256, norm = norm, activation = activation,dropout=dropout))
                self.decoder.add_module('layer2_mlp', MlpBlock(256, 512, norm = norm, activation = activation,dropout=dropout))

                self.decoder.add_module('layer3_mlp', MlpBlock(512, inpFlatSize, norm = False, activation = finalActivation))

    def runEncoder(self, x):
        batchSize = x.shape[0]
        if self.encoderType == 'MLP':
            x = x.view(batchSize, -1)
        x = self.encoder(x)
        mu = self.mu(x)
        # print(mu.shape)
        std = self.std(x)
        std = torch.nn.functional.softplus(std) + 1e-6

        distr = Normal(mu, std)
        return distr

    def forward(self, x):
        batchSize = x.shape[0]
        if self.encoderType == 'MLP':
            x = x.view(batchSize, -1)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        mu = self.mu(x)
        # print(mu.shape)
        std = self.std(x)
        std = torch.nn.functional.softplus(std) + 1e-6

        distr = Normal(mu, std)
        z = distr.rsample()
        # print(f"z {z.shape}")

        if self.decoderType == 'CNN':
            if self.encoderType == 'CNN':
                z = self.upsample(z)
                z = self.conv11(z)
            # if self.encoderType == 'MLP':
            #     z = z.view(batchSize, self.zDims, 1)
        # print(z.shap  e)
        out = self.decoder(z.squeeze())
        if self.decoderType == 'MLP':
            out = out.view(batchSize, self.inpChan, -1)



        return out, mu, std

#%%
class CurveEncoder(nn.Module):
    def __init__(self, device, inpChan, baseChan, zDims=128, layers = 6, kernel=3, nSamples = 512, decoderDilation=True, encoderDilation=True, finalActivation = 'lrelu',
                    mode='nearest', bottleneck=True, encoderType = 'CNN', decoderType = 'CNN',norm = True, pooling=None, activation = 'lrelu', dropout = 0.0):
        super(CurveEncoder, self).__init__()
        self.nSamples = nSamples
        self.device = device
        self.zDims = zDims
        self.inpChan = inpChan
        self.baseChan = baseChan
        self.dropout = dropout
        inpFlatSize = inpChan * nSamples
        self.encoderType = encoderType
        self.decoderType = decoderType
        self.encoder = nn.Sequential()

        if encoderType == 'CNN':
            # layers = 7
            currentChan = inpChan
            currentLength = nSamples
            kernel = kernel
            # self.decoderList = nn.ModuleList()
            strides = [2]*layers
            for i in range(layers):
                if encoderDilation is True:
                    currentDil = 2**i
                else:
                    currentDil = 2**0
                # currentDil = 2**i
                outChan = baseChan * 2**i
                resBlock =  ResDownBlock(currentChan, outChan, length = currentLength, kernel=kernel, stride=strides[i], dilation=currentDil,
                    norm = norm, activation = activation, pooling = pooling, poolingKernel = strides[i])
                self.encoder.add_module(f'layer{i}', resBlock)
                # print(f"encoder layer {i} inpC {currentChan} outC {outChan} dil {currentDil}")
                # resUpBlock =  ResUpBlock(outChan, currentChan, length = currentLength, kernel=kernel, scale=strides[i], dilation=currentDil,
                #       norm = norm, activation = activation, pooling = pooling, poolingKernel = strides[i])
                # self.decoderList.append()
                currentChan = outChan
                currentLength = resBlock.outLength
            # print(currentLength)
            self.mu = Conv1d(currentChan,zDims,length=currentLength,
                                                    kernel=currentLength, paddingType = None)
            self.std = Conv1d(currentChan,zDims,length=currentLength,
                                                    kernel=currentLength, paddingType = None)
        elif encoderType == 'MLP':
            if layers == 2:
                self.encoder.add_module('layer1_mlp', MlpBlock(inpFlatSize, 512, norm = norm, activation = activation, dropout=dropout))
                self.encoder.add_module('layer3_mlp', MlpBlock(512, 128, norm = norm, activation = activation, dropout=dropout))
            else:
                self.encoder.add_module('layer1_mlp', MlpBlock(inpFlatSize, 512, norm = norm, activation = activation, dropout=dropout))
                self.encoder.add_module('layer2_mlp', MlpBlock(512, 256, norm = norm, activation = activation, dropout=dropout))

                self.encoder.add_module('layer3_mlp', MlpBlock(256, 128, norm = norm, activation = activation, dropout=dropout))
            self.mu = nn.Linear(128, zDims)
            self.std = nn.Linear(128, zDims)

            # currentChan = baseChan * 2**(layers-1)


    def runEncoder(self, x):
        batchSize = x.shape[0]
        if self.encoderType == 'MLP':
            x = x.view(batchSize, -1)
        x = self.encoder(x)
        mu = self.mu(x)
        # print(mu.shape)
        std = self.std(x)
        std = torch.nn.functional.softplus(std) + 1e-6

        distr = Normal(mu, std)
        return distr

    def forward(self, x):
        batchSize = x.shape[0]
        if self.encoderType == 'MLP':
            x = x.view(batchSize, -1)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        mu = self.mu(x)
        # print(mu.shape)
        std = self.std(x)
        std = torch.nn.functional.softplus(std) + 1e-6

        distr = Normal(mu, std)
        # z = distr.rsample()
        # print(f"z {z.shape}")


        return distr