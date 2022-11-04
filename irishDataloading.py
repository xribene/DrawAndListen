#%%
import torch
import numpy as np
import pickle
from pathlib import Path
from utilsDataset import (butter, getDirs, getRandomDistortion4, 
                        distortCurve2, filtfilt, band_limited_noise, custom1)
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import interpolate
import copy
import random
#%%
class IrishDataset(Dataset):
    def __init__(self, datasetList, includeCurves=False, densityType="lp", 
                    condType = 'mean', dirType = 'bugFix', curveOrigin = 'linear',
                    dirs = 5, conds = 3, transpose = 1, nSamples=512,
                    onsetDistProb = 0.0, pitchDistProb = 0.0, curveDistProb = 0.0,
                    barsNum = 3
                    ):
        self.dataset = datasetList
        self.includeCurves = includeCurves
        self.onsetDistProb = onsetDistProb
        self.pitchDistProb = pitchDistProb
        self.curveDistProb = curveDistProb
        self.dirs = dirs
        self.condType = condType
        self.dirType = dirType
        self.nSamples = nSamples
        self.distortionProb = 0.0
        self.curveOrigin = curveOrigin
        self.densityType = densityType
        self.measureQuarterLength = 4
        self.barsNum = barsNum
        Fs = 100
        cutoff = 1
        order = 6
        nyq = 0.5 * Fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        self.samplesPerQuarter = 12

        
        self.transpose = transpose
        if conds == 0:
            # in this case, conds is ignored in the NN
            # I just set conds = 3, because of some errors
            self.conds = 3
        else :
            self.conds = conds
        self.condCenters = np.linspace(-12,12,self.conds)
        
    def __len__(self):
        return len(self.dataset)
    
    def onehot(self, d, vocabSize = 130):
        d = d.astype(np.int)
        midi_vec = np.zeros((len(d), vocabSize))
        k = np.arange(len(d))
        midi_vec[k,d] = 1
        return midi_vec

    def getRawItem(self, idx):
        return copy.deepcopy(self.dataset[idx])

    def __getitem__(self, idx):
        if self.barsNum == 1:
            return self.getSingleBar(idx)
        else:
            return self.getMultiBar(idx)
    
    def getMultiBar(self, idx):
        # check if idx is valid. 
        # a) idx - (self.barsNum//2) > -1 AND idx  < len - (self.barsNum//2) - 1 
        # if not, then clip from (self.barsNum//2) to len - (self.barsNum//2) - 1
        # b) if dataset[idx][measureInd] > (self.barsNum//2) - 1 and idx < dataset[idx][maxInd] - () 
        # if not the first part then add self.barsNum//2 
        # if not the second part then remove self.barsNum//2
        # at the end, do a check. all titles should be the same
        # print("A::A:A:A:")
        # a)
        half = self.barsNum//2
        idx = np.clip(idx, half + 1  , len(self.dataset) - half - 1 - 1)
        # b)
        if self.dataset[idx]['measureInd'] < half  :
            idx += half
        if self.dataset[idx]['measureInd'] > self.dataset[idx]['maxInd'] - half -1:
            idx -= half
        transp = np.random.randint(-5,7)
        try:
            # titles = [self.dataset[idx+x]['title'] for x in range(-half,half+1)]
            # if titles.count(titles[0]) != len(titles):
            #     print(f"idx is {idx} measureInd {self.dataset[idx]['measureInd']} maxInd {self.dataset[idx]['maxInd']}")
                # assert(False)
            multiDict = {}
            for i in range(-half, half+1) :
                multiDict[i] = self.getSingleBar(idx+i, groupTranspose = transp)
        except:
            idx = np.random.randint(10,len(self.dataset)-10)
            for i in range(-half, half+1) :
                multiDict[i] = self.getSingleBar(idx+i, groupTranspose = transp)
        return multiDict
            

    def getSingleBar(self, idx, groupTranspose = None):
        # try:
        # print(idx)
        # print(f"geddint indx {idx}")
        tempDataPoint = dict(copy.deepcopy(self.dataset[idx]))
        outputDict = {}
        outputDict['metadata'] = '-'.join([tempDataPoint['title'],str(tempDataPoint['measureInd']),str(tempDataPoint['maxInd'])])
        # print(tempDataPoint.keys())
        if self.transpose == 1:
            # print("transposing")
            # randomly transpose the whole song.
            # onsets = np.where(tempDataPoint['midisHolds'] < 128)[0]
            valids = tempDataPoint['midisHolds'] < 128
            # maxMidi = np.max(tempDataPoint['midisHolds'][onsets])
            if groupTranspose is None:
                transp = np.random.randint(-5,7)
            else:
                transp = groupTranspose
            tempDataPoint['midisHolds'][valids] += transp
            tempDataPoint['midisHoldsNoRestIrreg'][valids] += transp
            tempDataPoint['midisNoRestReg'] += transp
            tempDataPoint['midisNoRestIrreg'] += transp
            # if self.model in ['ecvaedir', 'curve2vaeNoise']:
                # tempDataPoint['midisNoRestReg'] += transp
            # elif self.model == 'curve2vae':
            #     tempDataPoint['midisNoRest'] += transp

            if np.sum(tempDataPoint['midisHolds'][valids]>127)>0 or np.sum(tempDataPoint['midisHolds'][valids]<0)>0:
                tempDataPoint['midisHolds'][valids] -= 2*transp
                tempDataPoint['midisHoldsNoRestIrreg'][valids] -= 2*transp

                tempDataPoint['midisNoRestReg'] -= 2*transp
                tempDataPoint['midisNoRestIrreg'] -= 2*transp
                # if self.model in ['ecvaedir', 'curve2vaeNoise']:
                #     tempDataPoint['midisNoRestReg'] -= 2*transp
                # else:
                #     tempDataPoint['midisNoRest'] -= 2*transp


        outputDict['midisHoldsOH'] = self.onehot(tempDataPoint['midisHolds'], vocabSize = 130)
        outputDict['midisHoldsNoRestIrregOH'] = self.onehot(tempDataPoint['midisHoldsNoRestIrreg'], vocabSize = 130)
        outputDict['midisNoRestRegOH'] = self.onehot(tempDataPoint['midisNoRestReg'], vocabSize = 130)
        outputDict['midisNoRestIrregOH'] = self.onehot(tempDataPoint['midisNoRestIrreg'], vocabSize = 130)
        outputDict['midisNoRestReg'] = tempDataPoint['midisNoRestReg']

        # FTIA#E TA CPC EDW. 0 for hold 1-12 for notes
        cpc = np.mod(tempDataPoint['midisHoldsNoRestIrreg'],12) + 1
        cpc[tempDataPoint['midisHoldsNoRestIrreg']==128] = 0
        outputDict['cpcsHoldsNoRestIrreg'] = cpc
        outputDict['cpcsHoldsNoRestIrregOH'] = self.onehot(outputDict['cpcsHoldsNoRestIrreg'], vocabSize = 13)
        # calculate the condition
            # the note in the middle of the staff is B = 71
            # assume the max range is 24 (two octaves)
            # i.e for conds = 3 --> 24/(3-1) = 12 --> [65,71,77] = 0 , [53,59,65] = -1 ....
            # i.e for conds = 5 --> 24/(5-1) = 6 --> [68,71,74] = 0 , [62,65,68] = -1, [56,59,62] = -2 ...

        # if self.model == 'music':
        if self.condType == 'mean':
            midiMean = np.mean(tempDataPoint['midisNoRestReg'] ) - 71
            cond = np.argmin(np.abs(np.array(self.condCenters)-midiMean))
        elif self.condType == 'first':
            first = tempDataPoint['midisNoRestReg'][0] - 71
            cond = np.argmin(np.abs(np.array(self.condCenters)-first))
        outputDict['cond'] = np.array([cond])
        outputDict['condOH'] = self.onehot(outputDict['cond'], vocabSize = self.conds)
    # curve2vaeDict['midisNoRest'] = measure['regGrid34']['midisNoRest']
    # curve2vaeDict['curveData'] = measure['curveData']
        # get the quantized dirs vector
        # dirsQuantized = np.clip(np.sign(dirs)*np.ceil(np.abs(dirs/(12/amp))), -amp, amp) + amp
        # dirsQuantized = dirsQuantized.astype(np.int)
        amp = self.dirs // 2
        outputDict['dirs'] = tempDataPoint['dirs']
        if self.dirType == 'complex':
            quantDirs = np.clip(np.sign(tempDataPoint['dirs'])*np.ceil(np.abs(tempDataPoint['dirs']/(12/amp))), -amp, amp) + amp
        elif self.dirType == 'simple':
            quantDirs = np.zeros(tempDataPoint['dirs'].shape)
            onsetInds = np.where(np.logical_and(tempDataPoint['midisNoRestIrreg']!=128, tempDataPoint['midisNoRestIrreg']!=129) == True)[0]
            midiPoints = tempDataPoint['midisNoRestIrreg'][onsetInds]
            dirPoints = np.insert(np.diff(midiPoints),0,0)
            for i, onsetInd in enumerate(onsetInds):
                quantDirs[onsetInd] = dirPoints[i]
            quantDirs = np.clip(np.sign(quantDirs)*np.ceil(np.abs(quantDirs/(12/amp))), -amp, amp) + amp
        elif self.dirType == 'bugFix':
            dirsBugFix = getDirs(tempDataPoint['midisHolds'], bugFix=1)
            outputDict['dirsBugFix'] = dirsBugFix
            # dirsBugFix = np.round(dirsBugFix)
            quantDirs = np.clip(np.sign(np.round(dirsBugFix))*np.ceil(np.abs(np.round(dirsBugFix)/(12/amp))), -amp, amp) + amp
        
        elif self.dirType == 'curveQuant':
            curveNorm =tempDataPoint['midisNoRestIrreg'] - tempDataPoint['midisNoRestIrreg'][0]
            outputDict['curveNorm'] = curveNorm
            quantDirs = np.clip(np.sign(np.round(curveNorm))*np.ceil(np.abs(np.round(curveNorm)/(12/amp))), -amp, amp) + amp

        else:
            assert(False)
        outputDict[f'quantDirs'] = quantDirs
        outputDict[f'dirs{self.dirs}OH'] = self.onehot(quantDirs, vocabSize = self.dirs)


        outputDict['rhythmOH'] = self.onehot(tempDataPoint['rhythm'], vocabSize = 3)

        # print(f"created {outputDict.keys()}")   
            # print("olakala")

        # elif self.model == 'curve':
            
        #     outputDict['midisNoRestReg'] = tempDataPoint['midisNoRest']
        #     midiMean = np.mean(tempDataPoint['midisNoRest'] ) - 71
        #     cond = np.argmin(np.abs(np.array(self.condCenters)-midiMean))
        #     outputDict['cond'] = np.array([cond])
        #     outputDict['condOH'] = self.onehot(outputDict['cond'], vocabSize = self.conds)

        #     outputDict['curveData'] = tempDataPoint['curveData']
        #     outputDict['curveData']['curve'] = (outputDict['curveData']['curve']-71)/12

        #     amp = self.dirs // 2
        #     quantDirs = np.clip(np.sign(tempDataPoint['dirs'])*np.ceil(np.abs(tempDataPoint['dirs']/(12/amp))), -amp, amp) + amp
        #     outputDict[f'dirs{self.dirs}OH'] = self.onehot(quantDirs, vocabSize = self.dirs)
        #     outputDict['rhythmOH'] = self.onehot(tempDataPoint['rhythm'], vocabSize = 3)

            # TODO do normalization
                # # a mask to determine if there are not quantized notes in the dataset
                # mask = []
                # for i in range(12):
                #     if i in [0,3,4,6,8,9]:
                #         mask.append(0)
                #     else : 
                #         mask.append(1)
                # mask = np.array(mask*4) # assumes 4/4 only
                
                # mistakes = np.sum(mask*tempDataPoint['onsets'])
                # if mistakes > 0 : 
                #     print(f"{ind} has {mistakes} mistakes")
        if self.includeCurves is True:

            outputDict['midisNoRestReg'] = tempDataPoint['midisNoRestReg']
            # midiMean = np.mean(tempDataPoint['midisNoRestReg'] ) - 71


            # cond = np.argmin(np.abs(np.array(self.condCenters)-midiMean)) # 71 is B4
            # outputDict['cond'] = np.array([cond])
            # outputDict['condOH'] = self.onehot(outputDict['cond'], vocabSize = self.conds)

            # amp = self.dirs // 2
            # quantDirs = np.clip(np.sign(tempDataPoint['dirs'])*np.ceil(np.abs(tempDataPoint['dirs']/(12/amp))), -amp, amp) + amp
            # outputDict[f'dirs{self.dirs}OH'] = self.onehot(quantDirs, vocabSize = self.dirs)
            # outputDict['rhythmOH'] = self.onehot(tempDataPoint['rhythm'], vocabSize = 3)

            # now create the curves 
            reg34Length = len(tempDataPoint['midisNoRestReg'])
            xAxis = np.linspace(0,reg34Length-1,self.nSamples)
            onsetsGrid = copy.deepcopy(tempDataPoint['rhythmReg'])
            onsetsGrid[onsetsGrid==2] = 1
            xNotePoints = np.where(onsetsGrid==1)[0]
            yNotePoints = tempDataPoint['midisNoRestReg'][xNotePoints]

            xNotePoints = np.append(xNotePoints, reg34Length-1)
            yNotePoints = np.append(yNotePoints, yNotePoints[-1])

            if self.onsetDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.onsetDistProb:
                    numDistOnsets = len(xNotePoints) - 2
                    if numDistOnsets > 0 :
                        shifts = np.random.randint(-3,4,numDistOnsets)
                        tmp = xNotePoints[1:-1] + shifts
                        tmpSort = np.sort(tmp)
                        last = -1
                        for aa in range(numDistOnsets):
                            if tmpSort[aa] == 0 or tmpSort[aa] == last:
                                tmpSort[aa] += 1
                            last = tmpSort[aa]
                        xNotePoints = tmpSort

            if self.pitchDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.pitchDistProb:
                    # numDistPitch = len(yNotePoints)
                    # if numDistOnsets > 0 :
                    shiftsOffset = np.random.choice([-1,1])
                    yNotePoints += shiftsOffset
                            


            # #FEATURE add noise to xNotePoints
            # xNotePoints = xNotePoints + np.sort(np.random.randint())
            
            # funcSquare = interpolate.interp1d(x=xNotePoints, y=yNotePoints, 
            #                     kind='zero',
            #                     fill_value='extrapolate',
            #                     assume_sorted=True)
            if self.curveOrigin == 'linear': 
                funcLinear = interpolate.interp1d(x=xNotePoints, y=yNotePoints, 
                                                kind='linear',
                                                fill_value='extrapolate',
                                                assume_sorted=True)
            # funcAkima = interpolate.Akima1DInterpolator(xNotePoints, y=yNotePoints)
                newYlinear = funcLinear(xAxis)
            elif self.curveOrigin == 'square':
                # newYlinear = tempDataPoint['midisNoRestReg']
                newYlinear = F.interpolate(torch.tensor(tempDataPoint['midisNoRestReg']).view(1,1,-1), self.nSamples, mode='linear').numpy().copy().squeeze()

            density = np.log2(1/tempDataPoint['dursMeasureReg']) # values from [0,4] [olokliro, 16kto]
            density = F.interpolate(torch.tensor(density).view(1,1,-1), self.nSamples, mode='nearest').numpy().copy().squeeze()

            #TODO apply any distortion here
            #TODO FEATURE
            # let's create a distortion, and see how to use it
            correction = np.linspace(-1,1,self.nSamples)
            fidelity = 1
            if self.curveDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.curveDistProb:
                # print("EEE")
                    distortion, correction, fidelity = getRandomDistortion4(self.nSamples, 150)
                    newYlinear = distortCurve2(newYlinear, distortion, self.nSamples)
                    # density = distortCurve2(density, distortion, self.nSamples)

            outputDict['correction'] = correction
            outputDict['fidelity'] = fidelity


            # curveLp = lowpassButter(newYlinear, 1, 100)
            curveLp = filtfilt(self.b, self.a, newYlinear)
            curveLp -= np.min(curveLp)

            # outputDict['curve']  = (curveLp - 71)/12
            outputDict['curve'] = curveLp.reshape(-1,1) #- np.mean(curveLp)
            # if  range biger than -12,12, then devide
            curveRange = np.max(curveLp) - np.min(curveLp)
            if curveRange > 24:
                outputDict['curve'] = outputDict['curve'] / np.max(np.abs(outputDict['curve']))
            else : 
                outputDict['curve'] = outputDict['curve'] / 24 #TODO HUGE MISTAKE # corrected
            # outputDict['noise'] = noiseMasked * noise
            newMin = np.min(outputDict['curve'])
            newMax = np.max(outputDict['curve'])
            if newMin < 0 or newMax>1:
                print(f"{newMin} {newMax}")
                assert(False)
            curveNormRange = newMax - newMin
            middlePoint = newMin + curveNormRange / 2
            diff = middlePoint - 0.5
            outputDict['curve'] = outputDict['curve'].reshape(self.nSamples) - diff
            outputDict['curveDiff'] = np.diff(outputDict['curve'])*100
            outputDict['curveDiff'] = np.insert(outputDict['curveDiff'],0, outputDict['curveDiff'][0])
            # outputDict['curveNoise'] = (noiseMasked * noise + curveLp - 71)/12
            quantCenters = [1.25, 3.29] # this is for 4/4 . For other timesignatures it's different. 

                
            if self.densityType == 'lpQuant':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2

                # low pass filter
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 1:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)

                # find onset offset
                outputDict['onsetOffset'] = np.min(quantizedDensity)
                

                outputDict['onsetDensity'] = lpQuantizedDensity.copy()

            elif self.densityType == 'lpQuantNorm':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2

                # low pass filter
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 1:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)

                # find onset offset
                outputDict['onsetOffset'] = np.min(quantizedDensity)
                
                # because it is lpQuantNorm remove the offset
                outputDict['onsetDensity'] = lpQuantizedDensity.copy() - np.min(quantizedDensity)



            elif self.densityType == 'quant':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                onsetOffsetOH = np.array([[0,0,0]])
                onsetOffsetOH[0,int(np.min(quantizedDensity))] = 1

                # normalize to 1
                quantizedDensity = quantizedDensity / 2
                outputDict['onsetOffsetOH'] = onsetOffsetOH
                outputDict['onsetDensity'] = quantizedDensity.copy()
                outputDict['onsetOffset'] = np.min(quantizedDensity)

            elif self.densityType == 'quantNorm':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2
                outputDict['onsetOffset'] = np.min(quantizedDensity)

                outputDict['onsetDensity'] = quantizedDensity.copy() - outputDict['onsetOffset']
                
            elif self.densityType == 'lp':
                # since this is not quantized, before lp the values are from 0 to 4
                # low pass filter
                quantizedDensity = np.copy(density)
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 4:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)
                else:
                    lpQuantizedDensity /= 4

                # lpQuantizedDensity /= 4 # TODO that was a big mistake

                outputDict['onsetDensity'] = lpQuantizedDensity.copy()
                outputDict['onsetOffset'] = np.min(lpQuantizedDensity)
            elif self.densityType == 'raw':
                # again values are from 0 to 4 so just divide by four
                outputDict['onsetDensity'] = density.copy() / 4
                outputDict['onsetOffset'] = np.min(outputDict['onsetDensity'] )
            else:
                assert(False)
            # quantize 3 levels
            # quantizedDensity = np.copy(density)
            # quantizedDensity[density<=quantCenters[0]] = 0
            # quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
            # quantizedDensity[density>quantCenters[1]] = 2
            # # lpQuantizedDensity = lowpassButter(quantizedDensity, 1, 100)
            # lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)
            # outputDict['onsetDensity'] = lpQuantizedDensity.copy()
            
            # print(outputDict['curve'].shape)
            # print(outputDict['onsetDensity'].shape)

            # distort the curves
            # TODO

        return dict(outputDict)

class RandomDataset(Dataset):
    def __init__(self, p1=0.0, p2=0.0,
                    includeCurves=False, densityType="lp", 
                    condType = 'mean', dirType = 'bugFix', curveOrigin = 'linear',
                    dirs = 7, conds = 3, transpose = 1, nSamples=512,
                    onsetDistProb = 0.0, pitchDistProb = 0.0, curveDistProb = 0.0):
        self.includeCurves = includeCurves
        self.onsetDistProb = onsetDistProb
        self.pitchDistProb = pitchDistProb
        self.curveDistProb = curveDistProb
        self.dirs = dirs
        self.condType = condType
        self.dirType = dirType
        self.conds = conds
        # print(f"eftase {self.dirs}")
        self.nSamples = nSamples
        self.distortionProb = 0.0
        self.curveOrigin = curveOrigin
        self.densityType = densityType
        self.measureQuarterLength = 4
        self.condCenters = np.linspace(-12,12,self.conds)

        Fs = 100
        cutoff = 1
        order = 6
        nyq = 0.5 * Fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        self.samplesPerQuarter = 12

        self.p1 = p1
        self.p2 = p2
        self.quarterDict = [
            [
                # 2 8ths
                [1,0,0,0,0,0,1,0,0,0,0,0],
                # 8 - 16 - 16
                [1,0,0,0,0,0,1,0,0,1,0,0],
                # 16- 16 - 8
                [1,0,0,1,0,0,1,0,0,0,0,0],
                # 8. 16
                [1,0,0,0,0,0,0,0,0,1,0,0],
                # 16 - 8.
                [1,0,0,1,0,0,0,0,0,0,0,0],
                # 16 16 16 16
                [1,0,0,1,0,0,1,0,0,1,0,0],
                # 3plets
                [1,0,0,0,1,0,0,0,1,0,0,0],
                [1,0,0,0,0,0,0,0,1,0,0,0],
                [1,0,0,0,1,0,0,0,0,0,0,0],

                # [1,0,1,0,1,0,0,0,1,0,0,0],
                # [1,0,0,0,1,0,1,0,1,0,0,0],
                # [1,0,0,0,1,0,0,0,1,0,1,0],
                # [1,0,0,0,1,0,1,0,1,0,1,0],
                # [1,0,1,0,1,0,0,0,1,0,1,0],
                # [1,0,1,0,1,0,1,0,1,0,0,0],
                # [1,0,1,0,1,0,1,0,1,0,1,0],
                [1,0,0,0,0,0,0,0,0,0,0,0]
            ],
            [1,1,1,1,1,1,1,1]
        ]
        self.wholeDict = [
            [
                [1000],
                [1010],
                [1100],
                [1001],
                
                [1110],
                [1011],
                [1111],
                [1101]
            ],
            [
                [4],
                [2,2],
                [1,3],
                [3,1],
                [1,1,2],
                [2,1,1],
                [1,1,1,1],
                [1,2,1]
            ],
            [1,3,3,3,3,3,5,3]
        ]
        # samplesNum = 10**1
        self.measureQuarterLength = 4
        # self.wholeConfigConst = random.choices(population=self.wholeDict[1], weights=self.wholeDict[2],k=1)[0]
        # self.localOnsetsConst = random.choice(self.quarterDict[0])
    def __len__(self):
        return 568001
    def onehot(self, d, vocabSize = 130):
        d = d.astype(np.int)
        midi_vec = np.zeros((len(d), vocabSize))
        k = np.arange(len(d))
        midi_vec[k,d] = 1
        return midi_vec
    def __getitem__(self, idx):
        gridIrreg34 = np.array([a + b for a in range(4) for b in [0, 1/4, 1/3, 2/4, 2/3, 3/4]])
        gridIrreg34 = np.append(gridIrreg34, 4.0)
        gridReg34 = np.linspace(0.0, 4, int(4*12)+1)
        
        outputDict = {}
        p = random.random()
        # print(wholeConfig)
        if p < self.p1:
            numberOfOnsets = random.randrange(2,48/2)
            onsetsInds = random.sample(range(1,48), numberOfOnsets-1)
            onsetsInds.append(0)
            onsetsInds.sort()
            onsets = np.array([1 if i in onsetsInds else 0 for i in range(48)])
            
            # print(f"new {onsets}")
        else:
            wholeConfig = random.choices(population=self.wholeDict[1], weights=self.wholeDict[2],k=1)[0]
            # with a probability, we can generate a more random wholeConfig
            
            pos = 0
            onsets = np.zeros(self.samplesPerQuarter*self.measureQuarterLength)
            # durs = str(wholeConfig)
            # print(wholeConfig)
            for dur in wholeConfig:
                # print(dur)
                if dur == 1:
                    localOnsets = random.choice(self.quarterDict[0])
                    onsets[pos:(pos+self.samplesPerQuarter)] = localOnsets
                else:
                    onsets[pos] = 1
                
                pos += dur*self.samplesPerQuarter
            # print(f"old {onsets}")

        # get the durs vector as we do during IrishParser
        onsetIndsReg = np.where(onsets==1)[0]
        regOffsets = gridReg34[onsetIndsReg]
        onsetIndsIrreg = []
        for regOffset in regOffsets:
            aa = np.argmin(np.abs(gridIrreg34 - regOffset))
            onsetIndsIrreg.append(aa)








        onsetIndsReg2 = np.append(onsetIndsReg, len(onsets))
        dursZH = np.zeros(48)
        # for k, ind in enumerate(onsetInds[:-1]):
        #     dursZH[ind] = onsetInds[k+1] - ind
        for k, ind in enumerate(onsetIndsReg2[:-1]):
            dursZH[ind] = onsetIndsReg2[k+1] - ind
        dursZH = custom1(dursZH, 0)
        dursZHMeasureRel = dursZH / (self.measureQuarterLength*self.samplesPerQuarter)

        p = random.random()
        if p>self.p2:
        # now generate also the notes
        # we start from 0. Keep track of range. Can't be more than 24
        # also add random noise +/- 1
            maxFreq = random.randint(2,15)
            noise = band_limited_noise(2,maxFreq, self.measureQuarterLength*self.samplesPerQuarter, 100)
            noise -= np.min(noise)
            if maxFreq > 2:
                noise = noise / np.abs(np.max(noise))
            #TODO maybe this should not be uniform.
            noteRange = random.choice(list(range(24)))
            noise *= noteRange
            # yNotePoints = noise[onsetInds[:-1]]
            yNotePoints = noise[onsetIndsReg]

        else:
            # notesNum = len(onsetInds) - 1
            notesNum = len(onsetIndsReg)

            yNotePoints = list(map(random.randrange, [24]*notesNum))

        # NOW I CAN CREATE THE IRREG MIDIS HOLDS VECTOR FOR VAE RANDOM TRAINING
        outputDict['midisHolds'] = np.zeros(24) + 128
        randomMean = random.choice(list(range(60,84)))
        outputDict['midisHolds'][onsetIndsIrreg] = np.round(yNotePoints) + randomMean - noteRange//2
        outputDict['midisHoldsOH'] = self.onehot(outputDict['midisHolds'], vocabSize = 130)

        outputDict['rhythm'] = np.zeros(24) 
        outputDict['rhythm'][onsetIndsIrreg] = 1

        midiMean = randomMean - 71
        cond = np.argmin(np.abs(np.array(self.condCenters)-midiMean))
        outputDict['cond'] = np.array([cond])
        outputDict['condOH'] = self.onehot(outputDict['cond'], vocabSize = self.conds)

        outputDict['dirs'] = getDirs(outputDict['midisHolds'], mode='linear')

        amp = self.dirs // 2
        dirsBugFix = getDirs(outputDict['midisHolds'], bugFix=1)
        outputDict['dirsBugFix'] = dirsBugFix

        quantDirs = np.clip(np.sign(np.round(dirsBugFix))*np.ceil(np.abs(np.round(dirsBugFix)/(12/amp))), -amp, amp) + amp

        outputDict[f'quantDirs'] = quantDirs
        outputDict[f'dirs{self.dirs}OH'] = self.onehot(quantDirs, vocabSize = self.dirs)


        outputDict['rhythmOH'] = self.onehot(outputDict['rhythm'], vocabSize = 3)

        # THE REST FROM HERE IS FOR CURVES
        if self.includeCurves is True:
            # xNotePoints = np.append(onsetInds[:-1], 48-1)
            yNotePoints = outputDict['midisHolds'][onsetIndsIrreg]
            xNotePoints = np.append(onsetIndsReg, 48-1)
            yNotePoints = np.append(yNotePoints, yNotePoints[-1])


            #TODO add noise here
            if self.onsetDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.onsetDistProb:
                    numDistOnsets = len(xNotePoints) - 2
                    if numDistOnsets > 0 :
                        shifts = np.random.randint(-3,4,numDistOnsets)
                        tmp = xNotePoints[1:-1] + shifts
                        tmpSort = np.sort(tmp)
                        last = -1
                        for aa in range(numDistOnsets):
                            if tmpSort[aa] == 0 or tmpSort[aa] == last:
                                tmpSort[aa] += 1
                            last = tmpSort[aa]
                        xNotePoints = tmpSort

            if self.pitchDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.pitchDistProb:
                    # numDistPitch = len(yNotePoints)
                    # if numDistOnsets > 0 :
                    shiftsOffset = np.random.choice([-1,1])
                    yNotePoints += shiftsOffset
                            
            # try:
            # if len(xNotePoints)>4:
            #     xNotePoints = xNotePoints.astype(np.float)
            #     xNotePoints[2:-2] = xNotePoints[2:-2]+np.random.random(len(xNotePoints)-4) 
            #     xNotePoints = np.sort(xNotePoints)
            funcLinear = interpolate.interp1d(x=xNotePoints, y=yNotePoints, 
                                                    kind='linear',
                                                    fill_value='extrapolate',
                                                    assume_sorted=True)
            # except:
            #     print(xNotePoints) 
            #     print(np.random.random(len(xNotePoints)-4) )


            # # funcAkima = interpolate.Akima1DInterpolator(xNotePoints, y=yNotePoints)
            xAxis = np.linspace(0,48-1,self.nSamples)

            newYlinear = funcLinear(xAxis)


            density = np.log2(1/dursZHMeasureRel) # values from [0,4] [olokliro, 16kto]
            density = F.interpolate(torch.tensor(density).view(1,1,-1), size=self.nSamples, mode='nearest').numpy().copy().squeeze()

            correction = np.linspace(-1,1,self.nSamples)
            fidelity = 1
            if self.curveDistProb > 0.0:
                zari = np.random.uniform()
                if zari < self.curveDistProb:
                # print("EEE")
                    distortion, correction, fidelity = getRandomDistortion4(self.nSamples, 150)
                    newYlinear = distortCurve2(newYlinear, distortion, self.nSamples)
                    # density = distortCurve2(density, distortion, self.nSamples)

            outputDict['correction'] = correction
            outputDict['fidelity'] = fidelity

            # curveLp = lowpassButter(newYlinear, 1, 100)
            curveLp = filtfilt(self.b, self.a, newYlinear)
            # if np.min(curveLp) < 0:
            curveLp -= np.min(curveLp)
            # outputDict['curve']  = (curveLp - 71)/12
            outputDict['curve'] = curveLp.reshape(-1,1) #- np.mean(curveLp)
            # outputDict['curve'] += 0.5
            # if  range biger than -12,12, then devide
            curveRange = np.max(curveLp) - np.min(curveLp)
            # print(curveRange)

            if curveRange > 24:
                outputDict['curve'] = outputDict['curve'] / np.max(np.abs(outputDict['curve']))
            else : 
                outputDict['curve'] = outputDict['curve'] / 24 #TODO HUGE MISTAKE
            # outputDict['noise'] = noiseMasked * noise

            # sanity checks
            newMin = np.min(outputDict['curve'])
            newMax = np.max(outputDict['curve'])
            if newMin < 0 or newMax>1:
                assert(False)
            curveNormRange = newMax - newMin
            middlePoint = newMin + curveNormRange / 2
            diff = middlePoint - 0.5
            # outputDict['curve'] -= diff
            outputDict['curve'] = outputDict['curve'].reshape(self.nSamples) - diff
            outputDict['curveDiff'] = np.diff(outputDict['curve'],prepend=outputDict['curve'][0])*100
            outputDict['curveDiff'] = np.insert(outputDict['curveDiff'],0, outputDict['curveDiff'][0])

            # outputDict['curveNoise'] = (noiseMasked * noise + curveLp - 71)/12
                

            # outputDict['curveNoise'] = (noiseMasked * noise + curveLp - 71)/12
            quantCenters = [1.25, 3.29] # this is for 4/4 . For other timesignatures it's different. 

            # normDensity = True
            if self.densityType == 'lpQuant':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2

                # low pass filter
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 1:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)

                # find onset offset
                outputDict['onsetOffset'] = np.min(quantizedDensity)
                

                outputDict['onsetDensity'] = lpQuantizedDensity.copy()

            elif self.densityType == 'lpQuantNorm':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2

                # low pass filter
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 1:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)

                # find onset offset
                outputDict['onsetOffset'] = np.min(quantizedDensity)
                
                # because it is lpQuantNorm remove the offset
                outputDict['onsetDensity'] = lpQuantizedDensity.copy() - np.min(quantizedDensity)



            elif self.densityType == 'quant':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2
                outputDict['onsetDensity'] = quantizedDensity.copy()
                outputDict['onsetOffset'] = np.min(quantizedDensity)
            
            elif self.densityType == 'quantNorm':
                # quantize 3 levels
                quantizedDensity = np.copy(density)
                quantizedDensity[density<=quantCenters[0]] = 0
                quantizedDensity[np.logical_and(density>quantCenters[0],density<=quantCenters[1])] = 1
                quantizedDensity[density>quantCenters[1]] = 2

                # normalize to 1
                quantizedDensity = quantizedDensity / 2
                outputDict['onsetOffset'] = np.min(quantizedDensity)

                outputDict['onsetDensity'] = quantizedDensity.copy() - outputDict['onsetOffset']

            elif self.densityType == 'lp':
                # since this is not quantized, before lp the values are from 0 to 4
                # low pass filter
                quantizedDensity = np.copy(density)
                if np.max(quantizedDensity) - np.min(quantizedDensity) == 0:
                    lpQuantizedDensity = np.copy(quantizedDensity)
                else:
                    lpQuantizedDensity = filtfilt(self.b, self.a, quantizedDensity)

                # make sure it is from 0 to 1 after filtering
                if np.min(lpQuantizedDensity) < 0:
                    lpQuantizedDensity -= np.min(lpQuantizedDensity)
                if np.max(lpQuantizedDensity) > 4:
                    lpQuantizedDensity = lpQuantizedDensity / np.max(lpQuantizedDensity)
                else:
                    lpQuantizedDensity /= 4

                # lpQuantizedDensity /= 4 # TODO that was a big mistake

                outputDict['onsetDensity'] = lpQuantizedDensity.copy()
                outputDict['onsetOffset'] = np.min(lpQuantizedDensity)
            elif self.densityType == 'raw':
                # again values are from 0 to 4 so just divide by four
                outputDict['onsetDensity'] = density.copy() / 4
                outputDict['onsetOffset'] = np.min(outputDict['onsetDensity'] )
            else:
                assert(False)

        return dict(outputDict)

#%%
def irishSplitter(datasetPath, size = None, includeCurves = False, dirs = 7, conds = 3, 
                    transpose = 1, densityType = 'lp', nSamples=512, curveOrigin = 'linear',
                    dirType = 'bugFix', condType = 'mean', barsNum = 1, 
                    onsetDistProb = 0.0, pitchDistProb = 0.0, curveDistProb = 0.0):
    cwd = Path.cwd()
    dataset = []

    with open(datasetPath / "irishDataset_parsed.dat", 'rb') as f:
        dataset = pickle.load(f)

    if size :
        dataset = dataset[:size]

    trainSize = int(0.8*len(dataset))
    validSize = (len(dataset) - trainSize)//2
    testSize = len(dataset) - trainSize - validSize
    print(f"{trainSize} {validSize} {testSize}")

    # TODO shuffledInds is not used
    np.random.seed(0)        
    # shuffledInds = np.arange(len(dataset))
    # np.random.shuffle(shuffledInds)
    # np.random.shuffle(dataset)
    trainData = IrishDataset(dataset[0:trainSize], includeCurves, dirs=dirs, conds=conds, 
                                transpose = transpose, densityType=densityType, curveOrigin = curveOrigin,
                                nSamples = nSamples, condType = condType, dirType=dirType, barsNum = barsNum,
                                onsetDistProb = onsetDistProb, pitchDistProb = pitchDistProb, curveDistProb = curveDistProb)
    validData = IrishDataset(dataset[trainSize:(trainSize+validSize)], includeCurves, curveOrigin = curveOrigin,
                                dirs=dirs, conds=conds, transpose = 0, densityType=densityType, 
                                nSamples = nSamples, condType = condType, dirType=dirType, barsNum = barsNum,
                                onsetDistProb = onsetDistProb, pitchDistProb = pitchDistProb, curveDistProb = curveDistProb)
    testData = IrishDataset(dataset[(trainSize+validSize):], includeCurves, dirs=dirs, 
                            conds=conds, transpose = 0, densityType=densityType, curveOrigin = curveOrigin,
                            nSamples = nSamples, condType = condType, dirType=dirType, barsNum = barsNum,
                            onsetDistProb = onsetDistProb, pitchDistProb = pitchDistProb, curveDistProb = curveDistProb)
    return trainData, validData, testData
