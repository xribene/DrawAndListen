#%%
import collections
import bisect
import numpy as np
from scipy import interpolate
import numpy as np
import torch.nn as nn
import torch
from scipy.signal import butter, lfilter, filtfilt
import torch.nn.functional as F


#%%
infinite_defaultdict = lambda: collections.defaultdict(infinite_defaultdict)
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpassButter(signal, cutoff, Fs, order=6):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    signal = filtfilt(b, a, signal)
    return signal
# def lowpassButterFast(signal, cutoff, Fs, order=6):
#     nyq = 0.5 * Fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     signal = filtfilt(b, a, signal)
#     return signal

# b, a = butter_lowpass(cutoff=0.1, fs=480, order=3)
def getDirs(midisHoldsRests, bugFix = 1, mode = 'linear'):
        midisHoldsRests = np.array(midisHoldsRests)
        scoreLen =  len(midisHoldsRests)
        # input here is the whole score not just a measure
        onsetInds = np.where(np.logical_and(midisHoldsRests!=128, midisHoldsRests!=129) == True)[0]

        # onsetInds = np.where(midisHolds!=128)[0]
        midiPoints = midisHoldsRests[onsetInds]

        onsetInds = np.insert(onsetInds, len(onsetInds) , scoreLen - 1)
        midiPoints = np.insert(midiPoints, len(midiPoints), midiPoints[-1])

        if bugFix == 0:
            dirPoints = np.insert(np.diff(midiPoints),0,0)
            # dirs = np.zeros(24)-10
            # for i, ind in enumerate(onsetInds):
            #     dirs[ind] = dirPoints[i]
            # dirs = custom2Fixed(dirs, -10)
            # either non causal zero hold
            dirs = None
            if mode == 'zero':
                funcZeroHold = interpolate.interp1d(x=np.sort(scoreLen-1-onsetInds), 
                                                    y=np.flip(dirPoints), 
                                                    kind='zero',
                                                    fill_value='extrapolate',
                                                    assume_sorted=True)
                dirs = np.flip(funcZeroHold(np.arange(scoreLen)))
            # or just linear
            elif mode == 'linear':
                # print("AAA")
                funcLinear = interpolate.interp1d(x=onsetInds, 
                                                    y=dirPoints, 
                                                    kind='linear',
                                                    fill_value='extrapolate',
                                                    assume_sorted=True)
                dirs = funcLinear(np.arange(scoreLen))
            return dirs
        elif bugFix == 1:
            midiPointsDuos = [[midiPoints[i],midiPoints[i+1]] for i in range(len(midiPoints)-1)]
            localDiffs = np.diff(midiPointsDuos).reshape(-1)
            onsetIndsDiff = np.diff(onsetInds)
            final = np.zeros(len(midisHoldsRests))
            ind = 0
            for i, diff in enumerate(localDiffs):
                aa = np.linspace(0,diff,onsetIndsDiff[i]+1)
                final[ind:ind+onsetIndsDiff[i]+1] += aa
                # print(final)
                ind += onsetIndsDiff[i]
            return final





def zero_order_hold(x, xp, yp, left=np.nan, assume_sorted=False):
    r"""
    Interpolates a function by holding at the most recent value.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    left: int or float, optional, default is np.nan
        Value to use for any value less that all points in xp
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by DStauffman in July 2020.

    Examples
    --------
    >>> import numpy as np
    >>> xp = np.array([0., 111., 2000., 5000.])
    >>> yp = np.array([0, 1, -2, 3])
    >>> x = np.arange(0, 6001, dtype=float)
    >>> y = zero_order_hold(x, xp, yp)

    """
    # force arrays
    x  = np.asanyarray(x)
    xp = np.asanyarray(xp)
    yp = np.asanyarray(yp)
    # find the minimum value, as anything left of this is considered extrapolated
    xmin = xp[0] if assume_sorted else np.min(xp)
    # check that xp data is sorted, if not, use slower scipy version
    if assume_sorted or np.all(xp[:-1] <= xp[1:]):
        ix = np.searchsorted(xp, x, side='right') - 1
        return np.where(np.asanyarray(x) < xmin, left, yp[ix])
    func = interp1d(xp, yp, kind='zero', fill_value='extrapolate', assume_sorted=False)
    return np.where(np.asanyarray(x) < xmin, left, func(x))

def replaceRests(midiVector):
    midiVector = midiVector.astype('float')
    midiVector[midiVector == 0] = np.nan # or use np.nan
    # A = np.array([nan, nan,  1, nan, nan, 2, 2, nan, 0, nan, nan])
    ok = ~np.isnan(midiVector)
    xp = ok.ravel().nonzero()[0]
    # find first non nan value
    if xp[0] == 0 : 
        pass
    else : 
        for i in range(xp[0]):
            midiVector[i] = midiVector[xp[0]]
    fp = midiVector[~np.isnan(midiVector)]
    x  = np.isnan(midiVector).ravel().nonzero()[0]

    midiVector[np.isnan(midiVector)] = zero_order_hold(x, xp, fp, assume_sorted=True)
    return midiVector
    
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {}
        self.n_tokens = 0 
      
    def index_tokens(self, tokenList):
        for token in tokenList:
            self.index_token(token)

    def index_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

class RhythmTemplate(object):
    def __init__(self,timeSignature):
        if not isinstance(timeSignature,str):
            inp = timeSignature.string
        else:
            inp = timeSignature
        if inp == '2/4':
            self.bar =  [1, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-2,-1,-2, 0,-2,-1,-2]
            self.accent=[0,-3,-2,-3,-1,-3,-2,-3]
        elif inp == '3/4':
            self.bar =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-2,-1,-2, 0,-2,-1,-2, 0,-2,-1,-2]
            self.accent=[0,-3,-2,-3,-1,-3,-2,-3,-1,-3,-2,-3] 
        elif inp == '4/4':
            self.bar =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-2,-1,-2, 0,-2,-1,-2, 0,-2,-1,-2, 0,-2,-1,-2]
            self.accent=[0,-3,-2,-3,-2,-4,-3,-4,-1,-3,-2,-3,-2,-4,-3,-4] 
        elif inp == '3/8':
            self.bar =  [1, 0, 0, 0, 0,-1]
            self.beat = [0,-1, 0,-1, 0,-1]
            self.accent=[0,-3,-2,-3,-2,-3] 
        elif inp == '4/8':
            self.bar =  [1, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-1, 0,-1, 0,-1, 0,-1]
            self.accent=[0,-3,-2,-3,-1,-3,-2,-3]  
        elif inp == '6/8':
            self.bar =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1]
            self.accent=[0,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3]
        elif inp == '9/8':
            self.bar =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1]
            self.accent=[0,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3]
        elif inp == '12/8':
            self.bar =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1]
            self.beat = [0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1]
            self.accent=[0,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3,-1,-3,-2,-3,-2,-3]
        else:
            self.bar = None
            self.beat = None
            self.accent= None
            print(f"no info for timeSignature {inp}")
    def getRhythmTokens(self, dur,mode):
            if mode == 'first':
                return [str(self.bar[i%len(self.bar)])+'_'+ str(self.beat[i%len(self.bar)])+'_'+str(self.accent[i%len(self.bar)]) for i in range(-dur,0)]
            elif mode == 'last' or mode == 'between' :
                return [str(self.bar[i%len(self.bar)])+'_'+ str(self.beat[i%len(self.bar)])+'_'+str(self.accent[i%len(self.bar)]) for i in range(0,dur)]
            else:
                return None

class TimeSignature(object):
    def __init__(self, nom = None, denom = None, beats = None, accents = None):
        self.nom = nom
        self.denom = denom 
        self.beats = beats 
        self.accents = accents 
        self.duration16 = int(self.nom*16/self.denom)
        self.string = str(self.nom) + '/' + str(self.denom)

def getTimeSignaturesNumber(measures):
    timeSignatureChanges = -1
    for i in range(len(measures)):
            try:
                if measures[i].timeSignature is not None:
                    timeSignatureChanges += 1
            except:
                pass
    return timeSignatureChanges + 1

def getTimeSignatureFraction(measures):
    beatCount = measures[0].timeSignature.beatCount
    beatDur = measures[0].timeSignature.beatDuration.quarterLength
    if beatDur-int(beatDur)>0:
        #then the denominator is 8
        denom = 8
        nom = int((beatCount*beatDur/0.5))
    else:
        #the denominator is 4
        denom = 4
        nom = int(beatCount*beatDur/1)
    return nom, denom

def custom1(pinakas, emptyVal = -10):

    telos = 0
    arxi = 0
    isActive = False
    for slot in range(len(pinakas)-1,-1,-1):
        if pinakas[slot] == emptyVal:
            if not isActive:
                telos = slot+1
                arxi = slot
                isActive = 1
            else:
                arxi = slot
        else:
            if not isActive:
                pass
            else:
                # arxi = slot
                pinakas[arxi:telos] = pinakas[slot]
                isActive = 0
    # print(pinakas)
    return pinakas

def custom2(pinakas, emptyVal = -10):

    telos = 0
    arxi = 0
    isActive = False
    for slot in range(len(pinakas)-1,-1,-1):
        if pinakas[slot] == emptyVal:
            if not isActive:
                # it means we are at the end of the vector
                pinakas[slot] = 0
            else:
                arxi = slot

        else:
            if not isActive:
                isActive = 1
                telos = slot
                arxi = slot #- 1
            else:
                pinakas[arxi:telos] = pinakas[telos]
                isActive = 0
    if isActive : 
        # empty values at the begining  
        pinakas[arxi:telos] = pinakas[telos]
    
    return pinakas

# TODO to pes, kanto kiolas.
def custom2Fixed(pinakas, emptyVal = -10):

    telos = 0
    arxi = 0
    isActive = False
    for slot in range(len(pinakas)-1,-1,-1):
        if pinakas[slot] == emptyVal:
            if not isActive:
                # it means we are at the end of the vector
                pinakas[slot] = 0
            else:
                arxi = slot

        else:
            if not isActive:
                isActive = 1
                telos = slot
                arxi = slot #- 1
            else:
                pinakas[arxi:telos] = pinakas[telos]
                isActive = 0
    if isActive : 
        # empty values at the begining  
        pinakas[arxi:telos] = pinakas[telos]
    
    return pinakas

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real
def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)
#%%
frames = 2500000
sqAl = np.zeros(frames) + 1
# sqAl[frames//2:] = -1
originalAlignment = np.linspace(-1,1,frames)
print(np.mean(np.abs(sqAl - originalAlignment)))

#%%

def getRandomDistortion2(n_frames, p_original = 0.5, maxDistPoints=5, mode='linear'):
    originalAlignment = np.linspace(-1,1,n_frames)
    distortedAlignment = np.zeros(n_frames)#np.linspace(-1,1,n_frames)
    
    # while np.max(np.abs(distortedAlignment - originalAlignment)) >= maxDist or np.max(np.abs(distortedAlignment - originalAlignment)) <= minDist :        
    # while np.mean(np.abs(np.zeros(n_frames) - originalAlignment)) < 1 :
    # newPoints = np.random.rand(int(n_frames*distRate))
    # distPoints = 2 gives the original
    isOriginal = np.random.choice([True, False], p=[p_original, 1-p_original])
    zari = np.random.uniform()
    if isOriginal == True:
        distortedAlignment = originalAlignment
        reverseAlignment = originalAlignment
    else : 
        distPoints = np.random.choice(np.arange(3, 3 + maxDistPoints))

        newPoints = np.random.rand(distPoints)
        newPoints = np.sort(
                        (newPoints - np.min(newPoints)) * 2 /
                        (np.max(newPoints) - np.min(newPoints)) - 1)
        f = interpolate.interp1d(np.linspace(-1, 1, len(newPoints)), newPoints, kind=mode)
        # f_i = interpolate.interp1d(newPoints, np.linspace(-1, 1, len(newPoints)), kind=mode)
        distortedAlignment = f(np.linspace(-1, 1, n_frames))
        # reverseAlignment = f_i(np.linspace(-1, 1, n_frames))

    # print(np.mean(np.abs(distortedAlignment - originalAlignment)))
    # plt.plot(np.linspace(-1,1,n_frames), distortedAlignment )
    # plt.plot(np.linspace(-1,1,n_frames), reverseAlignment )
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.draw()
    # fidelity = np.mean(np.abs(distortedAlignment - originalAlignment)) # always from 0 to 1
   
    return distortedAlignment#, reverseAlignment, fidelity#, distortedAlignmentSmooth[0,0,:].detach().numpy()

def getRandomDistortion3(n_frames, maxDistPoints=5, mode='linear'):
    originalAlignment = np.linspace(-1,1,n_frames)

    # distortedAlignment = np.zeros(n_frames)#np.linspace(-1,1,n_frames)
    
    distPoints = np.random.choice(np.arange(3, 3 + maxDistPoints))

    newPoints = np.random.rand(distPoints)
    newPoints = np.sort(
                    (newPoints - np.min(newPoints)) * 2 /
                    (np.max(newPoints) - np.min(newPoints)) - 1)
    f = interpolate.interp1d(np.linspace(-1, 1, len(newPoints)), newPoints, kind=mode)
    f_i = interpolate.interp1d(newPoints, np.linspace(-1, 1, len(newPoints)), kind=mode)
    distortedAlignment = f(np.linspace(-1, 1, n_frames))
    reverseAlignment = f_i(np.linspace(-1, 1, n_frames))
   
    fidelity = np.mean(np.abs(distortedAlignment - originalAlignment)) # always from 0 to 1

    return distortedAlignment, reverseAlignment, fidelity


def getRandomDistortion4(n_frames, distPoints=5, mode='linear'):
    originalAlignment = np.linspace(-1,1,n_frames)

    # distortedAlignment = np.zeros(n_frames)#np.linspace(-1,1,n_frames)
    
    distPoints = 3 + distPoints

    newPoints = np.random.rand(distPoints)
    newPoints = np.sort(
                    (newPoints - np.min(newPoints)) * 2 /
                    (np.max(newPoints) - np.min(newPoints)) - 1)
    f = interpolate.interp1d(np.linspace(-1, 1, len(newPoints)), newPoints, kind=mode)
    f_i = interpolate.interp1d(newPoints, np.linspace(-1, 1, len(newPoints)), kind=mode)
    distortedAlignment = f(np.linspace(-1, 1, n_frames))
    reverseAlignment = f_i(np.linspace(-1, 1, n_frames))
   
    fidelity = np.mean(np.abs(distortedAlignment - originalAlignment)) # always from 0 to 1

    return distortedAlignment, reverseAlignment, fidelity
def distortCurve2(curve, distortion, nSamples):
    # distort the curve
    curveT = torch.tensor(curve).view(1,1,nSamples,1).float()
    grid = torch.zeros(1,nSamples,1,2).float() - 1
    grid[0,:,0,1] = torch.from_numpy(distortion)
    curveDistT = F.grid_sample(curveT, grid, align_corners=True,padding_mode="reflection")
    return curveDistT.detach().squeeze().numpy()
#%%
def convLowPass(x, kernel = 101, returnNumpy = False):
    kernel = int((kernel//2)*2 + 1)
    weights = nn.Parameter(torch.tensor([1/kernel for i in range(kernel)]).view(1,1,kernel))
    convSmooth = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, padding = kernel//2,padding_mode = 'reflect', bias=False)
    convSmooth.weight = weights
    convSmooth.requires_grad = False

    # distortedAlignmentSmooth = convSmooth(torch.flip(torch.tensor(distortedAlignment).view(1,1,-1).float(), [2]))
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    y = convSmooth(x.view(1,1,-1).float())
    if returnNumpy is True:
        return y.squeeze().detach().numpy()
    return y


#%%

def getRandomDistortion(n_frames, distRate=0.5, fidelity = 0.5, minDist=0, maxDist=1, areaDist=0, mode='linear'):
    originalAlignment = np.linspace(-1,1,n_frames)
    distortedAlignment = np.zeros(n_frames)#np.linspace(-1,1,n_frames)
    
    while np.max(np.abs(distortedAlignment - originalAlignment)) >= maxDist or np.max(np.abs(distortedAlignment - originalAlignment)) <= minDist :        
    # while np.mean(np.abs(np.zeros(n_frames) - originalAlignment)) < 1 :
        newPoints = np.random.rand(int(n_frames*distRate))
        newPoints = np.sort(
                        (newPoints - np.min(newPoints)) * 2 /
                        (np.max(newPoints) - np.min(newPoints)) - 1)
        f = interpolate.interp1d(np.linspace(-1, 1, len(newPoints)), newPoints, kind=mode)
        f_i = interpolate.interp1d(newPoints, np.linspace(-1, 1, len(newPoints)), kind=mode)
        distortedAlignment = f(np.linspace(-1, 1, n_frames))
        reverseAlignment = f_i(np.linspace(-1, 1, n_frames))
        # print(np.max(np.abs(distortedAlignment - originalAlignment)))
    # plt.plot(np.linspace(-1,1,n_frames), distortedAlignment )
    # plt.plot(np.linspace(-1,1,n_frames), reverseAlignment )


    # TODO forget smoothing for now. It's not easy to smooth the reverse one. 
    # kernel = 201
    # weights = nn.Parameter(torch.tensor([1/kernel for i in range(kernel)]).view(1,1,kernel))
    # convSmooth = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, padding = kernel//2,padding_mode = 'reflect', bias=False)
    # convSmooth.weight = weights
    # # distortedAlignmentSmooth = convSmooth(torch.flip(torch.tensor(distortedAlignment).view(1,1,-1).float(), [2]))
    # distortedAlignmentSmooth = convSmooth(torch.tensor(distortedAlignment).view(1,1,-1).float())
    # reverseAlignmentSmooth = convSmooth(torch.tensor(reverseAlignment).view(1,1,-1).float())


    # plt.plot(originalAlignment, distortedAlignment)
    # plt.plot(originalAlignment, distortedAlignmentSmooth[0,0,:].detach().numpy())
    # plt.plot(originalAlignment, reverseAlignment)
    # plt.plot(originalAlignment, reverseAlignmentSmooth[0,0,:].detach().numpy())
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.draw()

    # plt.figure()
    # plt.plot(distortedAlignment - distortedAlignmentSmooth[0,0,:].detach().numpy())
    return distortedAlignment, reverseAlignment #, distortedAlignmentSmooth[0,0,:].detach().numpy()

# %%


#%%
def distortCurve(curve, distortedAlignment):
    floatIndices = (distortedAlignment + 1) / 2 * (len(curve)-1)
    floorIndices = np.floor(floatIndices).astype(np.int)
    ceilIndices = np.ceil(floatIndices).astype(np.int)
    ceil2Indices = floorIndices + 1
    curveDist = curve[floorIndices]*(ceil2Indices-floatIndices) + curve[ceilIndices]*(floatIndices-floorIndices)
    # plt.figure()
    # plt.plot(curveDist)
    # plt.figure()
    # plt.plot(distortedAlignment)
    return curveDist

def getMeasuresMap(measuresVector):
    # input must be np.array 1xlength, not just ,length
    measuresMap = collections.OrderedDict()
    indices = np.where(measuresVector.astype(np.int8)==1)[1]
    # if indices[0]!=0:
    #     pickupBegin = True   
    for i, ind in enumerate(indices):
        
        if i == len(indices)-1:
            nextInd = len(measuresVector)
        else : 
            nextInd = indices[i+1]
        measuresMap[i] = (ind, nextInd)
    return measuresMap

class SmoothInterp():
    def __init__(self, x,y,p,s, measureLength, endPoint=True):
        # x = indices of given points
        # y = values of given points
        self.funcList = []
        self.indList = []
        self.x = x
        self.y = y
        self.p = p # [0.01, 0.99] depends on the style of the user
        self.s = s # [0, 0.99] the same
        self.c = 2/(1-s) - 1
        self.measureLength = measureLength
        if endPoint == True :
            self.x.append(self.measureLength-1)
            self.y.append(self.y[-1])
        for i in range(len(self.x)-1):
            left = self.x[i]
            right = self.x[i+1]
            middle = p*left + (1-p)*right
            len1 = middle - left
            len2 = right - middle 
            lenTotal = right - left
            midi1 = self.y[i]
            midi2 = self.y[i+1]
            h = midi2 - midi1
            
            # f = self.myFunc(h, self.c, lenTotal)
            f = lambda x, n, h=h, c=self.c, l=lenTotal : ((h*x**c)/(n**(c-1)))/(l-0)
            # f = lambda x, n, h=h, l=lenTotal : ((h*x**3)/(n**(3-1)))/(l-0)
            l1 = lambda x, len1=len1, f=f, midi1 = midi1 : f(x,len1) + midi1
            l2 = lambda x, lenTotal=lenTotal, f=f, len1=len1, midi1=midi1, h=h: 1 - f(lenTotal-x, lenTotal-len1) + h - 1 + midi1
            # func = f(x,m)
            # self.funcList.append({"range":[left,middle],"func":l1})
            # self.funcList.append({"range":[middle,right],"func":l2})

            # print(f"left {left} right {right} middle {middle} midi1 {midi1} midi2 {midi2}")
            # print(f"l1(left) {l1(left - left)}")
            # print(f"l1(middle) {l1(middle - left)}")
            # print(f"l2(middle) {l2(middle - left)}")
            # print(f"l2(right) {l2(right - left)}")
            # if i ==0 :
            #     for j in range(11):
            #         print(l1(j))
            #     print("\n")
            #     for j in range(11):
            #         print(l2(j))
            # print(f"test low {midi1} : {l1(left)}  middle1 {middle} : {l1(middle)} middle2 {middle} : {l2(middle)} high {midi2} : {l2(right)}")
            self.indList.extend([left, middle])
            self.funcList.extend([l1,l2])
            # del l1, l2
    def __call__(self, newXs):
        # TODO x should be normalized from 0 to 1 ? In this case I need to store the initial size of the measure (i.e 96)
        # if newX<=1 : 
        #     newX *= self.measureLength
        if not isinstance(newXs, list):
            newXs = [newXs]
        if newXs[-1] >= self.measureLength:
            assert(False)
        out = []
        for newX in newXs: 
            ind = bisect.bisect_right(self.indList, newX) - 1
            ind2 = bisect.bisect_right(self.x, newX) - 1
            # print(f"ind {ind} newX {newX} real {newX - self.x[ind2]}")
            out.append(self.funcList[ind](newX - self.x[ind2]))
        return out
    def myFunc(self,h,c,l):
        print(f"myFunc h {h} c {c} l {l}")
        return lambda x, n, h=h, c=c, l=l : ((h*x**c)/(n**(c-1)))/(l-0)
