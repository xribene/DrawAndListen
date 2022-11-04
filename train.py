#%%
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utilsTraining import MinExponentialLR, batch_inference, batch_random_inference, moduleGradMean, get_lr
import platform
from datetime import datetime
from tensorboard import program
import numpy as np
from model import ModelFullEnd2End as Model
from irishDataloading import irishSplitter, RandomDataset
# CUDA_LAUNCH_BLOCKING=1
#%%

# From generic random curves to specific irish notes
if __name__ == "__main__":
    hostName = platform.uname().node
    method = 'Curve2Melody'
    ######################################################################################
    # Load Args and Params
    ######################################################################################
    parser = argparse.ArgumentParser(description='Train Arguments')
    parser.add_argument("--saveFolder", "-f", type=str, default='')  
    parser.add_argument("--sessionName", type=str, default='')
    parser.add_argument("--info", type=str, default = '')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--wdecay', type=float, default=1e-6 )  
    parser.add_argument('--tfDecay', type=float, default=10000 )  
    parser.add_argument('--minTf', type=float, default=0.25 )  

    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epochsNum', type=int, default=100) 
    parser.add_argument('--savePeriod', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrFinal', type=float, default=1e-5)

    parser.add_argument('--lrDecay', type=float, default=0.999975) 
    parser.add_argument('--densityType', type=str, default='quant') # quantNorm

    parser.add_argument('--size', type=int, default = 0)

    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--vaeBeta', type=float, default=0.1)
    # parser.add_argument('--hiddenDims', type=int, default=2048) 
    parser.add_argument('--trainDecoder', type=int, default=0)
    parser.add_argument('--detaching', type=int, default=0)
    parser.add_argument('--randomDataProb', type=float, default=0.1) 

    # parser.add_argument('--seq', type=int, default=0)

    parser.add_argument('--splitCurveEnc', type=int, default=0)
    parser.add_argument('--musicDecoderStatus', type=str, default='finetune') # frozen, fineTune, train
    parser.add_argument('--musicEncoderStatus', type=str, default='frozen') # frozen, fineTune, train
    parser.add_argument('--trainFull', type=int, default=0) # frozen, fineTune, train

    parser.add_argument('--includeCE', type=int, default = 1)
    parser.add_argument('--dirWeight', type=float, default = 1.0)
    parser.add_argument('--rhyWeight', type=float, default = 1.0)


    parser.add_argument('--rsample', type=int, default = 1)

    parser.add_argument('--zDirGt', type=int, default = 0)
    parser.add_argument('--zRhyGt', type=int, default = 0)
    parser.add_argument('--special', type=int, default = 0)
    parser.add_argument('--diff', type=int, default = 0)

    parser.add_argument('--onsetDistProb', type=float, default = 0.0)
    parser.add_argument('--pitchDistProb', type=float, default = 0.0)
    parser.add_argument('--curveDistProb', type=float, default = 0.0)


    # Curve networks parameters
    parser.add_argument('--channels', type=int, default=16) 
    parser.add_argument('--kernel', type=int, default=3) 
    parser.add_argument('--encoderType', type=str, default='CNN')
    parser.add_argument('--decoderType', type=str, default='CNN')
    parser.add_argument('--nSamples', type=int, default=512) 
    parser.add_argument('--decoderDil', type=int, default=0) 
    parser.add_argument('--encoderDil', type=int, default=0) 
    parser.add_argument('--layers', type=int, default=7) 
    parser.add_argument('--upMode', type=str, default='linear')
    parser.add_argument('--batchNorm', type=int, default = 1)
    parser.add_argument('--pooling', type=str, default='None')
    parser.add_argument('--activation', type=str, default='lrelu')
    parser.add_argument('--finalActivation', type=str, default='None')

    # parser.add_argument('--ablationInd', type=int, default=1)
    # parser.add_argument('--unicorn', type=int, default=1)

    parser.add_argument('--contextRhythmLink', type=int, default = 1)
    parser.add_argument('--embeddings', type=int, default = 1)
    parser.add_argument('--concatCond', type=int, default = 1)
    parser.add_argument('--transpose', type=int, default = 1)

    parser.add_argument('--seq', type=int, default=0)

    parser.add_argument('--condDims', type=int, default = 3)
    parser.add_argument('--condType', type=str, default='mean') # it can be mean also

    parser.add_argument('--dirDims', type=int, default=7)
    parser.add_argument('--dirType', type=str, default='bugFix') # it can be simple (linear interp)

    parser.add_argument('--hiddenDims', type=int, default=2048) 
    parser.add_argument('--decHiddenDims', type=int, default=2048) 
    parser.add_argument('--encHiddenDims', type=int, default=2048) 
    parser.add_argument('--dirHiddenDims', type=int, default=2048) 
    parser.add_argument('--rhyHiddenDims', type=int, default=2048) 

    parser.add_argument('--decoderLayers', type=int, default=2) 

    parser.add_argument('--z1Dims', type=int, default=85)
    parser.add_argument('--z2Dims', type=int, default=85)
    parser.add_argument('--z3Dims', type=int, default=85) 

    parser.add_argument('--barsNum', type=int, default=3) 

    parser.add_argument('--pretrain', type=int, default=0) 
    parser.add_argument('--checkpointPath', type=str, default='04-05-2021_20:30:14_end2end_quantNorm_onsetOffset3_tied0_concat1_chroma1_Airgpustation2_Curve2Melody/modelEnd2End_end2end_quantNorm_onsetOffset3_tied0_concat1_chroma1_Airgpustation2_Curve2Melody_last.pt') 
    parser.add_argument('--chromaLossWeight', type=float, default=1) 
    parser.add_argument('--tiedEncoders', type=int, default=0) 
    parser.add_argument('--concatContext', type=int, default=1) 
    parser.add_argument('--onsetOffsetConds', type=int, default=0)# 3 
    parser.add_argument('--chromaLink', type=int, default=0) 







    cwd = Path.cwd()
    checkpoints = Path("Checkpoints")
    # ablationPath = Path("Ablation/MelodyVAE")
    # datasetPath = Path("Dataset/Irish")
    datasetPath = cwd
    if hostName == 'AIR-GPU-STATION':
        checkpoints = Path("/dataNVME/christos/Checkpoints")
        datasetPath = Path("/dataNVME/christos/Dataset/Irish")
        # ablationPath = Path("/dataNVME/christos/Ablation/MelodyVAE")

    elif hostName in ['airgpustation3','Airgpustation2']:
        checkpoints = Path("/storageNVME/christos/Checkpoints")
        datasetPath = Path("/storageNVME/christos/Dataset/Irish")
        # ablationPath = Path("/storageNVME/christos/Ablation/MelodyVAE")



    args = parser.parse_args()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Using cuda {args.cuda}', torch.cuda.get_device_name(device))
    # device = torch.device("cpu")

    now = datetime.now()
    sessionInfo = f"{args.sessionName}_{hostName}_{method}"
    saveFolder =  checkpoints / Path(now.strftime("%d-%m-%Y_%H:%M:%S_") + sessionInfo)

    print(saveFolder)
    writer = SummaryWriter(saveFolder / "logs") 
#%%
    if args.pretrain == 1:
        checkpointPreviousPath = checkpoints / args.checkpointPath
        checkpointPretrained = torch.load(checkpointPreviousPath, map_location=device)
        argsMusicVae = checkpointPretrained['args']
    else:
        argsMusicVae = args

    concatCond = argsMusicVae.concatCond

    transpose = argsMusicVae.transpose
    condDims = argsMusicVae.condDims
    dirDims = argsMusicVae.dirDims
    hiddenDims = argsMusicVae.hiddenDims

    z1Dims = argsMusicVae.z1Dims
    z2Dims = argsMusicVae.z2Dims
    z3Dims = argsMusicVae.z3Dims

    inputDims = 130
    rhythmDims = 3
    seqLen = 6 * 4
    cpcDims = 13

    finalActivation = args.finalActivation
    if finalActivation == 'None':
        finalActivation = None
    pooling = args.pooling
    if pooling == 'None':
        pooling = None

    musicVae = Model(device, hidden_dims = argsMusicVae.hiddenDims, dir_dims = argsMusicVae.dirDims,
                    cond_dims = argsMusicVae.condDims,
                    zCpc_dims = argsMusicVae.z1Dims, zRhy_dims = argsMusicVae.z2Dims, zDir_dims = argsMusicVae.z3Dims, 
                    rhythm_dims = 3, input_dims = 130, useCPC = 0, orderlessCpc = 0,
                    cpc_dims = cpcDims, seq_len = 24, decay = args.tfDecay, detaching = argsMusicVae.detaching, seq = argsMusicVae.seq,
                    decoderLayers = argsMusicVae.decoderLayers, dec_hidden_dims = argsMusicVae.decHiddenDims, 
                    enc_hidden_dims = argsMusicVae.encHiddenDims,
                    rhy_hidden_dims = argsMusicVae.rhyHiddenDims, dir_hidden_dims=argsMusicVae.dirHiddenDims,
                    concat_cond = concatCond,
                    contextRhythmLink = argsMusicVae.contextRhythmLink, 
                    embeddings = argsMusicVae.embeddings, embeddingsDim = 128,
                    tiedEncoders = args.tiedEncoders,
                    concatContext= args.concatContext,
                    onsetOffsetConds = args.onsetOffsetConds,
                    chromaLink = argsMusicVae.chromaLink,
                    minTf = args.minTf,
                    channels = argsMusicVae.channels, kernel = argsMusicVae.kernel, encoderType = 'CNN', nSamples = argsMusicVae.nSamples,
                    encoderDil = False, layers = argsMusicVae.layers, batchNorm = bool(argsMusicVae.batchNorm), pooling = None,
                    activation = 'lrelu'
                    )
    if args.pretrain == 1:
        musicVae.load_state_dict(checkpointPretrained['state_dict'])
        print("AAAAAAAAAAAA")
    musicVae.to(device)

    print(f"GRAND MEAN IS {moduleGradMean(musicVae)}")
    parametersList = list(musicVae.parameters())

    optimizer = optim.Adam(parametersList, lr = args.lr)

    if args.lrDecay > 0:
        scheduler = MinExponentialLR(optimizer, gamma = args.lrDecay, minimum = args.lrFinal)

#%%
    print(argsMusicVae.dirDims)
    if args.size == 0:
        args.size = None
    half = args.barsNum //2
    print(argsMusicVae)
    trainDataset, validDataset, testDataset = irishSplitter(datasetPath, size = args.size, includeCurves = True,
                                                            dirs = argsMusicVae.dirDims, 
                                                            conds = argsMusicVae.condDims,
                                                            transpose = argsMusicVae.transpose,
                                                            densityType = argsMusicVae.densityType,
                                                            dirType = argsMusicVae.dirType,
                                                            condType = argsMusicVae.condType,
                                                            onsetDistProb = args.onsetDistProb, 
                                                            pitchDistProb = args.pitchDistProb, 
                                                            curveDistProb = args.curveDistProb,
                                                            barsNum = args.barsNum
                                                            )
    print(f"dirType {argsMusicVae.dirType}  condType {argsMusicVae.condType}")
    print(len(trainDataset),len(validDataset),len(testDataset))
    print(argsMusicVae.transpose)

    
    trainLoader = DataLoader(trainDataset, batch_size=args.batchSize,
                            shuffle=True, num_workers=args.workers)
    validLoader = DataLoader(validDataset, batch_size=args.batchSize,
                            shuffle=True, num_workers=args.workers)
    validIter = iter(validLoader)
    trainIter = iter(trainLoader)

    testLoader = DataLoader(testDataset, batch_size=args.batchSize,
                            shuffle=False, num_workers=0)

    randomDataset = RandomDataset(p1=0.0, p2=0.0,
                                            includeCurves=True, densityType=args.densityType, 
                                            condType = argsMusicVae.condType, dirType = argsMusicVae.dirType, 
                                            curveOrigin = 'linear', dirs = argsMusicVae.dirDims, 
                                            conds = argsMusicVae.condDims, transpose = 1, nSamples=512,
                                            onsetDistProb = args.onsetDistProb, pitchDistProb = args.pitchDistProb, 
                                            curveDistProb = args.curveDistProb)
    randomLoader = DataLoader(randomDataset, shuffle=False, batch_size=args.batchSize, num_workers=args.workers)
    randomIter = iter(randomLoader)

#%%
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(saveFolder/'logs')])
    url = tb.launch()
    print(url)

#%%
    print(sum(p.numel() for p in musicVae.curveEncoder.parameters() if p.requires_grad))
    epoch = 0
    step=0
    if args.pretrain == 1:
        epoch += checkpointPretrained['epoch']
        step += checkpointPretrained['step']
    while epoch < 10000:
        epoch += 1
        with tqdm(trainLoader, unit="batch") as t:

            for batchInd in range(len(trainLoader)):

                if np.random.random() < args.randomDataProb:
                    irishBatch = False
                    try:
                        batchRandom = next(randomIter)
                    except:
                        randomIter = iter(randomLoader)
                        batchRandom = next(randomIter)
                    batch = batchRandom
                else:
                    irishBatch = True
                    try:
                        batchIrish = next(trainIter)
                    except:
                        trainIter = iter(trainLoader)
                        batchIrish = next(trainIter)
                    batch = batchIrish



                try:
                    batchValid = next(validIter)
                except:
                    validIter = iter(validLoader)
                    batchValid = next(validIter)

                musicVae.train()
                optimizer.zero_grad()   

                if irishBatch is True:
                    out = batch_inference(batch, musicVae, args, device)
                    out['loss']['total'].backward()

                else:
                    outRandom = batch_random_inference(batch, musicVae, args, device)
                    outRandom['loss']['total'].backward()

                # torch.nn.utils.clip_grad_norm_(musicVae.parameters(), 1)

                optimizer.step()

                # VALIDATION 
                musicVae.eval()
                with torch.no_grad():
                    outVal = batch_inference(batchValid, musicVae, args, device)
                
                step += 1
                if args.lrDecay > 0:
                    scheduler.step()

                if irishBatch:
                    writer.add_scalars('totalLoss', {'train':out['loss']['total'].item(),
                                                    'valid':outVal['loss']['total'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('klLoss', {'train':out['loss']['kl'].item(),
                                                'valid':outVal['loss']['kl'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('reconLoss', {'train':out['loss']['melody'].item(),
                                                    'valid':outVal['loss']['melody'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('chromaLoss', {'train':out['loss']['chroma'].item(),
                                                    'valid':outVal['loss']['chroma'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('rhythmLoss',{'train':out['loss']['rhythm'].item(),
                                                    'valid':outVal['loss']['rhythm'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('dirLoss',  {'train':out['loss']['dir'].item(),
                                                    'valid':outVal['loss']['dir'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('accMelody', {'train':out['acc']['melody'].item(),
                                                    'valid':outVal['acc']['melody'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('accRhy', {
                                                    'train':out['acc']['rhythm'].item(),
                                                    'valid':outVal['acc']['rhythm'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('accDir', {
                                                    'train':out['acc']['dir'].item(),
                                                    'valid':outVal['acc']['dir'].item()
                                                    }, 
                                                    step)
                    writer.add_scalars('modelParams', {'eps': musicVae.eps,
                                                    'beta': args.vaeBeta,
                                                    'lr': optimizer.param_groups[0]['lr'],
                                                    'lr2' : get_lr(optimizer)
                                                    }, 
                                                    step)
                else:
                    writer.add_scalars('klLossRR', {'train':outRandom['loss']['kl'].item()}, step)
                    writer.add_scalars('dirLossRR',  {'train':outRandom['loss']['dir'].item()}, step)
                    writer.add_scalars('accDirRR', { 'train':outRandom['acc']['dir'].item()}, step)
                
                t.update(1)

        if (epoch + 1) % args.savePeriod == 0:
            if args.pretrain == 1:
                filename = f"{sessionInfo}_fromCheckpoint_last.pt"
            else:
                filename = f"{sessionInfo}_last.pt" # 
                   
            state1 = {
                        'epoch': epoch,
                        'state_dict': musicVae.state_dict(),
                        # 'minValLoss': minValLoss,
                        # 'optimizer': optimizer.state_dict(),
                        'args': args, 
                        'argsMusicVae':argsMusicVae,
                        'checkpointPath':saveFolder,
                        'step': step,
                        'lr' : get_lr(optimizer),
                        # 'argsHist' : prevArgs.append(args),
                        # 'epocs' : epocLocal,
                        'info' : args.info
                    }
            torch.save(state1, Path(saveFolder) / ("modelEnd2End_"+filename))

    writer.close()
    print(checkpoints)
    print(url)