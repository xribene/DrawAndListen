from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
import torch

def paramsGradMean(params):
    mean = 0
    for p in params:

        # if p.grad is not None:
        #     mean += torch.mean(p.grad)
        # else:
        #     mean += 0

        mean += torch.mean(p)
    return mean

def moduleGradMean(module):
    mean = 0
    for n, p in module.named_parameters():

        if p.grad is not None:
            mean += torch.mean(torch.abs(p.grad))
        else:
            mean += 0

        mean += torch.mean(p)
    return mean

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_random_inference(batch, model, args, device):
    half = args.barsNum // 2
    batchCurrent = batch # 0 is the key for current bar
    currentBatchSize = batchCurrent['curve'].shape[0]
    pitchCurve = batchCurrent['curve'].view(currentBatchSize, 1, -1).float().to(device)

    onsetDensity = batchCurrent['onsetDensity'].view(currentBatchSize, 1, -1).float().to(device)
    onsetOffset = batchCurrent['onsetOffset'].view(currentBatchSize, 1).float().to(device)

    target_dirOH = batchCurrent[f'dirs{args.dirDims}OH'].float().to(device)

    recon_dir, dis3 = model(None, None,None, pitchCurve, onsetDensity,
                                    None, target_dirOH, None, None, None, None,
                                    randomBatch = True)

    target_dir = target_dirOH.view(-1, target_dirOH.size(-1)).max(-1)[1]

    dir_CE = F.nll_loss(recon_dir.view(-1, recon_dir.size(-1)), target_dir, reduction = "mean")

    max_indices_dir = recon_dir.view(-1, recon_dir.size(-1)).max(-1)[-1]
    correct_dir = max_indices_dir == target_dir
    acc_dir = torch.sum(correct_dir.float()) / target_dir.size(0)
    
    normal3 =  std_normal(dis3.mean.size(), device)

    KLD3 = kl_divergence(dis3, normal3).mean()

    klLossA = KLD3

    total_loss = args.dirWeight*dir_CE + args.vaeBeta*klLossA
    return {'loss':{'total' : total_loss,
                    'kl' : klLossA, 
                    'dir' : dir_CE},
            'acc' : {'dir' : acc_dir}} 

def batch_inference(batch, model, args, device):
    half = args.barsNum // 2
    batchCurrent = batch[0] # 0 is the key for current bar
    batchContext = [batch[jj] for jj in range(-half,half+1) if jj != 0]
    listOfConds = [bb['condOH'].float().to(device) for bb in batchContext]
    listOfContextMIDI = [bb['midisHoldsNoRestIrregOH'].float().to(device) for bb in batchContext]
    listOfContextCPC = [bb['cpcsHoldsNoRestIrregOH'].float().to(device) for bb in batchContext]

    currentBatchSize = batchCurrent['curve'].shape[0]

    pitchCurve = batchCurrent['curve'].view(currentBatchSize, 1, -1).float().to(device)

    onsetDensity = batchCurrent['onsetDensity'].view(currentBatchSize, 1, -1).float().to(device)
    # print(batchCurrent.keys())
    onsetOffsetOH = batchCurrent['onsetOffsetOH'].float().to(device)
    # onsetDensity -= onsetOffset
    # onsetDensity *= 0
    # pitchCurve *= 0
    # midis = batchCurrent['midisHoldsNoRestIrreg'].float().to(device)
    midisOH = batchCurrent['midisHoldsNoRestIrregOH'].float().to(device)
    condOH = batchCurrent['condOH'].float().to(device)
    target = midisOH.view(-1, midisOH.size(-1)).max(-1)[1]

    target_rhythmOH = batchCurrent['rhythmOH'].float().to(device)
    target_dirOH = batchCurrent[f'dirs{args.dirDims}OH'].float().to(device)

    # create target_cpc OH
    # very special case. wouldn't work in any other case
    # I omit the last 8 midi numbers
    # tmp = inp # not used
    # target_cpc = tmp[:,:,:120].view(-1, 24, 10, 12).sum(axis=2)
    cpcs = batchCurrent['cpcsHoldsNoRestIrreg'].float().to(device)
    target_cpcOH = batchCurrent['cpcsHoldsNoRestIrregOH'].float().to(device)
    # print(f"GRAND MEAN IS {moduleGradMean(musicVae)}")
    # print(pitchCurve.mean())
    # print(onsetDensity.mean())
    recon, recon_cpc, recon_rhythm, recon_dir, dis1, dis2, dis3, chromaVector = model(midisOH, listOfContextMIDI, listOfContextCPC, 
                                                                pitchCurve, onsetDensity,
                                                                target_rhythmOH, target_dirOH, target_cpcOH, condOH, onsetOffsetOH, listOfConds,
                                                                randomBatch = False)


    target_dir = target_dirOH.view(-1, target_dirOH.size(-1)).max(-1)[1]
    target_rhythm = target_rhythmOH.view(-1, target_rhythmOH.size(-1)).max(-1)[1]
    target_cpc = target_cpcOH.view(-1, target_cpcOH.size(-1)).max(-1)[1]
    
    targetChroma = target_cpcOH.max(dim=1)[0][:,1:]

    # acc, recon_loss, cpc_loss, rhythm_loss, dir_loss, kl_loss, total_loss = loss_function_cpc(recon, recon_cpc, recon_rhythm, recon_dir, target, target_cpc, target_rhythm, target_dir, dis1MusicVae, dis2Link, dis3Link, args.vaeBeta, device)
    # reconstruction Losses
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    dir_CE = F.nll_loss(recon_dir.view(-1, recon_dir.size(-1)), target_dir, reduction = "mean")

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)

    max_indices_rhy = recon_rhythm.view(-1, recon_rhythm.size(-1)).max(-1)[-1]
    correct_rhy = max_indices_rhy == target_rhythm
    acc_rhy = torch.sum(correct_rhy.float()) / target_rhythm.size(0)

    max_indices_dir = recon_dir.view(-1, recon_dir.size(-1)).max(-1)[-1]
    correct_dir = max_indices_dir == target_dir
    acc_dir = torch.sum(correct_dir.float()) / target_dir.size(0)
    
    normal1 =  std_normal(dis1.mean.size(), device)
    normal2 =  std_normal(dis2.mean.size(), device)
    normal3 =  std_normal(dis3.mean.size(), device)

    KLD1 = kl_divergence(dis1, normal1).mean()

    KLD2 = kl_divergence(dis2, normal2).mean()
    KLD3 = kl_divergence(dis3, normal3).mean()
    klLossA = KLD1 + KLD2 + KLD3

    chromaLoss = F.binary_cross_entropy_with_logits(chromaVector, targetChroma)
    # chromaLoss = chromaCriterion(chromaVector, targetChroma)

    total_loss = args.includeCE*CE + args.rhyWeight*rhy_CE + args.dirWeight*dir_CE + args.vaeBeta*klLossA + args.chromaLossWeight*chromaLoss
    return {'loss':{'total' : total_loss,
                    'chroma' : chromaLoss,
                    'kl' : klLossA, 
                    'melody' : CE,
                    'rhythm' : rhy_CE,
                    'dir' : dir_CE},
            'acc' : {'melody' : acc,
                    'rhythm' : acc_rhy,
                    'dir' : acc_dir}}
# loss function
def std_normal(shape, device):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.to(device)
        N.scale = N.scale.to(device)
    return N

def klLoss(q, device):
    normal = std_normal(q.mean.size(), device)
    return kl_divergence(q, normal).mean()
    
def loss_function(recon, recon_cpc, recon_rhythm, recon_dir, target, target_cpc, target_rhythm, target_dir, dis1, dis2, dis3, beta, device,dirWeight=1.0, randomBatch = False):
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    dir_CE = F.nll_loss(recon_dir.view(-1, recon_dir.size(-1)), target_dir, reduction = "mean")

    
    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)

    max_indices_rhy = recon_rhythm.view(-1, recon_rhythm.size(-1)).max(-1)[-1]
    correct_rhy = max_indices_rhy == target_rhythm
    acc_rhy = torch.sum(correct_rhy.float()) / target_rhythm.size(0)

    max_indices_dir = recon_dir.view(-1, recon_dir.size(-1)).max(-1)[-1]
    correct_dir = max_indices_dir == target_dir
    acc_dir = torch.sum(correct_dir.float()) / target_dir.size(0)


    normal1 = std_normal(dis1.mean.size(), device)
    normal2 =  std_normal(dis2.mean.size(), device)
    normal3 =  std_normal(dis3.mean.size(), device)

    KLD1 = kl_divergence(dis1, normal1).mean()
    KLD2 = kl_divergence(dis2, normal2).mean()
    KLD3 = kl_divergence(dis3, normal3).mean()

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    if randomBatch is True:
        klLoss  = KLD2  + KLD3
        total = rhy_CE + dirWeight*dir_CE + beta * klLoss
    else:
        klLoss = KLD1 + KLD2 + KLD3
        total =  CE + rhy_CE + dirWeight*dir_CE + beta * klLoss
    return acc, acc_rhy, acc_dir, CE, 0, rhy_CE, dir_CE, klLoss, total

def loss_function_cpc(recon, recon_cpc, recon_rhythm, recon_dir, target, target_cpc, target_rhythm, target_dir, dis1, dis2, dis3, beta, device,dirWeight=1.0):
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    dir_CE = F.nll_loss(recon_dir.view(-1, recon_dir.size(-1)), target_dir, reduction = "mean")
    cpc_CE = F.nll_loss(recon_cpc.view(-1, recon_cpc.size(-1)), target_cpc, reduction = "mean")

    normal1 = std_normal(dis1.mean.size(), device)
    normal2 =  std_normal(dis2.mean.size(), device)
    normal3 =  std_normal(dis3.mean.size(), device)

    KLD1 = kl_divergence(dis1, normal1).mean()
    KLD2 = kl_divergence(dis2, normal2).mean()
    KLD3 = kl_divergence(dis3, normal3).mean()

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    klLoss = KLD1 + KLD2 + KLD3
    return acc, CE, cpc_CE, rhy_CE, dir_CE, klLoss, CE + cpc_CE + rhy_CE + dirWeight*dir_CE + beta * (klLoss)

def loss_function_cpc_orderless(recon, recon_cpc_logits, recon_rhythm, recon_dir, target, 
                                target_cpc_orderless, target_rhythm, target_dir, dis1, dis2, dis3, beta, device, dirWeight=1.0):
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")
    dir_CE = F.nll_loss(recon_dir.view(-1, recon_dir.size(-1)), target_dir, reduction = "mean")
    # cpc_CE = F.nll_loss(recon_cpc.view(-1, recon_cpc.size(-1)), target_cpc, reduction = "mean")
    cpc_BCE = F.binary_cross_entropy_with_logits(recon_cpc_logits.squeeze(1), target_cpc_orderless)

    normal1 = std_normal(dis1.mean.size(), device)
    normal2 =  std_normal(dis2.mean.size(), device)
    normal3 =  std_normal(dis3.mean.size(), device)

    KLD1 = kl_divergence(dis1, normal1).mean()
    KLD2 = kl_divergence(dis2, normal2).mean()
    KLD3 = kl_divergence(dis3, normal3).mean()

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    klLoss = KLD1 + KLD2 + KLD3
    return acc, CE, cpc_BCE, rhy_CE, dir_CE, klLoss, CE + cpc_BCE + rhy_CE + dirWeight*dir_CE + beta * (klLoss)

def loss_function_original(recon, recon_rhythm, target, target_rhythm, dis1, dis2, beta, device):
    CE = F.nll_loss(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    rhy_CE = F.nll_loss(recon_rhythm.view(-1, recon_rhythm.size(-1)), target_rhythm, reduction = "mean")

    normal1 = std_normal(dis1.mean.size(), device)
    normal2 =  std_normal(dis2.mean.size(), device)

    KLD1 = kl_divergence(dis1, normal1).mean()
    KLD2 = kl_divergence(dis2, normal2).mean()

    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
  
    return acc, CE, rhy_CE, CE + rhy_CE + beta * (KLD1 + KLD2)
class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

