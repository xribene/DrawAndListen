import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from cnnModels import CurveEncoder

class ModelFullEnd2End(nn.Module):
    def __init__(self, device, hidden_dims, dir_dims ,cond_dims,
                zCpc_dims,zRhy_dims, zDir_dims, rhythm_dims = 3, input_dims = 130, useCPC = 0, orderlessCpc = 1,
                cpc_dims = 13, seq_len = 24, decay = 1000, minTf=0.5, detaching = 1, seq = 1,
                decoderLayers = 2, dec_hidden_dims = 2048, enc_hidden_dims = 2048,
                rhy_hidden_dims = 2048, dir_hidden_dims=2048, concat_cond = 0, contextRhythmLink = 1, 
                embeddings = 0, embeddingsDim = 128, concatContext = 0, tiedEncoders = 0,
                onsetOffsetConds = 0, chromaLink = 0, 
                channels = 16, kernel = 3, encoderType = 'CNN', nSamples = 512,
                encoderDil = False, layers = 7, batchNorm = True, pooling = None,
                activation = 'lrelu'
                ):
        super(ModelFullEnd2End, self).__init__()
        self.detaching = detaching
        self.seq = seq
        self.decoderLayers = decoderLayers
        self.useCpc = useCPC
        self.orderless = orderlessCpc
        self.contextRhythmLink = contextRhythmLink
        self.embeddings = embeddings
        self.contextEncoderParameters = []
        self.curveEncoderParameters = []
        self.decoderParameters = []
        self.finalDecoderParameters = []
        self.rhythmDecoderParameters = []
        self.dirDecoderParameters = []
        self.concatContext = concatContext
        self.tiedEncoders = tiedEncoders
        self.onsetOffsetConds = onsetOffsetConds
        self.chromaLink = chromaLink
        # context encoder
        self.device = device

        if embeddings == 1:
            self.midiEmbedding = nn.Embedding(input_dims, embeddingsDim)
            # input_dims = embeddingsDim
            self.cpcEmbedding = nn.Embedding(cpc_dims, embeddingsDim)
            # input_dims = embeddingsDim

        if self.tiedEncoders == 1:
            self.encoder_gru = nn.GRU(embeddingsDim , enc_hidden_dims, batch_first=True, bidirectional=True) # cond_dims
            self.contextEncoderParameters += list(self.encoder_gru.parameters())
        elif self.tiedEncoders == 0:
            self.encoder_gru_past = nn.GRU(embeddingsDim , enc_hidden_dims, batch_first=True, bidirectional=True) # cond_dims
            self.encoder_gru_future = nn.GRU(embeddingsDim , enc_hidden_dims, batch_first=True, bidirectional=True) # cond_dims
            self.contextEncoderParameters += list(self.encoder_gru_past.parameters())
            self.contextEncoderParameters += list(self.encoder_gru_future.parameters())
        if self.concatContext == 1:
            self.linear_mu = nn.Linear(enc_hidden_dims * 2 * 2, zCpc_dims )
            self.linear_var = nn.Linear(enc_hidden_dims * 2 * 2, zCpc_dims )
        elif self.concatContext == 0:
            self.linear_mu = nn.Linear(enc_hidden_dims * 2, zCpc_dims )
            self.linear_var = nn.Linear(enc_hidden_dims * 2, zCpc_dims )

        
        self.contextEncoderParameters += list(self.linear_var.parameters())
        self.contextEncoderParameters += list(self.linear_mu.parameters())

        self.chromaLayer = nn.Linear(zCpc_dims, 12)
        # curves encoder

        # inpChan = 2
        # self.curveEncoder = CurveEncoder(device, inpChan=inpChan, baseChan= channels, kernel = kernel, 
        #                                 zDims= zRhy_dims + zDir_dims, 
        #                                 encoderType = encoderType, decoderType = None,
        #                                 nSamples = nSamples, decoderDilation=False, 
        #                                 encoderDilation = encoderDil, layers = layers, dropout = 0.0,
        #                                 mode=None, bottleneck=True, norm = batchNorm, pooling=pooling, 
        #                                 activation = activation, finalActivation=None)
        inpChan = 2
        self.curveEncoder = CurveEncoder(device, inpChan=inpChan, baseChan= channels, kernel = kernel, zDims= zRhy_dims + zDir_dims, 
                                        encoderType = encoderType, 
                                        decoderType = None, nSamples = nSamples, decoderDilation=False, encoderDilation = False, 
                                        layers = layers, dropout = 0.0,
                                        mode=None, bottleneck=True, norm = batchNorm, pooling=pooling, activation = 'lrelu', 
                                        finalActivation=None)

        # print(f"inside model init {sum(p.numel() for p in self.curveEncoder.parameters() if p.requires_grad)}")

        self.curveEncoderParameters += list(self.curveEncoder.parameters())
        finalDecoderInputDims = 0

        # rhythm decoder
        self.rdecoder_0 = nn.GRUCell(zRhy_dims + rhythm_dims + contextRhythmLink*zCpc_dims + onsetOffsetConds,rhy_hidden_dims)
        self.rdecoder_hidden_init = nn.Linear(zRhy_dims + contextRhythmLink*zCpc_dims + onsetOffsetConds, rhy_hidden_dims)
        self.rdecoder_out = nn.Linear(rhy_hidden_dims, rhythm_dims)
        self.decoderParameters += list(self.rdecoder_0.parameters())
        self.decoderParameters += list(self.rdecoder_hidden_init.parameters())
        self.decoderParameters += list(self.rdecoder_out.parameters())
        self.rhythmDecoderParameters += list(self.rdecoder_0.parameters()) + list(self.rdecoder_hidden_init.parameters()) + list(self.rdecoder_out.parameters())


        # dir decoder
        if self.seq == 1:
            self.dirdecoder_0 = nn.GRUCell(zDir_dims + dir_dims + rhythm_dims, dir_hidden_dims)
        else:
            self.dirdecoder_0 = nn.GRUCell(zDir_dims + dir_dims, dir_hidden_dims)
        self.dirdecoder_hidden_init = nn.Linear(zDir_dims, dir_hidden_dims)
        self.dirdecoder_out = nn.Linear(dir_hidden_dims, dir_dims)
        self.decoderParameters += list(self.dirdecoder_0.parameters())
        self.decoderParameters += list(self.dirdecoder_hidden_init.parameters())
        self.decoderParameters += list(self.dirdecoder_out.parameters())
        self.dirDecoderParameters += list(self.dirdecoder_0.parameters()) + list(self.dirdecoder_hidden_init.parameters()) + list(self.dirdecoder_out.parameters())

        finalDecoderInputDims += rhythm_dims # 3
        finalDecoderInputDims += dir_dims # 7
        finalDecoderInputDims += zCpc_dims # 85
        finalDecoderInputDims += cond_dims # 3
        finalDecoderInputDims += input_dims # 128

        finalDecoderInputDims += chromaLink*12
        # print(f"finalDecoderInputDims {finalDecoderInputDims}")
        # final reconstruction decoder
        self.decoder_0 = nn.GRUCell(finalDecoderInputDims, dec_hidden_dims)
        if self.decoderLayers == 2:
            self.decoder_1 = nn.GRUCell(dec_hidden_dims, dec_hidden_dims)
            self.decoderParameters += list(self.decoder_1.parameters())
            self.finalDecoderParameters += list(self.decoder_1.parameters())

        self.decoder_hidden_init = nn.Linear(cond_dims + concat_cond*zCpc_dims + chromaLink*12, dec_hidden_dims)
        # print(f"inputShape for decoder HiddenInit is {cond_dims + concat_cond*zCpc_dims + chromaLink*12}")
        self.decoder_out = nn.Linear(dec_hidden_dims, input_dims)


        self.decoderParameters += list(self.decoder_0.parameters())
        self.decoderParameters += list(self.decoder_hidden_init.parameters())
        self.decoderParameters += list(self.decoder_out.parameters())

        self.finalDecoderParameters += list(self.decoder_0.parameters()) + list(self.decoder_hidden_init.parameters()) + list(self.decoder_out.parameters())

        # parameter initialization
        self.zCpc_dims = zCpc_dims
        self.zRhy_dims = zRhy_dims
        self.zDir_dims = zDir_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.rhythm_dims = rhythm_dims
        self.cpc_dims = cpc_dims
        self.dir_dims = dir_dims
        self.cond_dims = cond_dims
        self.seq_len = seq_len    
        self.rhy_hidden_dims = rhy_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims
        self.dir_hidden_dims = dir_hidden_dims
        self.concat_cond = concat_cond
        # input
        self.x = None
        self.rx = None
        self.dx = None
        self.cx = None
        # teacher forcing hyperparameters
        self.iteration = 0
        self.eps = 1.0
        self.decay = torch.FloatTensor([decay])
        self.minTf = minTf

    def _findmax(self, x):
        argx = x.argmax(1)
        x = torch.zeros_like(x)
        line = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            line = line.to(self.device)
        x[line, argx] = 1.0
        return x

    def contextEncoder(self, listOfContexts):
        if self.tiedEncoders == 1:
            past = self.encoder_gru(listOfContexts[0])[-1]
            future = self.encoder_gru(listOfContexts[1])[-1]
        elif self.tiedEncoders == 0:
            past = self.encoder_gru_past(listOfContexts[0])[-1]
            future = self.encoder_gru_future(listOfContexts[1])[-1]

        if self.concatContext == 0:
            # for i, context in enumerate(listOfContexts):
                
            #     if i == 0:  
            #         x = self.encoder_gru(context)[-1]
            #     else:
            #         x += self.encoder_gru(context)[-1]
            # print(x.size())
            # x = self.encoder_gru(listOfContexts[0])[-1]  # TODO fix that
            x = past/2 + future/2
        else:
            x = torch.cat((past,future),dim=0)
        x = x.transpose_(0,1).contiguous()
        x = x.view(x.size(0), -1)



        mu = self.linear_mu(x)
        var = self.linear_var(x)
        var = torch.nn.functional.softplus(var) + 1e-6
        # dis1 = Normal(mu[:,:self.zCpc_dims], var[:,:self.zCpc_dims])
        # dis2 = Normal(mu[:,self.zCpc_dims:-self.zDir_dims], var[:,self.zCpc_dims:-self.zDir_dims])
        # dis3 = Normal(mu[:,-self.zDir_dims:], var[:,-self.zDir_dims:])
        return Normal(mu, var)

    def runCurveEncoder(self, pitchCurve, onsetDensity):
        x = torch.cat((pitchCurve, onsetDensity),1)
        disCurve = self.curveEncoder.runEncoder(x)
        return disCurve




    def rhythm_decoder(self, z, zCpc,  onsetOffset, emergencyEval = 0):
        y = torch.zeros((z.size(0), self.rhythm_dims)).to(self.device)
        # for y_-1, it is rest
        y[:, -1] = 1
        ys = []
        condVector = z
        if self.contextRhythmLink == 1:
            condVector = torch.cat([condVector,zCpc],1)
        if self.onsetOffsetConds > 1:
            condVector = torch.cat([condVector,onsetOffset[:,0,:]],1)

        # if self.contextRhythmLink == 1:
        #     # print(z.shape)
        #     # print(zCpc.shape)
        #     h0 = torch.tanh(self.rdecoder_hidden_init(torch.cat([z,zCpc],1)))
        # else:
        #     h0 = torch.tanh(self.rdecoder_hidden_init(z))
        h0 = torch.tanh(self.rdecoder_hidden_init(condVector))
        hx = h0
        for i in range(self.seq_len):
            # if self.contextRhythmLink == 1:
            #     y = torch.cat([y, z, zCpc], 1)
            # else:
            #     y = torch.cat([y, z], 1)
            y = torch.cat([y, condVector], 1)
            hx = self.rdecoder_0(y, hx)
            y = F.log_softmax(self.rdecoder_out(hx), 1)
            ys.append(y)
            if self.training and emergencyEval == 0:
                p = torch.rand(1).item()
                if p < self.eps:
                    y = self.rx[:,i,:]
                else:
                    y = self._findmax(y)
            else:
                y = self._findmax(y)
        return torch.stack(ys,1)
    
    def dir_decoder(self, z, rhythm,  emergencyEval = 0):
        y = torch.zeros((z.size(0), self.dir_dims)).to(self.device)
        # for y_-1, it is rest
        y[:, -1] = 1
        ys = []
        h0 = torch.tanh(self.dirdecoder_hidden_init(z))
        hx = h0
        for i in range(self.seq_len):
            if self.seq == 1:
                y = torch.cat([y, z, rhythm[:,i,:]], 1)
                # print("sec")
            else:
                y = torch.cat([y, z], 1)
                # print("no sec")

            hx = self.dirdecoder_0(y, hx)
            y = F.log_softmax(self.dirdecoder_out(hx), 1)
            ys.append(y)
            if self.training and emergencyEval == 0:
                p = torch.rand(1).item()
                if p < self.eps:
                    y = self.dx[:,i,:]
                    # print("tf")
                else:
                    y = self._findmax(y)
                    # print("ntf")
            else:
                y = self._findmax(y)
        return torch.stack(ys,1)

    def decoder(self, z, cpc, rhythm, dir, cond, chromaVector = None, emergencyEval = 0):
        y = torch.zeros((rhythm.size(0),self.input_dims)).to(self.device)
        # for y_-1, it is rest
        y[:, -1] = 1
        ys = []
        cors = []

        condVector = z
        # print(condVector.shape)
        if self.cond_dims > 0:

            condVector = torch.cat([condVector,cond[:,0,:]],1)
            # print(condVector.shape)

        if self.chromaLink == 1:
            # print("in chroma link")
            # print(chromaVector[0])
            chromaVector = F.sigmoid(chromaVector.view(-1,1,12))
            # print(chromaVector[0])

            condVector = torch.cat([condVector,chromaVector[:,0,:]],1)
            # print(condVector.shape)


        h0 = torch.tanh(self.decoder_hidden_init(condVector))


        # if self.cond_dims > 0:
        #     if self.concat_cond == 0:
        #         h0 = torch.tanh(self.decoder_hidden_init(cond[:,0,:]))
        #     else:
        #         if self.chromaLink == 1:
        #             chromaVector = F.sigmoid(chromaVector.view(-1,1,12))
        #             h0 = torch.tanh(self.decoder_hidden_init(torch.cat((z, cond[:,0,:], chromaVector[:,0,:]),dim=1)))
        #         else:
        #             h0 = torch.tanh(self.decoder_hidden_init(torch.cat((z, cond[:,0,:]),dim=1)))
        # elif self.cond_dims == 0:
        #     if self.chromaLink == 1:
        #         h0 = torch.tanh(self.decoder_hidden_init(torch.cat((z, chromaVector[:,0,:]),dim=1)))
        #     else:
        #         h0 = torch.tanh(self.decoder_hidden_init(z))
        hx = [None, None]
        hx[0] = h0
        # print(f"{y.shape} {cpc.shape} {rhythm.shape} {dir.shape} {cond.shape}")
        for i in range(self.seq_len):
            # print(rhythm.size())
            # TODO concat one hot repre of conditioning also
            # if self.cond_dims > 0 :
            #     y = torch.cat([y, z, rhythm[:,i,:], dir[:,i,:], cond[:,0,:]], 1)
            # elif self.cond_dims == 0 :
            #     y = torch.cat([y, z, rhythm[:,i,:], dir[:,i,:] ], 1)
            y = torch.cat([y, rhythm[:,i,:], dir[:,i,:], condVector], 1)
            # print(y.shape)

            hx[0] = self.decoder_0(y, hx[0])
            if self.decoderLayers  == 2:
                if i == 0:
                    # next hidden state first input if the first output of last state
                    hx[1] = hx[0]
                hx[1] = self.decoder_1(hx[0],hx[1])
                y = F.log_softmax(self.decoder_out(hx[1]), 1)
            else:
                y = F.log_softmax(self.decoder_out(hx[0]), 1)

            cors.append(self._findmax(y))
            ys.append(y)
            if self.training and emergencyEval == 0:
                p = torch.rand(1).item()
                if p < self.eps:
                    y = self.x[:,i,:]
                else:
                    y = self._findmax(y)
                # update the eps after one batch
                # print(self.eps)
                # eps can be less than minTf

                self.eps = max(self.decay / (self.decay + torch.exp(self.iteration / self.decay)), self.minTf)
                self.iteration += 1
                # print(self.eps)
            else:
                y = self._findmax(y)
        return torch.stack(ys, 1), torch.stack(cors, 1)  

    def forward(self, x, listOfContexts, listOfContextsCPC, pitchCurve, onsetDensity,
                    targetRhythmOH, targetDirOH, targetCpcOH, condOH, onsetOffsetOH, listOfConds,
                    randomBatch = False, sample=True):
#         print("vae forward", self.training)
        if randomBatch is False:
            
            if self.embeddings == 1:
                listOfEmbeddedContexts = []
                for ii in range(len(listOfContexts)):
                    midisInds = listOfContexts[ii].view(-1, 24, listOfContexts[ii].size(-1)).max(-1)[1]
                    cpcsInds = listOfContextsCPC[ii].view(-1, 24, listOfContextsCPC[ii].size(-1)).max(-1)[1]
                    # print(midisInds.max())
                    # print(cpcsInds.max()    )
                    contextEmbMidi = self.midiEmbedding(midisInds)
                    contextEmbCpc = self.cpcEmbedding(cpcsInds)
                    listOfEmbeddedContexts.append(contextEmbCpc + contextEmbMidi)
            listOfContexts = listOfEmbeddedContexts
            if self.training:
                self.x = x
                # TODO check if rx is the same as targetRhythmOH 
                # self.rx = x[:,:,:-2].sum(-1).unsqueeze(-1)
                # self.rx = torch.cat((self.rx,x[:,:,-2:]),-1)
                self.rx = targetRhythmOH
                self.iteration += 1
                self.dx = targetDirOH
                self.cx = targetCpcOH
            if self.cond_dims > 0 :
                # print(x.shape)
                # print(condOH.repeat(1,24,1).shape)
                # x = torch.cat((x,condOH.repeat(1,24,1)),-1)

                # for i in range(len(listOfContexts)):
                #     listOfContexts[i] = torch.cat((listOfContexts[i],listOfConds[i].repeat(1,24,1)),-1)
                pass
                # print(x.shape)
            
            # dis1, dis2, dis3 = self.encoder(x)
            dis1 = self.contextEncoder(listOfContexts)
            disCurve = self.runCurveEncoder(pitchCurve, onsetDensity)
            dis2Curve = Normal(disCurve.mean[:,:85], disCurve.stddev[:,:85])
            dis3Curve = Normal(disCurve.mean[:,85:], disCurve.stddev[:,85:])
            
            if sample is True:
                z1 = dis1.rsample()
                z2 = dis2Curve.rsample()
                z3 = dis3Curve.rsample()
            else:
                z1 = dis1.mean
                z2 = dis2Curve.mean
                z3 = dis3Curve.mean
            z2 = z2.view(z2.shape[0], -1)#.detach()
            z3 = z3.view(z3.shape[0], -1)#.detach()

            recon_rhythm = self.rhythm_decoder(z2, z1, onsetOffsetOH)
            chromaVector = self.chromaLayer(z1)
            recon_cpc = torch.tensor(0)
            if self.detaching == 1: # or randomBatch is True:
                recon_dir = self.dir_decoder(z3, recon_rhythm.detach())
                recon_x, _ = self.decoder(z1, recon_cpc.detach(), recon_rhythm.detach(), recon_dir.detach(), condOH, chromaVector)
            else:
                recon_dir = self.dir_decoder(z3, recon_rhythm)
                recon_x, _ = self.decoder(z1, recon_cpc, recon_rhythm, recon_dir, condOH, chromaVector)
            # else:
            #     recon_dir = self.dir_decoder(z3)
            
            output = (recon_x, recon_cpc, recon_rhythm, recon_dir, dis1, dis2Curve, dis3Curve, chromaVector)
        else:
            if self.training:
                self.dx = targetDirOH
            disCurve = self.runCurveEncoder(pitchCurve, onsetDensity)
            dis3Curve = Normal(disCurve.mean[:,85:], disCurve.stddev[:,85:])
            z3 = dis3Curve.rsample()
            z3 = z3.view(z3.shape[0], -1)
            recon_dir = self.dir_decoder(z3, None)
            output = (recon_dir, dis3Curve)
        return output