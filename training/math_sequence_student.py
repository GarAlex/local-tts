"""Whole-utterance latent student training architecture; not a native voice exporter.

Supervise token durations and full-utterance teacher decoder inputs. Keep the
pretrained waveform decoder frozen; never train a new vocoder from scratch.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SequenceStudent(nn.Module):
    def __init__(self, vocabulary, hidden=96, channels=192):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary,hidden)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.0,batch_first=True)
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.duration=nn.Linear(hidden,1)
        self.frame_input=nn.Conv1d(hidden+2,hidden,1)
        self.refine=nn.ModuleList([nn.Conv1d(hidden,hidden,5,padding=2*d,dilation=d) for d in (1,2,4)])
        self.output=nn.Conv1d(hidden,channels,1)

    def _encode(self,tokens,lengths):
        if tokens.ndim!=2 or tokens.shape[0]==0 or not 0<tokens.shape[1]<=1024 or lengths.shape!=(tokens.shape[0],) or tokens.dtype!=torch.long or lengths.dtype!=torch.long:
            raise ValueError('invalid token/length contract')
        if not ((lengths>0)&(lengths<=tokens.shape[1])).all():raise ValueError('invalid token lengths')
        token_mask=torch.arange(tokens.shape[1],device=tokens.device)[None,:]<lengths[:,None]
        embedded=self.embedding(tokens)
        position=torch.arange(tokens.shape[1],device=tokens.device,dtype=embedded.dtype)[:,None]
        frequency=torch.exp(torch.arange(0,embedded.shape[-1],2,device=tokens.device,dtype=embedded.dtype)*(-torch.log(torch.tensor(10000.0,device=tokens.device))/embedded.shape[-1]))
        phase=position*frequency[None,:]
        positional=torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(1)
        hidden=self.context(embedded+positional[None,:,:],src_key_padding_mask=~token_mask)
        log_duration=self.duration(hidden).squeeze(-1).masked_fill(~token_mask,0)
        return hidden,log_duration,token_mask

    @torch.no_grad()
    def infer(self,tokens,lengths,duration_scale=1.0):
        if self.training:raise ValueError('inference requires eval mode')
        if not math.isfinite(duration_scale) or not 0.5<=duration_scale<=2.0:raise ValueError('duration scale must be 0.5..2')
        _,log_duration,token_mask=self._encode(tokens,lengths)
        if not torch.isfinite(log_duration).all():raise ValueError('nonfinite predicted duration')
        durations=(torch.expm1(log_duration.clamp(0,math.log1p(128)))*duration_scale).round().long().clamp(1,256)
        durations=durations.masked_fill(~token_mask,0)
        if (durations.sum(1)>7740).any():raise ValueError('predicted speech exceeds frame limit')
        latent,_,mask=self.forward(tokens,lengths,durations)
        if not torch.isfinite(latent).all():raise ValueError('nonfinite predicted latent')
        return latent,durations,mask

    def forward(self,tokens,lengths,durations):
        if tokens.ndim!=2 or durations.shape!=tokens.shape or lengths.shape!=(tokens.shape[0],):
            raise ValueError('invalid token/duration/length shapes')
        if tokens.dtype!=torch.long or durations.dtype!=torch.long or lengths.dtype!=torch.long:
            raise ValueError('tokens, lengths and durations must be int64')
        if not ((lengths>0)&(lengths<=tokens.shape[1])).all() or (durations<0).any():
            raise ValueError('invalid lengths or negative durations')
        token_mask=torch.arange(tokens.shape[1],device=tokens.device)[None,:]<lengths[:,None]
        if (durations.masked_select(~token_mask)!=0).any():
            raise ValueError('padding tokens cannot have acoustic duration')
        totals=durations.sum(1)
        if not ((totals>0)&(totals<=7740)).all():
            raise ValueError('expected 1..7740 acoustic frames per utterance')
        hidden,log_duration,token_mask=self._encode(tokens,lengths)
        positions=torch.arange(int(totals.max()),device=tokens.device)[None,:].expand(tokens.shape[0],-1)
        boundaries=durations.cumsum(1)
        index=torch.searchsorted(boundaries.contiguous(),positions.contiguous(),right=True).clamp(max=tokens.shape[1]-1)
        expanded=hidden.gather(1,index[:,:,None].expand(-1,-1,hidden.shape[-1]))
        starts=F.pad(boundaries[:,:-1],(1,0))
        within=positions-starts.gather(1,index)
        fraction=within.to(hidden.dtype)/(durations.gather(1,index)-1).clamp_min(1)
        phase=torch.stack((fraction,torch.sin(fraction*torch.pi)),dim=1)
        mask=(positions<totals[:,None])[:,None,:]
        frames=F.gelu(self.frame_input(torch.cat((expanded.transpose(1,2),phase),1)))*mask
        for layer in self.refine:
            frames=(frames+F.gelu(layer(frames)))*mask
        return self.output(frames)*mask,log_duration,mask


def supervised_loss(predicted,target,log_duration,durations,token_mask,frame_mask):
    """Masked teacher-latent and log-duration losses; padding never trains speech."""
    if predicted.shape!=target.shape or frame_mask.shape!=(target.shape[0],1,target.shape[2]):
        raise ValueError('teacher latent shape differs from student/alignment')
    if token_mask.shape!=durations.shape or log_duration.shape!=durations.shape:
        raise ValueError('teacher duration shape differs')
    if not frame_mask.any() or not token_mask.any():
        raise ValueError('empty supervision masks')
    latent=(predicted-target).abs().masked_select(frame_mask.expand_as(predicted)).mean()
    duration=F.mse_loss(log_duration.masked_select(token_mask),torch.log1p(durations.float()).masked_select(token_mask))
    return latent,duration
