import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

""" Implementation of the self-supervised model for 
    audio-video correlation, proposed in the Objects that Sound 
    paper.
"""


def getNonLinear(nlu):

    if nlu == "relu":
       return nn.ReLU(inplace=True)
    elif nlu == "glu": #gated linear unit
       return F.glu
    else:
       return None
   
def getNorm(norm):

    if norm =="bn": #default batchnorm
       return nn.BatchNorm2d
    else:
       return None
   
def Conv3x3(in_ch, out_ch, stride):
    """ 3x3 convolution with padding """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, 
                     padding=1)
    
    return conv

def Conv1x1(in_ch, out_ch, stride=1):
    """ 1x1 convolution """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False)
    return conv

class BasicBlock(nn.Module):
  """ Basic conv block"""
  
  def __init__(self, in_ch, out_ch, stride, ps, norm):
    
      super(BasicBlock, self).__init__()

      norm_fn = getNorm(norm)  

      self.conv1 = Conv3x3(in_ch, out_ch, stride)
      self.bn1 = norm_fn(out_ch)
        
      self.conv2 = Conv3x3(out_ch, out_ch, stride=1)
      self.bn2 = norm_fn(out_ch)

      self.relu = getNonLinear("relu")

  def forward(self, x):
    
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      return out

class AudNet(nn.Module):
  """ Audio subnet """

  def __init__(self, norm, init_wts=True):

      super(AudNet, self).__init__()

      out_ch = [64, 128, 256, 512]
      #for spectrogram features as mentioned in Objects that Sound paper
      self.c1 = BasicBlock(in_ch=1, out_ch=out_ch[0], stride=2, ps=2) 
      self.c2 = BasicBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, ps=2, norm=norm)
      self.c3 = BasicBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, ps=2, norm=norm)
      self.c4 = BasicBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, ps=2, norm=norm)
      self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool_final = nn.MaxPool2d(kernel_size=(16, 12), stride=(16, 12))


      fc_dim = 128
      self.norm = F.normalize
      self.embed = nn.Sequential(
                nn.Linear(512, fc_dim),
                getNonLinear("relu"),
                nn.Linear(fc_dim, fc_dim)
                )
      
      
  def forward(self, x):
     
      #conv features
      out = self.c1(x)
      #print("c1 features", out.shape)
      out = self.pool1(out)
      #print("p1 features", out.shape)
      out = self.c2(out)
      #print("c2 features", out.shape)
      out = self.pool(out)
      #print("p2 features", out.shape)
      out = self.c3(out)
      #print("c3 features", out.shape)
      out = self.pool(out)
      #print("p3 features", out.shape)
      out = self.c4(out)
      #print("c4 features", out.shape)
      out = self.pool_final(out)
      #print("aud features:", out.shape)
      
      out = out.view(out.shape[0], -1)
      #print("flatten:", out.shape)

      #conv embeddings
      out = self.embed(out)
      #print("embed:", out.shape)

      #normalize embeddings to length=1
      out = self.norm(out, p=2, dim=1)
      #print("norm:", out.shape)

      return out

     

class VisNet(nn.Module):
  """ Visual subnet """

  def __init__(self, norm, init_wts=True):

      super(VisNet, self).__init__()

      out_ch = [64, 128, 256, 512]
      self.c1 = BasicBlock(in_ch=3, out_ch=out_ch[0], stride=2, ps=2, norm=norm)
      self.c2 = BasicBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, ps=2, norm=norm )
      self.c3 = BasicBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, ps=2, norm=norm)
      self.c4 = BasicBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, ps=2, norm=norm)

      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool_final = nn.MaxPool2d(kernel_size=14, stride=14)


      fc_dim = 128
      self.norm = F.normalize
      self.embed = nn.Sequential(
                nn.Linear(512, fc_dim),
                getNonLinear("relu"),
                nn.Linear(fc_dim, fc_dim)
                )
      
      
  def forward(self, x):
      
      #conv features
      out = self.c1(x)
      #print("c1 features", out.shape)
      out = self.pool(out)
      #print("p1 features", out.shape)
      out = self.c2(out)
      #print("c2 features", out.shape)
      out = self.pool(out)
      #print("p2 features", out.shape)
      out = self.c3(out)
      #print("c3 features", out.shape)
      out = self.pool(out)
      #print("p3 features", out.shape)
      out = self.c4(out)
      #print("c4 features", out.shape)
      out = self.pool_final(out)
      #print("vis features:", out.shape)
      
      ##out = self.features(x)
     
      out = out.view(out.shape[0], -1)
      #print("flatten:", out.shape)

      out = self.embed(out)
      #print("embed:", out.shape)
      
      #normalize embeddings to length=1
      out = self.norm(out, p=2, dim=1)
      #print("norm:", out.shape)

      return out



class AVENet(nn.Module):
  def __init__(self, n_cls, norm, init_wts=True):

      super(AVENet, self).__init__()

      self.num_class = n_cls
      
      self.vfeatures = VisNet(norm=norm) 
      self.afeatures = AudNet(norm=norm) 

      #0/1 binary classifier.  1 if audio is related to the visual, else 0.
      self.av_classifier = nn.Linear(1, n_cls)          
      
      #self.output = nn.Softmax(dim=1)
      #self.output = nn.Sigmoid()
      
  def forward(self, x_v, x_a):
      #print("==========")
      v_out = self.vfeatures(x_v)
      #print("v_out:", v_out.shape)
 
      #print("==========")
      a_out = self.afeatures(x_a)
      #print("a_out:", a_out.shape)

      #euclidean distance between embeddings
      av_dist = F.mse_loss(v_out, a_out, reduction='none').mean(1)
      #print("av_dist:", av_dist.shape)
      out = av_dist.view(av_dist.shape[0], 1)
      #print("extend:", out.shape)

      out = self.av_classifier(out)
      #print("av_classifier:", out.shape)
      
      return out, av_dist, v_out, a_out


  def getImgEmb(self, x_v):
      """Get visual embeddings for a given image"""
      
      v_emb = self.vfeatures(x_v)
      
      return v_emb
  
  def getAudEmb(self, x_a):
      """Get audio embeddings for a given audio sample"""
      
      a_emb = self.afeatures(x_a)
      
      return a_emb 

  def getEmbCor(self, x_1, x_2):
      """Get correlation between input embeddings """
    
      print(x_1.shape, x_2.shape) 
      #euclidean distance between embeddings
      dist = F.mse_loss(x_1, x_2, reduction='none').mean(1)
      #print("dist:", dist.shape)
      out = dist.view(dist.shape[0], 1)
      #print("extend:", out.shape)
      out = self.av_classifier(out)
      out = F.softmax(out, dim=1)
      
      return dist, out
