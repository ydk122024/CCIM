from turtle import forward
from numpy import rint
from sklearn import datasets
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
from torch.nn import Parameter

class CCIM(nn.Module):
  def __init__(self, num_joint_feature, num_gz, strategy):
    super(CCIM, self).__init__()
    self.num_joint_feature = num_joint_feature
    self.num_gz = num_gz
    if strategy == 'dp_cause':
      self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature )
    elif strategy == 'ad_cause':
      self.causal_intervention = additive_intervention(num_gz, num_joint_feature )
    else:
      raise ValueError("Do Not Exist This Strategy.")

    self.w_h = Parameter(torch.Tensor(self.num_joint_feature, 128)) 
    self.w_g = Parameter(torch.Tensor(self.num_gz, 128)) 
    self.classifier = classifier()
    self.emotic_fc = nn.Linear(128, 26)
    self.caers_fc = nn.Linear(128, 7)
    self.groupwalk_fc = nn.Linear(128, 4)
 


    
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_normal_(self.w_h)
    nn.init.xavier_normal_(self.w_g)

  def forward(self, joint_feature, confounder_dictionary, prior, dataset):   
    g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)
    proj_h = torch.matmul(joint_feature, self.w_h)  
    proj_g_z = torch.matmul(g_z, self.w_g) 
    do_x = proj_h + proj_g_z
    out = self.classifier(do_x)
   


    if dataset == 'EMOTIC':
      fin = self.emotic_fc(out)
    
    elif dataset == 'CAER_S':
      fin = self.caers_fc(out)
    
    elif dataset == 'GroupWalk':
      fin = self.groupwalk_fc(out)

    else:
      raise ValueError("Do Not Exist This Dataset.")


    return fin

class classifier(nn.Module):
  def __init__(self):
    super(classifier,self).__init__()
    self.fc1 = nn.Linear(128, 128*4)
    self.fc2 = nn.Linear(128*4, 128)
    self.drop = nn.Dropout(p=0.5)
    self.norm = nn.BatchNorm1d(128)
    


  def forward(self, out):
    residual = out
    out = self.norm(out)
    out = gelu(self.fc1(out))
    out = self.drop(out)
    out = self.fc2(out)
    out = self.drop(out)
    out = residual + out*0.3

    return out

def gelu(x):
      return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class dot_product_intervention(nn.Module):
  def __init__(self, con_size, fuse_size):
    super(dot_product_intervention, self).__init__()
    self.con_size = con_size 
    self.fuse_size = fuse_size  
    self.query = nn.Linear(self.fuse_size, 256, bias= False) 
    self.key = nn.Linear(self.con_size, 256, bias= False) 


  def forward(self, confounder_set, fuse_rep, probabilities):
    query = self.query(fuse_rep) 
    key = self.key(confounder_set)  
    mid = torch.matmul(query, key.transpose(0,1)) / math.sqrt(self.con_size)  
    attention = F.softmax(mid, dim=-1) 
    attention = attention.unsqueeze(2)  
    fin = (attention*confounder_set*probabilities).sum(1)  
    
    return fin


class additive_intervention(nn.Module):
  def __init__(self, con_size, fuse_size):
    super(additive_intervention,self).__init__()
    self.con_size = con_size
    self.fuse_size = fuse_size
    self.Tan = nn.Tanh()
    self.query = nn.Linear(self.fuse_size, 256, bias = False)
    self.key = nn.Linear(self.con_size, 256, bias = False)
    self.w_t = nn.Linear(256, 1, bias=False)

  def forward(self, confounder_set, fuse_rep, probabilities):
 
    query = self.query(fuse_rep) 
  
    key =  self.key(confounder_set)  
  
    query_expand = query.unsqueeze(1)  
    fuse = query_expand + key 
    fuse = self.Tan(fuse)
    attention = self.w_t(fuse) 
    attention = F.softmax(attention, dim=1)
    fin = (attention*confounder_set*probabilities).sum(1)  

    return fin

if __name__ == '__main__':
  ''' An example showing how CCIM is used '''
  joint_feature = torch.randn(64, 256)
  confounder = torch.randn(1024, 2048)
  probabilities = torch.rand(1024, 1)
  ccim = CCIM(256, 2048, strategy = 'dp_cause')  #options: ad_cause, dp_cause
  out = ccim(joint_feature, confounder, probabilities, dataset = 'CAER_S')  # options: EMOTIC, CAER-S, GroupWalk
  print(out.shape)

  
