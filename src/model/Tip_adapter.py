import torch
import torch.nn as nn
import sys
sys.path.append("../tools")
sys.path.append(".")
from constants import CLIP_FEATURE_DIM, FAKE_LABEL, REAL_LABEL
import open_clip
from open_clip import tokenizer
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class TipAdapter(nn.Module): 
    """ 
    See 'Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification'
    """
    def __init__(self, 
                 img_per_class: int,
                 data: Dataset, 
                 device: str="cpu", 
                 alpha: float = 5,
                 beta: float = 5.5):
        super().__init__()

        self.alpha = alpha # weight of the few-shot knwoledge
        self.beta  = beta # exp sharpness in phi function

        features_fake = data.features[data.label == FAKE_LABEL]
        features_real = data.features[data.label == REAL_LABEL]
        
        self.F_train = torch.cat((features_fake[:img_per_class],features_real[:img_per_class]))
        self.F_train = self.F_train / torch.linalg.vector_norm(self.F_train,dim=1,keepdim=True)
        self.F_train = self.F_train.to(device)
        # print(torch.linalg.vector_norm(self.F_train,dim=-1,keepdim=True))
        self.L_train = [torch.Tensor([1, 0]) for i in range(img_per_class)] # fake label one hot encoding
        self.L_train += [torch.Tensor([0, 1]) for i in range(img_per_class)] # real label encoding
        self.L_train = torch.stack(self.L_train).type(torch.LongTensor).to(device)

        model,_ , preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
        model.eval()

        tokens =  tokenizer.tokenize(["a" + label + "image" for label in ("generated","real")])
        with torch.no_grad():
            self.Wc = model.encode_text(tokens)
        self.Wc /= torch.linalg.vector_norm(self.Wc,dim=-1,keepdim=True)

    def __len__(self):
        return len(self.L_train)
    
    def phi(self, x):
        return torch.exp(-self.beta*(1-x))

    def forward(self, x):
        # normalization
        x /= torch.linalg.vector_norm(x,dim=-1,keepdim=True)
        return self.alpha * self.phi(x@self.F_train.T)@self.L_train.float() + x@self.Wc.T
    
    def predict(self, x):
        return torch.argmax(self.forward(x),dim=-1)

    def get_accuracy(self,features,true_labels,device: str):
        self.eval()
        with torch.no_grad():
            pred = self.predict(features.to(device))
        return torch.mean(torch.eq(pred,true_labels.to(device)).float()).item()



        
