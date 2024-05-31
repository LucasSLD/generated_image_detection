import torch.nn as nn
import sys
sys.path.append("../tools")
from constants import CLIP_FEATURE_DIM


class MultiClassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CLIP_FEATURE_DIM,512)
        self.fc2 = nn.Linear(512,18)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act((self.fc1(x)))
        logits = self.fc2(x)
        return logits     