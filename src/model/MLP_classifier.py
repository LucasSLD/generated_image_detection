import torch
import torch.nn as nn
import sys
sys.path.append("../tools")
sys.path.append(".")
from constants import CLIP_FEATURE_DIM, FAKE_LABEL, REAL_LABEL
from dataset import DeepFakeDataset

class MultiClassClassifier(nn.Module):
    def __init__(self,n_features: int=CLIP_FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(n_features,512)
        self.fc2 = nn.Linear(512,18)
        self.act = nn.ReLU()
        self.features_min = None
        self.features_max = None
        self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 
        self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL}
        self.int_to_gen = {}
        self.gen_to_int = {}
    
    def forward(self, x, normalize: bool = False):
        if normalize:
            x = (x - self.features_min)/(self.features_max - self.features_min)
        x = self.act((self.fc1(x)))
        logits = self.fc2(x)
        return logits
        
    def set_normalization_params(self,features_min,features_max):
        """Save normalization parameters from training set

        Args:
            features_min (Tensor): minimum of each feature
            features_max (Tensor): maximum of each feature 
        """
        self.features_min = features_min
        self.features_max = features_max

    def set_generators_maps(self, gen_to_int: dict, int_to_gen: dict):
        self.gen_to_int = gen_to_int
        self.int_to_gen = int_to_gen

    def class_to_label(self, classes):
        """Maps an array of integers representing classes toa list of 0 and 1 depending on whether the class represented a generated or a real image.

        Args:
            classes (_type_): array of classes taking values in DeepFakeDataset.int_to_gen.keys()

        Returns:
            _type_: array of 0 and 1 representing fake and real labels 
        """
        if not self.int_to_gen:
            raise Exception("set_generators_map hasn't been called. Can't map classes to binary labels.")
        if isinstance(classes,torch.Tensor):
            classes = classes.cpu().numpy()
        labels = torch.zeros(len(classes))
        for i, e in enumerate(classes):
            labels[i] = FAKE_LABEL if self.int_to_gen[e] != "null" else REAL_LABEL
        return labels

    def predict_classes(self, features, device: str):
        """Predict the generators associated to the given features. See MultiClassClassifier.int_to_gen to see the mapping between integer values and generators' name.

        Args:
            features (_type_): (n_rows,n_features) tensor
            device (str): device on which the model is located

        Returns:
            _type_: (n_rows) tensor
        """
        return torch.argmax(self.forward(features.to(device)),dim=1)
    
    def predict_binary(self, features, device: str):
        """Associate features to binary labels. See MultiClassClassifier.int_to_label to see the mapping.

        Args:
            features (Tensor): Tensor of shape (n_rows,n_features).
            device (str): device on which the model is located.

        Returns:
            Tensor: tensor of shape (n_rows)
        """
        return self.class_to_label(self.predict_classes(features,device)).to(device)

    def get_model_accuracy_binary(self,
                                  features, 
                                  true_labels,
                                  device):
        pred = self.predict_binary(features,device)
        return torch.mean(torch.eq(pred,true_labels.to(device)).float()).item()
    
    def get_model_accuracy_multiclass(self, 
                                      features, 
                                      true_classes,
                                      device): 
        pred = self.predict_classes(features,device)
        return torch.mean(torch.eq(pred,true_classes.to(device)).float()).item()