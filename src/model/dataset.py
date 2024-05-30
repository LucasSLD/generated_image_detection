import albumentations as A
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("../tools")
from constants import CLIP_FEATURE_DIM, REAL_LABEL, FAKE_LABEL
import open_clip
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

transform_torch = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
	A.OneOf([
    A.ImageCompression(quality_lower=40,quality_upper=40,p=1/3),
    A.ImageCompression(quality_lower=65,quality_upper=65,p=1/3),
	A.ImageCompression(quality_lower=90,quality_upper=90,p=1/3),
    ], p=3/4)
    ], p=1.0)

device = "cuda" if torch.cuda.is_available() else "cpu"

REAL_IMG_GEN = "null"
REAL_FOLDER_NAME = "Flickr2048"
CUDA_MEMORY_LIMIT = 1000

class DeepFakeDataset(Dataset):
    def __init__(self, 
                 path_to_data: str,
                 img_per_gen: int,
                 balance_real_fake: bool,
                 device: str = device):
        if not path_to_data.endswith("/"): path_to_data += "/"
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
        model.eval()

        generators = [gen for gen in os.listdir(path_to_data) if os.listdir(path_to_data + gen)[0].endswith(".png")] # fake images are .png images
        self.img_per_gen = img_per_gen
        self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))
        self.features = torch.empty((0,CLIP_FEATURE_DIM))
        self.label = []
        self.gen = []
        self.int_to_gen = {}
        self.gen_to_int = {}
        self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 
        self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL} 
        self.n_fake = len(generators) * img_per_gen
        self.n_real = self.n_fake if balance_real_fake else img_per_gen
        
        #================= Real images processing =================================================================================#
        k = 0
        self.int_to_gen[k] = REAL_IMG_GEN
        self.gen_to_int[REAL_IMG_GEN] = k
        jpg_files = os.listdir(path_to_data + "Flickr2048")
        imgs = []
        
        for i, file in enumerate(tqdm(jpg_files,total=self.n_real,desc="Processing images from Flickr2048")):
            if i >= self.n_real:
                break
            imgs.append(self.transform(Image.open(path_to_data + REAL_FOLDER_NAME + "/" + file)).unsqueeze(0).to(device))
            self.label.append(REAL_LABEL)
            self.gen.append(k)
            
            if len(imgs) == CUDA_MEMORY_LIMIT: # avoiding CUDA OutOfMeMory Error
                with torch.no_grad():
                    features = model.encode_image(torch.cat(imgs,dim=0))
                self.features = torch.cat((self.features,features.cpu()),dim=0)
                imgs = []
                torch.cuda.empty_cache()
        
        if imgs: # not empty
            with torch.no_grad(): # extracting the features from the last images
                features = model.encode_image(torch.cat(imgs,dim=0))
            self.features = torch.cat((self.features,features.cpu()),dim=0)
        torch.cuda.empty_cache()
        
        #================= Fake images processing =================================================================================#
        for gen in generators: # fake images
            k += 1
            self.int_to_gen[k] = gen
            self.gen_to_int[gen] = k
            png_files = [file for file in os.listdir(path_to_data + gen) if file.endswith(".png")]
            imgs = []
        
            for i, file in enumerate(tqdm(png_files,f"Processing images from {gen}",total=min(len(png_files),img_per_gen))):
                if i >= img_per_gen:
                    break
                imgs.append(self.transform(Image.open(path_to_data + gen + "/" + file)).unsqueeze(0).to(device))
                self.label.append(FAKE_LABEL)
                self.gen.append(k)
            
            with torch.no_grad():
                features = model.encode_image(torch.cat(imgs,dim=0))
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {"label" : self.label[index], 
                "features" : self.features[index],
                "generator" : self.gen[index]}
    
    def class_to_label(self, classes):
        """Maps an array of integers representing classes toa list of 0 and 1 depending on whether the class represented a generated or a real image.

        Args:
            classes (_type_): array of classes taking values in DeepFakeDataset.int_to_gen.keys()

        Returns:
            _type_: array of 0 and 1 representing fake and real labels 
        """
        if isinstance(classes,torch.Tensor):
            classes = classes.cpu().numpy()
        labels = torch.zeros(len(classes))
        for i, e in enumerate(classes):
            labels[i] = FAKE_LABEL if self.int_to_gen[e] != "null" else REAL_LABEL
        return labels
    
    def save(self,output_path: str):
        torch.save({
            "features": self.features,
            "label": self.label,
            "gen": self.gen,
            "int_to_gen": self.int_to_gen,
            "gen_to_int": self.gen_to_int,
            "img_per_gen": self.img_per_gen,
            "n_real": self.n_real,
            "n_fake": self.n_fake
        },output_path)
    
class DeepFakeDatasetFastLoad(Dataset):
    """Class used to load DeepFakeDataset that has been serialized"""
    def __init__(self,path: str) -> None:
        data = torch.load(path,"cpu")
        self.img_per_gen  = data["img_per_gen"]
        self.features     = data["features"]
        self.gen          = data["gen"]
        self.label        = data["label"]
        self.gen_to_int   = data["gen_to_int"]
        self.int_to_gen   = data["int_to_gen"]
        self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 
        self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL}
        self.n_real       = data["n_real"]
        self.n_fake       = data["n_fake"]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"features": self.features[index],
                "label": self.label[index],
                "generator": self.gen[index]}
    
    def class_to_label(self, classes):
        """Maps an array of integers representing classes toa list of 0 and 1 depending on whether the class represented a generated or a real image.

        Args:
            classes (_type_): array of classes taking values in DeepFakeDataset.int_to_gen.keys()

        Returns:
            _type_: array of 0 and 1 representing fake and real labels 
        """
        if isinstance(classes,torch.Tensor):
            classes = classes.cpu().numpy()
        labels = torch.zeros(len(classes))
        for i, e in enumerate(classes):
            labels[i] = FAKE_LABEL if self.int_to_gen[e] != "null" else REAL_LABEL
        return labels