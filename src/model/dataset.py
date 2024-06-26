import albumentations as A
from torch.utils.data import Dataset
import sys
sys.path.append("../tools")
sys.path.append("../utils")
from constants import *
from utils import extract_color_features
import open_clip
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoImageProcessor, AutoModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
from copy import deepcopy
import pandas as pd

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

transform_40 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ImageCompression(quality_lower=40,quality_upper=40,p=1.0)]) 

transform_65 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ImageCompression(quality_lower=65,quality_upper=65,p=1.0)]) 

transform_90 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ImageCompression(quality_lower=90,quality_upper=90,p=1.0)]) 

transform_100 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0)]) 

device = "cuda" if torch.cuda.is_available() else "cpu"

REAL_FOLDER_NAME = "Flickr2048"
CUDA_MEMORY_LIMIT = 1000

def int2gen(i: int) -> str:
    """Maps intgers to generators

    Args:
        i (int): a key from constants.INT_TO_GEN

    Returns:
        str: the name of the generator assiciated with the given key
    """
    return INT_TO_GEN[i]

def class2label(i: int) -> int:
    """maps class to real or fake label

    Args:
        i (int): A key from constants.INT_TO_GEN

    Returns:
        int: an integer representing either "real" or "fake" image (see constants.REAL_LABEL and constants.FAKE_LABEL)
    """
    return REAL_LABEL if INT_TO_GEN[i] == REAL_IMG_GEN else FAKE_LABEL

def generate(path_to_img: str, 
             model: BlipForConditionalGeneration=BlipForConditionalGeneration.from_pretrained("unography/blip-long-cap",cache_dir="/data4/saland/cache"), 
             processor: BlipProcessor=BlipProcessor.from_pretrained("unography/blip-long-cap"),
             diffusion_pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"),
             num_inference_steps: int=100,
             device: str=device):
    """Extracts the caption from a given image and use this caption as a prompt to generate an image. 

    Args:
        path_to_img (_type_): _description_

    Returns:
        _type_: a PIL image
    """

    model.to(device)
    diffusion_pipeline.to(device)

    raw_image = Image.open(path_to_img).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    out = model.generate(pixel_values=pixel_values, max_length=1000, num_beams=3, repetition_penalty=2.5)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    # print(prompt)

    seed = torch.Generator(device='cuda').manual_seed(2)
    compel = Compel(tokenizer=[diffusion_pipeline.tokenizer, diffusion_pipeline.tokenizer_2] , text_encoder=[diffusion_pipeline.text_encoder, diffusion_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True]) 
    conditioning, pooled = compel("a camera highly detailed shot of "+prompt)
    images = diffusion_pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled,num_inference_steps=num_inference_steps).images[0]


    inputs2 = processor(images, return_tensors="pt").to(device)
    pixel_values2 = inputs2.pixel_values
    out2 = model.generate(pixel_values=pixel_values2, max_length=1000, num_beams=3, repetition_penalty=2.5)
    prompt2 = processor.decode(out2[0], skip_special_tokens=True)
    # print(prompt2)

    return images

def shuffle_data(dataset: Dataset,
                 in_place: bool = False, 
                 seed: int=SEED):
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset),generator=rng)
    if not in_place:
        dataset = deepcopy(dataset)
    for key in dataset.__dict__:
        if type(dataset.__getattribute__(key)) == torch.Tensor:
            print(f"shuffling {key}")
            dataset.__setattr__(key,dataset.__getattribute__(key)[permutation])
    return dataset

def select_slice(data: Dataset,
                 n_elements: int, 
                 attributes_to_silce=("label","features","gen")):
    data = deepcopy(data)
    for key in data.__dict__:
        if key in attributes_to_silce:
            data.__setattr__(key,data.__getattribute__(key)[:n_elements])
    return data
            

class DeepFakeDataset(Dataset): # data3/AID
    def __init__(self, 
                 path_to_data: str,
                 img_per_gen: int=100,
                 balance_real_fake: bool=True,
                 device: str = device,
                 load_from_disk = False,
                 feature_type: str = CLIP):

        assert feature_type in (CLIP,DINO)
        
        if load_from_disk:
            data = torch.load(path_to_data)
            self.features = data["features"]
            self.label    = data["label"]
            self.gen      = data["gen"]
            self.gen_original_name = data["gen_original_name"]
            self.name     = data["name"]
            self.int_to_gen = INT_TO_GEN
            self.gen_to_int = GEN_TO_INT
        else:
            if not path_to_data.endswith("/"): path_to_data += "/"
            if feature_type == CLIP:
                model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
                self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))
            elif feature_type == DINO:
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                model = AutoModel.from_pretrained('facebook/dinov2-base')
                model.to(device)
                model.eval()
                        
            model.eval()

            generators = [gen for gen in os.listdir(path_to_data) if os.listdir(path_to_data + gen)[0].endswith(".png")] # fake images are .png images
            self.img_per_gen = img_per_gen
            self.features = torch.empty((0,CLIP_FEATURE_DIM))
            self.label = []
            self.gen = []
            self.gen_original_name = [] # for sanity check mapping generators <-> families of generators
            self.name = [] # name of the image
            self.int_to_gen = INT_TO_GEN
            self.gen_to_int = GEN_TO_INT
            self.n_fake = len(generators) * img_per_gen
            self.n_real = self.n_fake if balance_real_fake else img_per_gen
            
            #================= Real images processing =================================================================================#
            jpg_files = os.listdir(path_to_data + "Flickr2048")
            imgs = []

            def extract_features(imgs: list, feature_type: str):
                with torch.no_grad():
                    if feature_type == CLIP:
                        features = model.encode_image(torch.cat(imgs,dim=0))
                    elif feature_type == DINO:
                        inputs = processor(images=imgs,return_tensors="pt")
                        features = model(inputs["pixel_values"].to(device))[1]
                return features
                

            for i, file in enumerate(tqdm(jpg_files,total=self.n_real,desc="Processing images from Flickr2048")):
                if i >= self.n_real:
                    break
                img = Image.open(path_to_data + REAL_FOLDER_NAME + "/" + file)
                if feature_type == CLIP:
                    imgs.append(self.transform(img).unsqueeze(0).to(device))
                elif feature_type == DINO:
                    imgs.append(img)
                self.label.append(REAL_LABEL)
                self.gen.append(gen2int(REAL_IMG_GEN))
                self.gen_original_name.append(REAL_IMG_GEN)
                self.name.append(file)
                # CLIP features
                if len(imgs) == CUDA_MEMORY_LIMIT: # avoiding CUDA OutOfMemory Error
                    features = extract_features(imgs,feature_type)
                    self.features = torch.cat((self.features,features.cpu()),dim=0)
                    imgs = []
                    torch.cuda.empty_cache()
            
            if imgs: # not empty
                features = extract_features(imgs,feature_type) # extracting the features from the last images
                self.features = torch.cat((self.features,features.cpu()),dim=0)
            torch.cuda.empty_cache()
            
            #================= Fake images processing =================================================================================#
            for gen in generators:
                png_files = [file for file in os.listdir(path_to_data + gen) if file.endswith(".png")]
                imgs = []
            
                for i, file in enumerate(tqdm(png_files,f"Processing images from {gen}",total=min(len(png_files),img_per_gen))):
                    if i >= img_per_gen:
                        break
                    img = Image.open(path_to_data + gen + "/" + file)
                    if feature_type == CLIP:
                        imgs.append(self.transform(img).unsqueeze(0).to(device))
                    elif feature_type == DINO:
                        imgs.append(img)
                    self.label.append(FAKE_LABEL)
                    self.gen.append(gen2int(gen))
                    self.gen_original_name.append(gen)
                    self.name.append(file)
                    # CLIP features
                    if len(imgs) == CUDA_MEMORY_LIMIT: # avoiding CUDA OutOfMemory Error
                        features = extract_features(imgs,feature_type)
                        self.features = torch.cat((self.features,features.cpu()),dim=0)
                        imgs = []
                        torch.cuda.empty_cache()
                
                # CLIP features
                if imgs:    
                    features = extract_features(imgs,feature_type)
                    self.features = torch.cat((self.features,features.cpu()),dim=0)
                torch.cuda.empty_cache()

            self.label = torch.Tensor(self.label).type(torch.LongTensor) # pytorch's built-in loss function only works with LongTensor
            self.gen   = torch.Tensor(self.gen).type(torch.LongTensor)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {"label" : self.label[index], 
                "features" : self.features[index],
                "generator" : self.gen[index],
                "name": self.name[index]}
    
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
            labels[i] = FAKE_LABEL if self.int_to_gen[e] != REAL_IMG_GEN else REAL_LABEL
        return labels
    
    def save(self,output_path: str):
        torch.save({
            "features": self.features,
            "label": self.label,
            "gen": self.gen,
            "gen_original_name": self.gen_original_name,
            "int_to_gen": self.int_to_gen,
            "gen_to_int": self.gen_to_int,
            "img_per_gen": self.img_per_gen,
            "n_real": self.n_real,
            "n_fake": self.n_fake,
            "name": self.name
        },output_path)

    
class DeepFakeDatasetFastLoad(Dataset): # /data3/AID
    """Class used to load DeepFakeDataset that has been serialized"""
    def __init__(self, path_to_data: str, remove_blacklisted_gen=False) -> None:
        data = torch.load(path_to_data,"cpu")
        self.img_per_gen  = data["img_per_gen"]
        self.gen_to_int   = data["gen_to_int"]
        self.int_to_gen   = data["int_to_gen"]
        self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 
        self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL}
        self.n_real       = data["n_real"]
        self.n_fake       = data["n_fake"]
        if remove_blacklisted_gen:
            self.features = torch.empty((0,CLIP_FEATURE_DIM))
            self.label = torch.empty(0).type_as(data["label"])
            self.gen = torch.empty(0).type_as(data["gen"])

            allowed_gen = [key for key in INT_TO_GEN if key not in AID_BLACKLIST]
            
            for gen in tqdm(allowed_gen,"filtering"):
                mask = data["gen"] == gen
                self.features = torch.cat((self.features,data["features"][mask]),dim=0)
                self.label    = torch.cat((self.label,data["label"][mask]),dim=0)
                self.gen      = torch.cat((self.gen,data["gen"][mask]))

        else:
            self.features = data["features"]
            self.gen      = data["gen"]
            self.label    = data["label"]

            
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"features": self.features[index],
                "label": self.label[index],
                "generator": self.gen[index]}
    
    def class_to_label(self, classes):
        """Maps an array of integers representing classes to a list of 0 and 1 depending on whether the class represented a generated or a real image.

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
    
    
class OOD(Dataset):
    def __init__(self, path_to_data: str,
                 load_preprocessed: bool,
                 device: str=device,
                 gen_to_int: dict=GEN_TO_INT_OOD,
                 int_to_gen: dict=INT_TO_GEN_OOD,
                 remove_blacklisted_gen: bool=False):
        if load_preprocessed:
            data = torch.load(path_to_data,device)
            self.gen_to_int = data["gen_to_int"] 
            self.int_to_gen = data["int_to_gen"] 
            self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"}
            self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL}
            
            if remove_blacklisted_gen:
                self.features = torch.empty((0,CLIP_FEATURE_DIM)).to(device)
                self.label = torch.empty(0).type_as(data["label"]).to(device)
                self.gen = torch.empty(0).type_as(data["gen"]).to(device)
                allowed_gen = [gen for gen in INT_TO_GEN_OOD if gen not in OOD_BLACKLIST_OLD]
                for gen in tqdm(allowed_gen,"filtering"):
                    mask = data["gen"] == gen
                    self.features = torch.cat((self.features,data["features"][mask]),dim=0)
                    self.label    = torch.cat((self.label,data["label"][mask]),dim=0)
                    self.gen      = torch.cat((self.gen,data["gen"][mask]),dim=0)

            else:
                self.features = data["features"]
                self.label    = data["label"]
                self.gen      = data["gen"]
        else:
            if not path_to_data.endswith("/"): path_to_data += "/"
            self.features = torch.empty((0,DINO_FEATURE_DIM))
            self.label    = torch.empty((0)).int()
            self.gen      = torch.empty((0)).int()
            self.gen_to_int = gen_to_int
            self.int_to_gen = int_to_gen
            self.int_to_label = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 
            self.label_to_int = {"fake":FAKE_LABEL,"real":REAL_LABEL}
            self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))
            
            model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
            model.eval()
            
            REAL_IMG_PATH = path_to_data + "real_images/"
            FAKE_IMG_PATH = path_to_data + "AI_images/"

            real_files = os.listdir(REAL_IMG_PATH)
            
            #================================= Real images processing =================================#
            real_imgs = [self.transform(Image.open(REAL_IMG_PATH + file)).unsqueeze(0).to(device) for file in tqdm(real_files,"Processing real images")]
            with torch.no_grad():
                features = model.encode_image(torch.cat(real_imgs,dim=0))
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label,torch.Tensor(len(real_imgs) * [REAL_LABEL])),dim=0)
            self.gen = torch.cat((self.gen,torch.ones(len(real_imgs)) * self.gen_to_int[REAL_IMG_GEN]),dim=0)
            torch.cuda.empty_cache()
            #================================= Fake images processing =================================#
            gen_folder = ['Lexica_images',
                          'Ideogram_images',
                          'Leonardo_images',
                          'Copilot_images',
                          'img2img_SD1.5_images',
                          'Photoshop_images_generativemagnification',
                          'Photoshop_images_generativefill']
            
            generators = ['Lexica',
                          'Ideogram',
                          'Leonardo',
                          'Copilot',
                          'img2img_SD1.5',
                          'Photoshop_generativemagnification',
                          'Photoshop_generativefill'] # clean names used as keys in gen_to_int 
            
            for i, gen in enumerate(gen_folder):
                fake_imgs = []
                desc = f"Processing images from {generators[i]}"
                
                for file in tqdm(os.listdir(FAKE_IMG_PATH + gen),desc=desc):
                    img = Image.open(FAKE_IMG_PATH + gen + "/" + file)
                    fake_imgs.append(self.transform(img).unsqueeze(0).to(device))
                with torch.no_grad():
                    features = model.encode_image(torch.cat(fake_imgs,dim=0))
                self.features = torch.cat((self.features,features.cpu()),dim=0)
                self.label = torch.cat((self.label,torch.Tensor(len(fake_imgs) * [FAKE_LABEL])),dim=0)
                self.gen = torch.cat((self.gen,torch.Tensor(len(fake_imgs) * [self.gen_to_int[generators[i]]])),dim=0)
                torch.cuda.empty_cache()

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"label": self.label[index],
                "features": self.features[index],
                "generator": self.gen[index]}

    def class_to_label(self, classes):
        """Maps an array of integers representing classes to a list of 0 and 1 depending on whether the class represented a generated or a real image.

        Args:
            classes (_type_): array of classes taking values in DeepFakeDataset.int_to_gen.keys()

        Returns:
            _type_: array of 0 and 1 representing fake and real labels 
        """
        if isinstance(classes,torch.Tensor):
            classes = classes.cpu().numpy()
        labels = torch.zeros(len(classes))
        for i, e in enumerate(classes):
            labels[i] = FAKE_LABEL if self.int_to_gen[e] != REAL_IMG_GEN else REAL_LABEL
        return labels

    def save(self, output_path: str):
        torch.save({
            "features": self.features,
            "label": self.label,
            "gen": self.gen,
            "int_to_gen": self.int_to_gen,
            "gen_to_int": self.gen_to_int,
        }, output_path)


class DeepFakeTest(Dataset): #/data3/AID_TEST
    def __init__(self, 
                 path_to_data: str,
                 load_from_disk: bool = False,
                 img_per_gen: int = 100,
                 balance_real_fake: bool = True,
                 device: str = device,
                 remove_blacklisted_gen: bool=False):
        
        if load_from_disk:
            data = torch.load(path_to_data,device)
            if remove_blacklisted_gen:
                allowed_gen = [gen for gen in INT_TO_GEN if gen not in AID_TEST_BLACKLIST]
                self.features = torch.empty((0,CLIP_FEATURE_DIM)).to(device)
                self.label = torch.empty(0).type_as(data["label"]).to(device)
                self.gen = torch.empty(0).type_as(data["gen"]).to(device)
                allowed_gen.sort(reverse=True) # sort reverse -> gen 0 last (need to process other generators first)
                for gen in tqdm(allowed_gen,"filtering"):
                    mask = data["gen"] == gen
                    if gen == GEN_TO_INT[REAL_IMG_GEN]:
                        n_real = len(self.label)
                        self.features = torch.cat((self.features,data["features"][mask][:n_real]),dim=0)
                        self.label    = torch.cat((self.label,data["label"][mask][:n_real]),dim=0)
                        self.gen      = torch.cat((self.gen,data["gen"][mask][:n_real]),dim=0)
                    else:
                        self.features = torch.cat((self.features,data["features"][mask]),dim=0)
                        self.label    = torch.cat((self.label,data["label"][mask]),dim=0)
                        self.gen      = torch.cat((self.gen,data["gen"][mask]),dim=0)

            else:
                self.features   = data["features"]
                self.label      = data["label"]
                self.gen        = data["gen"]
                self.gen_original_name = data["gen_original_name"]
        else:
            if not path_to_data.endswith("/"):
                path_to_data += "/"

            model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
            model.eval()

            self.features = torch.empty((0,CLIP_FEATURE_DIM))
            self.label = torch.empty(0).int()
            self.gen   = torch.empty(0).int()
            self.gen_original_name = []
            self.transform = self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))

            real_folder = "Source_00_RealPhoto"
            N_GENERATORS = 30 # 32 generators initialy but 2 are in constants.BLACKLIST

            for gen in os.listdir(path_to_data):
                if gen in BLACKLIST: continue
                imgs = []
                files = os.listdir(path_to_data + gen)
                if gen == real_folder and balance_real_fake:
                    n_imgs = N_GENERATORS * img_per_gen
                else:
                    n_imgs = img_per_gen
                k = 0
                with tqdm(total=n_imgs,desc=f"{gen}") as bar:
                    for i, file in enumerate(files):
                        bar.n = i + 1
                        bar.refresh()
                        
                        img = Image.open(path_to_data + gen + "/" + file)
                        imgs.append(self.transform(img).unsqueeze(0).to(device))
                        k += 1
                        if k == n_imgs: break

                        if len(imgs) == CUDA_MEMORY_LIMIT:
                            with torch.no_grad():
                                features = model.encode_image(torch.cat(imgs,dim=0))
                            self.features = torch.cat((self.features,features.cpu()),dim=0)
                            imgs = []
                            torch.cuda.empty_cache()

                    if imgs:
                        with torch.no_grad():
                            features = model.encode_image(torch.cat(imgs,dim=0))
                        self.features = torch.cat((self.features,features.cpu()),dim=0)
                        torch.cuda.empty_cache()

                    self.label = torch.cat((self.label, torch.ones(n_imgs) * class2label(gen2int(gen))),dim=0)
                    self.gen = torch.cat((self.gen,torch.ones(n_imgs) * gen2int(gen)),dim=0)
                    self.gen_original_name += n_imgs * [gen]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"label": self.label[index],
                "features": self.features[index],
                "generator": self.gen[index]}
    
    def save(self, output_path: str):
        torch.save({
        "features": self.features,
        "label": self.label,
        "gen": self.gen,
        "gen_original_name": self.gen_original_name}, output_path)


class RealFakePairs(Dataset): #/data3/AID_pairs_orig_gen
    def __init__(self, 
                 path_to_real_imgs: str="",
                 path_to_fake_imgs: str="", 
                 img_per_class: int=1, 
                 device: str="cpu", 
                 load_from_disk: bool=False,
                 path: str="",
                 feature_type=CLIP):
        assert feature_type in (CLIP,DINO)
        
        if not path_to_real_imgs.endswith("/"): path_to_real_imgs += "/"
        if not path_to_fake_imgs.endswith("/"): path_to_fake_imgs += "/"
        self.int_to_label = INT_TO_LABEL
        self.label_to_int = LABEL_TO_INT
        
        if load_from_disk:
            assert path != ""
            data = torch.load(path)
            self.features = data["features"]
            self.label    = data["label"]
            # self.name     = data["name"] 

        else:
            if feature_type == CLIP:
                model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
                self.features = torch.empty((0,CLIP_FEATURE_DIM))
                self.transform= lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))
            elif feature_type == DINO:
                DINO_BATCH_SIZE = 100
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
                self.features = torch.empty((0,DINO_FEATURE_DIM))
                self.transform = lambda img : Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"])

            model.eval()

            real_imgs = [file for file in os.listdir(path_to_real_imgs) if file.endswith(".jpg")]
            fake_imgs = [file for file in os.listdir(path_to_fake_imgs) if file.endswith(".jpg")]

            self.label    = torch.empty(0)
            self.name     = []

            def extract_features_from_files(files: list, 
                                            path_to_folder: str, 
                                            device: str, 
                                            feature_type=feature_type) -> torch.Tensor:
                if feature_type == CLIP:
                    preprocessed_imgs = []
                    for file in tqdm(files[:img_per_class],total=img_per_class):
                        img = Image.open(path_to_folder + file)
                        preprocessed_imgs.append(self.transform(img).unsqueeze(0).to(device))
                        self.name.append(file)
                    with torch.no_grad():
                        return model.encode_image(torch.cat(preprocessed_imgs,dim=0))
                elif feature_type == DINO:
                    outputs = []
                    files = files[:img_per_class]
                    self.name += [file for file in files]
                    for i in tqdm(range(0,len(files),DINO_BATCH_SIZE),path_to_folder):
                        imgs = [self.transform(Image.open(path_to_folder + file)) for file in files[i:min(len(files),(i+DINO_BATCH_SIZE))]]
                        inputs  = processor(images=imgs,return_tensors="pt")
                        with torch.no_grad():
                            outputs.append(model(inputs["pixel_values"].to(device))[1].cpu())
                    return torch.cat(outputs,dim=0)


            # Real images
            features = extract_features_from_files(real_imgs, path_to_real_imgs, device, feature_type)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label    = torch.cat((self.label,REAL_LABEL * torch.ones(len(features))))
            torch.cuda.empty_cache()

            # Fake images
            features = extract_features_from_files(fake_imgs, path_to_fake_imgs, device, feature_type)
            self.features = torch.cat((self.features, features.cpu()),dim=0)
            self.label    = torch.cat((self.label,FAKE_LABEL * torch.ones(len(features))))
            torch.cuda.empty_cache()

            self.label = self.label.type(torch.LongTensor)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"label": self.label[index],
                "features": self.features[index],
                "name": self.name[index]}
    
    def save(self, output_path: str):
        torch.save({
            "features": self.features,
            "label": self.label,
            "label_to_int": self.label_to_int,
            "int_to_label": self.int_to_label,
            "name":self.name}, output_path)


class DoubleCLIP(Dataset):
    def __init__(self,
                 load_from_disk: bool=False,
                 path_to_datset: str="",
                 path_to_Blip_model_cache: str="/data4/saland/cache",
                 path_to_imgs: str="",
                 imgs_per_label: int=100,
                 num_inference_steps: int=100,
                 device: str="cpu"):
        
        if load_from_disk:
            data = torch.load(path_to_datset)
            self.features   = data["features"]
            self.label      = data["label"]
            self.imgs_names = data["imgs_names"]
        else:
            if not path_to_imgs.endswith("/"): path_to_imgs += "/"
            real_folder_name = "originals/"
            fake_folder_name = "generated/"


            model_CLIP, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
            model_CLIP.eval()
            
            processor = BlipProcessor.from_pretrained("unography/blip-long-cap",cache_dir=path_to_Blip_model_cache)
            model_Blip = BlipForConditionalGeneration.from_pretrained("unography/blip-long-cap",cache_dir=path_to_Blip_model_cache).to(device)
            diffusion_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
            diffusion_pipeline.set_progress_bar_config(disable=True)

            self.features = torch.empty((0,CLIP_FEATURE_DIM*2))
            self.label    = torch.empty(0)
            self.imgs_names = []
            self.transform = self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))

            def extract_features_from_files(files: list, 
                                            path_to_folder: str,
                                            model_CLIP: open_clip.CLIP,
                                            model_Blip: BlipForConditionalGeneration,
                                            processor_Blip: BlipProcessor,
                                            diffusion_pipeline: str,
                                            num_inference_steps: int,
                                            device: str) -> torch.Tensor:
                preprocessed_imgs = []
                preprocessed_imgs_gen = []
                for file in tqdm(files[:imgs_per_label],total=imgs_per_label):
                    img = Image.open(path_to_folder + file)
                    generated_img = generate(path_to_folder + file,
                                              model=model_Blip,
                                              processor=processor_Blip,
                                              diffusion_pipeline=diffusion_pipeline,
                                              num_inference_steps=num_inference_steps,
                                              device=device)
                    preprocessed_imgs.append(self.transform(img).unsqueeze(0).to(device))
                    preprocessed_imgs_gen.append(self.transform(generated_img).unsqueeze(0).to(device))
                    self.imgs_names.append(path_to_folder + file)

                with torch.no_grad():
                    features_imgs = model_CLIP.encode_image(torch.cat(preprocessed_imgs,dim=0))
                    features_imgs_gen = model_CLIP.encode_image(torch.cat(preprocessed_imgs_gen,dim=0))
                    return torch.cat((features_imgs,features_imgs_gen),dim=1)

            real_files = [file for file in os.listdir(path_to_imgs + real_folder_name) if file.endswith(".jpg")]
            fake_files = [file for file in os.listdir(path_to_imgs + fake_folder_name) if file.endswith(".jpg")]
            
            # Real images processing
            features = extract_features_from_files(files=real_files,
                                                   path_to_folder=path_to_imgs + real_folder_name,
                                                   model_CLIP=model_CLIP,
                                                   model_Blip=model_Blip,
                                                   processor_Blip=processor,
                                                   diffusion_pipeline=diffusion_pipeline,
                                                   num_inference_steps=num_inference_steps,
                                                   device=device)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label,torch.ones(imgs_per_label) * REAL_LABEL))
            torch.cuda.empty_cache()

            # Fake images processing
            features = extract_features_from_files(files=fake_files,
                                                   path_to_folder=path_to_imgs + fake_folder_name,
                                                   model_CLIP=model_CLIP,
                                                   model_Blip=model_Blip,
                                                   processor_Blip=processor,
                                                   diffusion_pipeline=diffusion_pipeline,
                                                   num_inference_steps=num_inference_steps,
                                                   device=device)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label,torch.ones(imgs_per_label) * FAKE_LABEL))
            torch.cuda.empty_cache()

            self.label = self.label.type(torch.LongTensor)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {"label":self.label[index],
                "features":self.features[index],
                "name":self.imgs_names[index]}
    
    def save(self,output_path: str):
        torch.save({
            "features":self.features,
            "label": self.label,
            "imgs_names": self.imgs_names},output_path)


class LongCaption(Dataset):
    def __init__(self, 
                 path: str="/data4/saland/data/Long_caption_images/", 
                 load_from_disk:bool=False,
                 device: str="cpu"):


        if load_from_disk:
            data = torch.load(path,device)
            self.features = data["features"]
            self.label = data["label"].type(torch.LongTensor)
            self.names = data["names"]
        else:
            if not path.endswith("/"):  path += "/"
            self.features = torch.empty((0,CLIP_FEATURE_DIM)).to(device)
            self.label    = torch.empty(0).type(torch.LongTensor).to(device)
            self.names    = []

            real_folder = "real_images/"
            fake_folder = "generated_images/"
            real_files  = [file for file in os.listdir(path + real_folder) if file.endswith(".jpg") or file.endswith(".jpeg")]
            fake_files  = [file for file in os.listdir(path + fake_folder) if file.endswith(".jpg") or file.endswith(".jpeg")]

            model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
            model.eval()

            self.transform = self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))

            def extract_features_from_files(files: list, path_to_folder: str, device: str) -> torch.Tensor:
                    preprocessed_imgs = []
                    for file in tqdm(files):
                        img = Image.open(path_to_folder + file)
                        preprocessed_imgs.append(self.transform(img).unsqueeze(0).to(device))
                        self.names.append(file)
                    with torch.no_grad():
                        return model.encode_image(torch.cat(preprocessed_imgs,dim=0))

            features = extract_features_from_files(real_files,path+real_folder,device)
            self.features = torch.cat((self.features,features),dim=0)
            self.label = torch.cat((self.label,torch.ones(len(features)).to(device) * REAL_LABEL),dim=0)

            features = extract_features_from_files(fake_files,path+fake_folder,device)
            self.features = torch.cat((self.features,features),dim=0)
            self.label    = torch.cat((self.label,torch.ones(len(features)).to(device) * FAKE_LABEL),dim=0)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"label": self.label[index],
                "features": self.features[index],
                "names":self.names[index]}

    def save(self,output_path:str):
        torch.save({"features":self.features,
                    "label":self.label,
                    "names":self.names},output_path)


class TestMeta(Dataset):
    def __init__(self,
                 path: str="/data3/test_meta_learning/",
                 load_from_disk:bool = False,
                 device: str="cpu",
                 feature_type: str=CLIP):
        
        if load_from_disk:
            data = torch.load(path)
            self.features = data["features"]
            self.label = data["label"]
            self.gen = data["gen"]
            self.gen_original_name = data["gen_original_name"]
            self.name = data["name"]
            self.folder = data["folder"]
            self.quality = data["quality"]
        else:
            if not path.endswith("/"): path += "/"
            self.features = torch.empty((0,CLIP_FEATURE_DIM))
            self.label = []
            self.gen = []
            self.gen_original_name = []
            self.name = []
            self.folder = []
            self.quality = [] # 100 - 90 - 65 - 40 (jpg)

            qualities = (40, 65, 90, 100)

            if feature_type == CLIP:
                model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
                self.transform = {
                    40: lambda img : preprocess(Image.fromarray(transform_40(image=np.asarray(img.convert("RGB")))["image"])),
                    65: lambda img : preprocess(Image.fromarray(transform_65(image=np.asarray(img.convert("RGB")))["image"])),
                    90: lambda img : preprocess(Image.fromarray(transform_90(image=np.asarray(img.convert("RGB")))["image"])),
                    100: lambda img : preprocess(Image.fromarray(transform_100(image=np.asarray(img.convert("RGB")))["image"]))}
            elif feature_type == DINO:
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
                self.transform = {
                    40: lambda img : Image.fromarray(transform_40(image=np.asarray(img.convert("RGB")))["image"]),
                    65: lambda img : Image.fromarray(transform_65(image=np.asarray(img.convert("RGB")))["image"]),
                    90: lambda img : Image.fromarray(transform_90(image=np.asarray(img.convert("RGB")))["image"]),
                    100: lambda img : Image.fromarray(transform_100(image=np.asarray(img.convert("RGB")))["image"])}
                
            model.eval()

            real_folder = "Orig/"
            fake_folder = "Gen/"

            generators = [gen for gen in os.listdir(path + fake_folder) if not gen.startswith(".")]


            def extract_features(imgs):
                with torch.no_grad():
                    if feature_type == CLIP:
                        return model.encode_image(torch.cat(imgs,dim=0))
                    elif feature_type == DINO:
                        outputs = []
                        for i in tqdm(range(0,len(imgs),DINO_BATCH_SIZE)):
                            imgs_batch = imgs[i:min(len(imgs),i+DINO_BATCH_SIZE)]
                            inputs  = processor(images=imgs_batch,return_tensors="pt")
                            with torch.no_grad():
                                outputs.append(model(inputs["pixel_values"].to(device))[1].cpu())
                        return torch.cat(outputs,dim=0)

            for gen in generators:
                files = [file for file in os.listdir(path + fake_folder + gen) if file.endswith("jpg") or 
                                                                                  file.endswith("jpeg") or
                                                                                  file.endswith("png")]
                for q in tqdm(qualities,"quality"):
                    imgs = []
                    for file in tqdm(files,str(gen)):
                        name = path + fake_folder + gen + "/" + file
                        img = Image.open(name)
                        if feature_type == CLIP:
                            imgs.append(self.transform[q](img).unsqueeze(0).to(device))
                        elif feature_type == DINO:
                            imgs.append(self.transform[q](img))

                        self.label.append(FAKE_LABEL)
                        self.gen.append(gen2int(gen))
                        self.gen_original_name.append(gen)
                        self.name.append(name)
                        self.folder.append(fake_folder + gen)
                        self.quality.append(q)

                        if len(imgs) == CUDA_MEMORY_LIMIT:
                            features = extract_features(imgs)
                            self.features = torch.cat((self.features,features.cpu()),dim=0)
                            imgs = []
                            torch.cuda.empty_cache()

                    if imgs:
                        features = extract_features(imgs)
                        self.features = torch.cat((self.features,features.cpu()),dim=0)
                        imgs = []
                        torch.cuda.empty_cache()

            real_subfolders = os.listdir(path + real_folder)
            for subfolder in real_subfolders:
                files = [file for file in os.listdir(path + real_folder + subfolder) if file.endswith("jpg") or 
                                                                                        file.endswith("jpeg") or
                                                                                        file.endswith("png")]
                for q in tqdm(qualities):
                    imgs = []
                    for file in tqdm(files,str(subfolder)):
                        name = path + real_folder + subfolder + "/" + file
                        img = Image.open(name)
                        if feature_type == CLIP:
                            imgs.append(self.transform[q](img).unsqueeze(0).to(device))
                        elif feature_type == DINO:
                            imgs.append(self.transform[q](img))

                        self.label.append(REAL_LABEL)
                        self.gen.append(gen2int(REAL_IMG_GEN))
                        self.gen_original_name.append(REAL_IMG_GEN)
                        self.name.append(name)
                        self.folder.append(real_folder + subfolder)
                        self.quality.append(q)

                        if len(imgs) == CUDA_MEMORY_LIMIT:
                            features = extract_features(imgs)
                            self.features = torch.cat((self.features,features.cpu()),dim=0)
                            imgs = []
                            torch.cuda.empty_cache()

                    if imgs:
                        features = extract_features(imgs)
                        self.features = torch.cat((self.features,features.cpu()),dim=0)
                        imgs = []
                        torch.cuda.empty_cache()


            self.gen = torch.Tensor(self.gen).type(torch.LongTensor)
            self.label = torch.Tensor(self.label).type(torch.LongTensor)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        return {"features":self.features[index],
                "label":self.label[index],
                "gen":self.gen[index],
                "name":self.name[index],
                "gen_original_name":self.gen_original_name[index]}

    def save(self, output_path:str):
        torch.save({
            "features": self.features,
            "label": self.label,
            "gen": self.gen,
            "name": self.name,
            "folder": self.folder,
            "quality":self.quality,
            "gen_original_name": self.gen_original_name}, output_path)
        
class FlickrAndPairs(Dataset): # mix of data from real_fake_pairs and Flickr + generated images from AID
    def __init__(self,
                 path: str="",
                 load_from_disk: bool=False,
                 device: str="cpu",
                 feature_type: str=CLIP):
        
        if load_from_disk:
            data = torch.load(path)
            self.features = data["features"]
            self.label = data["label"]
        else:
            pairs_data = RealFakePairs(device=device,
                                       load_from_disk=True,
                                       path="/data4/saland/data/real_fake_pairs_1000_name_DinoV2.pt")

            self.features = pairs_data.features
            self.label    = pairs_data.label
            
            if feature_type == CLIP:
                model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)  
                self.transform = lambda img : preprocess(Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"]))
            elif feature_type == DINO:
                DINO_BATCH_SIZE = 100
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
                self.transform = lambda img : Image.fromarray(transform_torch(image=np.asarray(img.convert("RGB")))["image"])

            model.eval()


            def extract_features_from_files(files, path: str):
                if feature_type == CLIP:
                    imgs = []
                    for file in tqdm(files,path):
                        img = Image.open(path + file)
                        imgs.append(self.transform(img).unsqueeze(0).to(device))
                    with torch.no_grad():
                        return model.encode_image(torch.cat(imgs,dim=0))
                elif feature_type == DINO:
                    outputs = []
                    for i in tqdm(range(0,len(files),DINO_BATCH_SIZE),path):
                        imgs    = [self.transform(Image.open(path + file)) for file in files[i:min(len(files),(i+DINO_BATCH_SIZE))]]
                        inputs  = processor(images=imgs,return_tensors="pt")
                        with torch.no_grad():
                            outputs.append(model(inputs["pixel_values"].to(device))[1].cpu())
                    return torch.cat(outputs,dim=0)

            path_to_flickr = "/data4/saland/data/2k_real_2k_fake/Flickr/"

            files = os.listdir(path_to_flickr)
            features = extract_features_from_files(files,path_to_flickr)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label, torch.ones(len(features)) * REAL_LABEL))

            n_per_gen = 333

            path_to_firefly = "/data4/saland/data/firefly/"
            files = os.listdir(path_to_firefly)[:n_per_gen]
            features = extract_features_from_files(files,path_to_firefly)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label, torch.ones(len(features)) * FAKE_LABEL))


            path_to_dalle3 = "/data4/saland/data/dalle3/"
            files = os.listdir(path_to_dalle3)[:n_per_gen]
            features = extract_features_from_files(files,path_to_dalle3)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label, torch.ones(len(features)) * FAKE_LABEL))
            
            path_to_midjourney = "/data4/saland/data/midjourney_v6/"
            files = os.listdir(path_to_midjourney)[:n_per_gen]
            features = extract_features_from_files(files,path_to_midjourney)
            self.features = torch.cat((self.features,features.cpu()),dim=0)
            self.label = torch.cat((self.label, torch.ones(len(features)) * FAKE_LABEL))

            self.label = self.label.type(torch.LongTensor)


    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        return {"features": self.features[index],
                "label": self.label[index]}
    
    def save(self, output_path: str):
        torch.save({"features":self.features,"label":self.label},output_path)

class TaskA(Dataset): # /data3/TEST/A
    def __init__(self, 
                 load_from_disk: bool, 
                 path: str="" ,
                 device: str="cpu",
                 features_type: str=CLIP):
        self.dir_name = "/data3/TEST/A/"
        
        if load_from_disk:
            data = torch.load(path)
            self.features = data["features"]
            self.image_name = data["image_name"] 
        else:
            assert features_type in (CLIP, DINO)
            if features_type == CLIP:
                model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
            elif features_type == DINO:
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
            model.eval()

            self.image_name = []
            self.features   = {}
            
            files = os.listdir(self.dir_name)

            for img_name in tqdm(files): 
                self.image_name.append(img_name)
                if features_type == CLIP:
                    img = preprocess(Image.open(self.dir_name + img_name)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        self.features[img_name] = model.encode_image(img)
                elif features_type == DINO:
                    img = Image.open(self.dir_name + img_name)
                    inputs = processor(images=[img],return_tensors="pt")
                    with torch.no_grad():
                        output = model(inputs["pixel_values"].to(device))[1].cpu()
                    self.features[img_name] = output
                
        
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self,index):
        return {
            "image_name": self.image_name[index],
            "dir_name": self.dir_name,
            "features": self.features[self.image_name[index]]}
    
    def save(self, output_path: str):
        torch.save({"features":self.features,"image_name":self.image_name},output_path)


class TaskAWithLabel(Dataset):
    def __init__(self, path_to_csv: str, path_to_taskA: str):
        data_csv = pd.read_csv(path_to_csv)
        data = TaskA(load_from_disk=True,path=path_to_taskA)

        self.features = []
        self.label = []
        self.features_dict = {}

        for name in tqdm(sorted(data.image_name)):
            self.features.append(data.features[name])
            self.label.append(1 - data_csv[data_csv["image_name"] == name]["class"].item()) # labels in csv are reversed

        self.features = torch.cat(self.features,dim=0)
        self.label = torch.Tensor(self.label).type(torch.LongTensor)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return {"features":self.features[index],"label":self.label[index]}
    
    def save(self, output_path: str):
        torch.save({"features":self.features,"label":self.label},output_path)

class SimpleDataset(Dataset):
    def __init__(self, features: torch.Tensor, label: torch.Tensor):
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index: int):
        return {"features":self.features[index], "label": self.label[index]}