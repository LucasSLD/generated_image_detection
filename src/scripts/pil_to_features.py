import sys
sys.path.append("../tools")
from utils import extract_clip_features_batch
from constants import SEED
from datasets import load_from_disk
import argparse
from open_clip import create_model_and_transforms
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str,required=True,help="path to the dataset to convert")
parser.add_argument("-o","--output",type=str,required=True,help="where to save the new dataset")

args = parser.parse_args()

seed = SEED

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
model.eval()
