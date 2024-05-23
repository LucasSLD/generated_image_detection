import sys
sys.path.append("../tools")
from utils import extract_clip_features
from constants import SEED
from datasets import load_from_disk, Dataset, DatasetInfo
import argparse
from open_clip import create_model_and_transforms
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str,required=True,help="path to the dataset to convert")
parser.add_argument("-o","--output",type=str,required=True,help="where to save the new dataset")

args = parser.parse_args()

seed = SEED

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
model.eval()


d = {"id":[], "image": [], "label":[], "features":[]}
data_train = load_from_disk(args.input)["train"]
data_test = load_from_disk(args.input)["test"]

for key in data_train.column_names:
    d[key] = data_train[key] + data_test[key]

for i in tqdm(range(len(d["id"]))):
    d["features"].append(extract_clip_features(d["image"][i],model,preprocess,device).flatten())

data = Dataset.from_dict(d).train_test_split(test_size=.5,shuffle=True, seed=seed)
data.save_to_disk(args.output)