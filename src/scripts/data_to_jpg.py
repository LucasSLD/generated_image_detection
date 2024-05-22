from datasets import load_from_disk, Dataset, DatasetInfo
from copy import deepcopy
import sys
sys.path.append("../tools")
from utils import convert_to_jpg, remove_directory
import argparse
from tqdm import tqdm
import os

"""
This script is used to convert dataset withan "image" field that contains PIL images to jpg images.
"""


seed = 7
img_directory = "./_dump/"

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True,type=str,help="path to the source dataset")
parser.add_argument("-o","--output",required=True,type=str,help="output dataset directory")
parser.add_argument("-q","--quality",required=True,type=int,help="quality factor of the jpeg compression")

args = parser.parse_args()

ds_train = load_from_disk(args.input)["train"]
ds_test  = load_from_disk(args.input)["test"]

ds = {"id":[],"image":[],"label":[]}
for key in ds.keys():
    # concatenation of train and test datasets
    ds[key] = ds_train[key] + ds_test[key]

ds_jpg = deepcopy(ds)

for i in tqdm(range(len(ds["id"]))):
    ds_jpg["image"][i] = convert_to_jpg(ds["image"][i],str(i) +".jpeg",quality=args.quality)

info = DatasetInfo(
    f"""Each data point as an id linked to a (real,fake) images pair,
    \njpg image with quality {args.quality}
    \nlabel: 0 for fake and 1 for real\n""") 

data = Dataset.from_dict(ds_jpg,info=info).train_test_split(test_size=.5,shuffle=True,seed=seed)
data.save_to_disk(args.output + "_QF_" + str(args.quality))

# cleaning
remove_directory(img_directory)