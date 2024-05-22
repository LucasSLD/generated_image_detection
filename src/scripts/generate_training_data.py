import logging.config
from datasets import load_dataset, Dataset, DatasetInfo
import sys
sys.path.append("../tools/")
from utils import img_from_url, convert_to_jpg, remove_directory, extract_clip_features
from constants import SEED, TMP_DIR
from tqdm import tqdm
import logging
import pytz
import argparse
from open_clip import create_model_and_transforms
import torch


parser = argparse.ArgumentParser()
parser.add_argument("-c","--count",required=True,type=int,help="Number of samples")
parser.add_argument("-o","--output",required=True,type=str,help="Path to the output directory")
parser.add_argument("-a","--all",required=True,type=bool,help="Include the 4 generators when True")
parser.add_argument("-q","--quality",required=False,type=int,default=100,help="Quality of the jpeg compression")
args = parser.parse_args()

logging_config = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S %Z',  # Include time zone name
            'timezone': pytz.timezone('Europe/Paris'),  # Set time zone explicitly
        },
    },
    'handlers': {
        'file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': '../../misc/data_generation.log',
            'formatter': 'standard',
        },
    },
    'root': {
        'handlers': ['file'],
        'level': 'ERROR',
    },
}

logging.config.dictConfig(logging_config)

num_samples = args.count
img_directory = TMP_DIR
seed = SEED

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)
model.eval()

# loading the dataset with streaming (not downloading the whole dataset at once)
ds = load_dataset("elsaEU/ELSA_D3",split="train",streaming=True)
d = {"id":[], "image":[], "label":[], "features":[]}
k = 0
with tqdm(total=num_samples) as progress_bar:
    for e in ds:
        if k < num_samples:
            try:
                img = img_from_url(e["url"])
            except Exception as e:
                logging.error(e)
                continue
            
            if args.quality < 100:
                img = convert_to_jpg(img,"real_"+e["id"]+".jpg",quality=args.quality)
            
            d["image"].append(img)
            d["features"].append(extract_clip_features(img,model,preprocess,device))
            d["id"].append(e["id"])
            d["label"].append(1) # real images are labeled 1
            
            if args.all:
                img = e["image_gen" + str(k%4)]
            else:
                img = e["image_gen0"]

            if args.quality < 100:
                img = convert_to_jpg(img,e["id"] + ".jpg",args.quality)

            d["image"].append(img)
            d["features"].append(extract_clip_features(img,model,preprocess,device))
            d["id"].append(e["id"])
            d["label"].append(0) # fake images are labeled 0
            k += 1

            progress_bar.n = k
            progress_bar.refresh()
        else:
            break

info = DatasetInfo(f"""
                   'id': identifies real/fake pairs,\n
                   'image': pil image,\n
                   'label': 1 if real, 0 if fake,\n
                   'features': clip features associated to the image
                   """
                   + f",\nquality of jpeg: {args.quality}" if args.quality < 100 else ""
                   + f",\nused several generators: " + str(args.all)) 

data = Dataset.from_dict(d,info=info).train_test_split(test_size=.5,shuffle=True,seed=seed)
data.save_to_disk(args.output)

# Removing temporary files
remove_directory(img_directory)