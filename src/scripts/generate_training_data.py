import logging.config
from datasets import load_dataset, Dataset, DatasetInfo
import sys
sys.path.append("../tools/")
from utils import img_from_url
from tqdm import tqdm
import logging
import pytz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c","--count",required=True,type=int,help="number of samples")
parser.add_argument("-o","--output",required=True,type=str,help="path to the output directory")
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
seed = 7

# loading the dataset with streaming (not downloading the whole dataset at once)
ds = load_dataset("elsaEU/ELSA_D3",split="train",streaming=True)

d = {"id":[], "image":[], "label":[]}
k = 0
with tqdm(total=num_samples) as progress_bar:
    for e in ds:
        if k < num_samples:
            try:
                d["image"].append(img_from_url(e["url"])) 
            except Exception as e:
                logging.error(e)
                continue
            
            d["id"].append(e["id"])
            d["label"].append(1) # real images are labeled 1
            
            d["image"].append(e["image_gen0"])
            d["id"].append(e["id"])
            d["label"].append(0) # fake images are labeled 0
            k += 1

            progress_bar.n = k
            progress_bar.refresh()
        else:
            break

info = DatasetInfo("Each data point as an id linked to a (real,fake) images pair, an image and a label: 0 for fake and 1 for real") 

data = Dataset.from_dict(d,info=info).train_test_split(test_size=.5,shuffle=True,seed=seed)
data.save_to_disk(args.output)