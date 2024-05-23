import argparse
import zipfile
import sys
sys.path.append("../tools")
from utils import convert_to_jpg, extract_clip_features, remove_directory
from constants import SEED, TMP_DIR
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_from_disk, DatasetInfo, Dataset
import open_clip
import torch

seed = SEED
img_directory = TMP_DIR

parser = argparse.ArgumentParser()
parser.add_argument("-o","--output",required=True,type=str,help="output dataset directory")
parser.add_argument("-q","--quality",required=True,type=int,help="quality factor of the jpeg compression")
parser.add_argument("-d","--device",required=False,default="0",type=str,help="which GPU to use '0' or '1'")
args = parser.parse_args()

d = {
    "generator": [],
    "features": [],
    "label": []
}

device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',device=device)

model.eval()

# Adding the generated images
with zipfile.ZipFile("../../data/synthbuster/synthbuster.zip",mode="r") as zf:
    nb_files = len([item for item in zf.infolist() if not item.is_dir()])
    with tqdm(total=nb_files,desc="Fake images processing...") as bar:
        i = 0
        for path in zf.namelist():
            if path.endswith(".png"):
                _, generator, name = path.split("/")
                with zf.open(path,"r") as f:
                    img = Image.open(f)
                    img_jpg = convert_to_jpg(img,name=name.split(".")[0] + ".jpg",quality=args.quality)
                    d["generator"].append(generator)
                    d["label"].append("fake")
                    d["features"].append(
                        extract_clip_features(
                            img_jpg,
                            model,
                            preprocess,
                            device))
                    i += 1
                    bar.n = i
                    bar.refresh()

# Adding the real images
data = load_from_disk("../../data/big_QF_" + str(args.quality))["test"].filter(lambda e : e["label"] == 1) # loading real images

with tqdm(total=data.num_rows,desc="Real images processing...") as bar:
    for i, e in enumerate(data):
        d["generator"].append("null")
        d["label"].append("real")
        d["features"].append(
            extract_clip_features(
                e["image"],
                model,
                preprocess,
                device))
        bar.n = i
        bar.refresh()

info = DatasetInfo(f"""
                   'generator': what model was used to generate the image, null if the image is real
                   \n'label': 'real' or 'generated',
                   \n'features': clip features of the image
                   """)

data = Dataset.from_dict(d,info=info)
data.save_to_disk(args.output)

# Removing temporary files
remove_directory(img_directory)