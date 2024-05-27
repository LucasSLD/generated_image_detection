import zipfile
import sys
sys.path.append("../tools")
from utils import convert_to_jpg, remove_directory, get_histograms
from constants import TMP_DIR
from datasets import load_from_disk, Dataset, DatasetInfo
from tqdm import tqdm
from PIL import Image
import argparse
import numpy as np

QUALITY = 40

parser = argparse.ArgumentParser()
parser.add_argument("-b","--bins",type=int,required=True,help="Number of bins for the histograms used as color features")
parser.add_argument("-o","--output",required=True,type=str,help="path of the output")
args = parser.parse_args()

data = load_from_disk("../../data/synthbuster_test")
d = {"generator": [],"features":[],"label":[]}
for key in data.column_names:
    d[key] = data[key]

with zipfile.ZipFile("../../data/synthbuster/synthbuster.zip",mode="r") as zf:
    nb_files = len([item for item in zf.infolist() if not item.is_dir()])
    with tqdm(total=nb_files,desc="Fake images processing") as bar:
        i = 0
        for path in zf.namelist():
            if path.endswith(".png"):
                _, generator, name = path.split("/")
                with zf.open(path,"r") as f:
                    img = Image.open(f)
                    img_jpg = convert_to_jpg(img,name=name.split(".")[0] + ".jpg",quality=QUALITY)
                    img_ycbcr = np.asarray(img_jpg.convert("YCbCr"))
                    cb, _ = np.histogram(img_ycbcr[:,:,1].flatten(),bins=args.bins)
                    cr, _ = np.histogram(img_ycbcr[:,:,2].flatten(),bins=args.bins)
                    d["features"][i] += list(cb) + list(cr)
                    i += 1
                    bar.n = i
                    bar.refresh()


data = load_from_disk("../../data/big_QF_" + str(QUALITY))["test"].filter(lambda e : e["label"] == 1) # loading real images

with tqdm(total=data.num_rows,desc="Real images processing") as bar:
    for j, e in enumerate(data):
        img_ycbcr = np.asarray(e["image"].convert("YCbCr"))
        cb, _ = np.histogram(img_ycbcr[:,:,1].flatten(),bins=args.bins)
        cr, _ = np.histogram(img_ycbcr[:,:,2].flatten(),bins=args.bins)
        d["features"][i] += list(cb) + list(cr)
        i += 1
        bar.n = j
        bar.refresh()

info = DatasetInfo(f"""
                   'generator': what model was used to generate the image, null if the image is real
                   \n'label': 'real' or 'generated',
                   \n'features': clip features + color features (histograms YCbCr)
                   """)

data = Dataset.from_dict(d,info=info)
data.save_to_disk(args.output)

remove_directory(TMP_DIR)