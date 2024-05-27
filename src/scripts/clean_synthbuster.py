from datasets import load_from_disk
import numpy as np

data = load_from_disk("../../data/synthbuster_test")

def squeeze(example):
    example["features"] = np.squeeze(np.array(example["features"]))
    return example

data = data.map(squeeze)

data.save_to_disk("../../data/sb_clip_color")