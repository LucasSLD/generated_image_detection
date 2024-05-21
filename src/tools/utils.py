import numpy as np
from math import log
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
from PIL import Image
from open_clip.model import CLIP
import torchvision
import torch
import os
from datasets import load_from_disk
from sklearn.svm import LinearSVC
from tqdm import tqdm

to_tensor = transforms.ToTensor()

def plot_np_array(img_numpy_array, title : str = None, colorbar=False,figsize=None):
    if figsize is not None: plt.figure(figsize=figsize)
    if title is not None: plt.title(title)
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
    plt.imshow(img_numpy_array)
    if colorbar: plt.colorbar()

def plot_tensor(tensor, title : str = None, colorbar=False,figsize=None):
    """Plots a tensor image

    Args:
        tensor (_type_): image to plot
        title (str, optional): Title of the plot. Defaults to None.
        colorbar (bool, optional): Color bar next to the plot. Defaults to False.
        figsize (_type_, optional): Plot size. Defaults to None.
    """
    if len(tensor.shape) == 4:
        array = tensor[0].permute(1,2,0).cpu().numpy()
    elif len(tensor.shape) == 3:
        array = tensor.permute(1,2,0).cpu().numpy()
    else:
        print("ShapeError: the input array should be 3D or 4D")
        return
    plot_np_array(array, title, colorbar, figsize)


def img_from_url(url: str):
    """Returns a PIL image from a given url

    Args:
        url (str): the url where the image is stored

    Returns:
        _type_: PIL image retrieved from the given url
    """
    try:
        response = requests.get(url,stream=True,timeout=1)
        if response.status_code == requests.codes.ok:
            return Image.open(response.raw)
        else:
            raise requests.exceptions.InvalidURL(f"No image was found at: {url}")
    except requests.Timeout as e:
        raise requests.exceptions.Timeout(e)
    except requests.RequestException as e:
        raise e
    
def convert_to_jpg(img, name: str, quality: int = 100):
    """Converts a PIL image to a JPEG image with the given quality

    Args:
        img: PIL image
        path (str): name of the image
        quality (int, optional): Quality factor between 0 and 100. Defaults to 100.

    Returns:
        _type_: jpeg image
    """
    tmp_directory="./_dump/"
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)
    img.convert("RGB").save(tmp_directory + name,"JPEG",quality=quality,subsampling=False)
    img_jpg = Image.open(tmp_directory + name,formats=["JPEG"])
    return img_jpg

def extract_clip_features(
        img,
        model: CLIP,
        preprocess: torchvision.transforms.transforms.Compose,
        device: str,
        normalize: bool = False):
    """Extract CLIP features from an image with the given CLIP pretrained model

    Args:
        model (CLIP): CLIP pretrained model
        preprocess (torchvision.transforms.transforms.Compose): preprocess function obtained with open_clip.create_model_and_transforms method
        device (str): "cpu" or "cuda"
        img (_type_): image from which features are extracted
        normalize (bool): when set to True, normalize the features. Default to False

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        preprocessed_img = preprocess(img).unsqueeze(0).to(device)
        return model.encode_image(preprocessed_img,normalize=normalize).detach().cpu().numpy()

def extract_clip_features_batch(
        batch, 
        model: CLIP, 
        preprocess: torchvision.transforms.transforms.Compose,
        device: str,
        normalize: bool = False,
        show_progress_bar: bool = False):
    """Extract CLIP features from a batch of images

    Args:
        model (CLIP): CLIP pretrained model
        preprocess (torchvision.transforms.transforms.Compose): preprocess function obtained with open_clip.create_model_and_transforms method
        device (str): "cpu" or "cuda"
        normalize (bool): when set to True, normalize the features. Default to False
        show_progress_bar (bool): when True, plot progress bar
    """
    with tqdm(total=len(batch), disable=not show_progress_bar) as bar:
        for i, img in enumerate(batch):
            batch[i] = extract_clip_features(img,model,preprocess,device,normalize)
            if show_progress_bar:
                bar.n = i+1
                bar.refresh()

    return batch

def load_data_split(dataset_path: str,
                    split: str,
                    model: CLIP,
                    preprocess: torchvision.transforms.transforms.Compose,
                    device: str,
                    normalize: bool = False,
                    show_progress_bar: bool = False):
    """Load a split from a dataset

    Args:
        dataset_path (str): path to the dataset
        split (str): the split to load
        model (CLIP): CLIP pretrained model
        preprocess (torchvision.transforms.transforms.Compose): preprocess function obtained with open_clip.create_model_and_transforms method
        device (str): "cpu" or "cuda"
        normalize (bool): when set to True, normalize the features. Default to False
        show_progress_bar (bool): when True, plot progress bar

    Returns:
        _type_: (n_samples,n_features) feature matrix and (n_samples,) label vector
    """
    if split not in ("train", "test"):
        print("split argument must be equal to 'train' or 'test")
        exit
    if not dataset_path.endswith("/"):
        dataset_path += "/"

    ds = load_from_disk(dataset_path=dataset_path+split)
    X_split = extract_clip_features_batch(ds["image"],model,preprocess,device,normalize,show_progress_bar)
    X_split = np.array([x.flatten() for x in X_split])
    y_split = ds["label"]
    return X_split, y_split

def load_train_test(dataset_path: str,
                    model: CLIP,
                    preprocess: torchvision.transforms.transforms.Compose,
                    device: str,
                    normalize: bool = False,
                    show_progress_bar: bool = False):
    """Loads train and test data

    Args:
        dataset_path (str): path to the dataset
        model (CLIP): CLIP pretrained model
        preprocess (torchvision.transforms.transforms.Compose): preprocess function obtained with open_clip.create_model_and_transforms method
        device (str): "cpu" or "cuda"
        normalize (bool): when set to True, normalize the features. Default to False
        show_progress_bar (bool): when True, plot progress bar

    Returns:
        _type_: X_train (n_train, n_features), X_test (n_test, n_features), y_train (n_train,), y_test (n_test,)
    """
    X_train, y_train = load_data_split(dataset_path,"train",model,preprocess,device,normalize, show_progress_bar)
    X_test, y_test   = load_data_split(dataset_path,"test",model,preprocess,device,normalize, show_progress_bar)
    return X_train, X_test, y_train, y_test

def get_accuracy_clip_svc(train_dataset_path: str,
                 test_dataset_path: str,
                 model: CLIP,
                 preprocess: torchvision.transforms.transforms.Compose,
                 device: str,
                 normalize: bool = False,
                 show_progress_bar: bool = False):
    """Compute the accuracy of a SVM classifier on a given test set

    Args:
        train_dataset_path (str): path to the training dataset
        test_dataset_path (str): path to the test dataset
        model (CLIP): CLIP pretrained model
        preprocess (torchvision.transforms.transforms.Compose): preprocess function obtained with open_clip.create_model_and_transforms method
        device (str): "cpu" or "cuda"
        normalize (bool): when set to True, normalize the features. Default to False
        show_progress_bar (bool): when True, plot progress bar

    Returns:
        _type_: _description_
    """
    X_train, y_train = load_data_split(train_dataset_path,
                                       split="train",
                                       model=model,
                                       preprocess=preprocess,
                                       device=device,
                                       normalize=normalize,
                                       show_progress_bar=show_progress_bar)
    
    X_test, y_test = load_data_split(test_dataset_path,
                                     split="test",
                                     model=model,
                                     preprocess=preprocess,
                                     device=device,
                                     normalize=normalize,
                                     show_progress_bar=show_progress_bar)
    
    clf = LinearSVC()
    return clf.fit(X_train,y_train).score(X_test,y_test)

def get_histograms(img: Image.Image, mode: str):
    """Returns the histograms from the 3 channels of the image in the given mode

    Args:
        img (Image.Image): PIL image
        mode (str): "RGB" or "HSV"

    Returns:
        _type_: (1st channel histogram, 2nd channel histogram, 3rd channel histogram)
    """

    img = img.convert(mode)
    c1, c2, c3 = img.split()
    return c1.histogram(), c2.histogram(), c3.histogram()