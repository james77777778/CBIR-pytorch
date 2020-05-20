import os
import os.path as osp
from pathlib import Path
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_size=float("inf")):
    images = []
    for root, dirs, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = osp.join(root, fname)
                images.append(path)
            if len(images) >= max_size:
                break
    return images


class ImageDataset(data.Dataset):
    def __init__(self, dataroot=""):
        self.dataroot = Path(dataroot)
        self.paths = sorted(make_dataset(self.dataroot))
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        area = img.size[0] * img.size[1]
        img = TF.resize(img, size=(224, 224), interpolation=Image.ANTIALIAS)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return {"tensor": tensor, "path": path,
                "area": torch.LongTensor([area])}


class DatasetWithIdx(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], torch.LongTensor([idx])
