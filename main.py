import os
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from absl import app, flags
from tqdm import tqdm

from src.utils.dataset import DatasetWithIdx
from src.modules.models import FeatureExtractor


DISTANCE_TYPES = ["cosine_similarity", "ssim", "mse"]
FLAGS = flags.FLAGS
flags.DEFINE_string("output", "", help="define output dir")
flags.DEFINE_string("runname", "test", help="define the runname for weights")
flags.DEFINE_enum("distance", "cosine_similarity", DISTANCE_TYPES,
                  help="choose which distance between two feature vectors")
flags.DEFINE_bool("pre_calculate", False, help="whether to pre_calculate")

# parameters
device = torch.device("cuda")


def find_similarity(query):
    weight_paths = list((Path("weights")/FLAGS.runname).glob("*.pt"))
    res = []
    for wp in weight_paths:
        data = torch.load(wp)
        img_idxs = data["img_idxs"]
        feas = data["features"].to(device)
        queries = query.repeat(feas.size(0), 1)
        with torch.no_grad():
            if FLAGS.distance == "cosine_similarity":
                dis = torch.cosine_similarity(queries, feas)
        for d, i in zip(dis, img_idxs):
            res.append((i, d.cpu()))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res


def save_features(features, img_idxs, path):
    features = torch.cat(features)
    img_idxs = torch.cat(img_idxs)
    data = {
        "features": features,
        "img_idxs": img_idxs,
    }
    torch.save(data, path)


def denormalize(imgs, mean, std):
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    if imgs.dim() == 4:
        imgs = torch.stack([denorm(img) for img in imgs])
    elif imgs.dim() == 3:
        imgs = denorm(imgs)
    return imgs


def main(argv):
    # setup
    os.makedirs("weights", exist_ok=True)
    os.makedirs(Path("weights")/FLAGS.runname, exist_ok=True)
    # os.makedirs("outputs", exist_ok=True)
    os.makedirs(FLAGS.output, exist_ok=True)

    # prepare dataset (CIFAR10)
    transform = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=Image.ANTIALIAS),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    database = DatasetWithIdx(torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=True, transform=transform))
    query_imgs = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transform)
    # run only 1/10 dataset size for test
    indices = list(range(len(database)))
    indices = indices[:len(indices)//10]
    database_sampler = SequentialSampler(indices)
    database_loader = DataLoader(
        database, batch_size=64, sampler=database_sampler, shuffle=False,
        num_workers=4)

    # compute and store feature vectors of all candidates
    model = FeatureExtractor(fea_ext=models.vgg19_bn).to(device)
    model.eval()
    if FLAGS.pre_calculate:
        pbar = tqdm(
            total=(len(database)//database_loader.batch_size+1),
            dynamic_ncols=True)
        features = []
        img_idxs = []
        for i, batch in enumerate(database_loader):
            (images, labels), idxs = batch
            images = images.to(device)
            img_idxs.extend(idxs)
            with torch.no_grad():
                # extract features
                outputs = model(images)
                outputs = torch.flatten(outputs, start_dim=1)
                features.append(outputs)
                # save the features
                if len(features) == 50:
                    save_features(
                        features, img_idxs,
                        Path("weights")/FLAGS.runname/"{}.pt".format(i+1))
                    del features, img_idxs
                    features = []
                    img_idxs = []
                pbar.update(1)
        # reminder features
        if len(features) > 0:
            save_features(
                features, img_idxs,
                Path("weights")/FLAGS.runname/"{}.pt".format(i+1))
        pbar.close()

    # find top 1 similarity
    for j, (image, label) in enumerate(query_imgs):
        # query 5 imgs
        if j >= 5:
            break
        found_imgs = []
        found_imgs.append(
            denormalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        image = image.to(device)
        # save_path = Path("outputs") / Path(path).stem
        save_path = Path(FLAGS.output)
        os.makedirs(save_path, exist_ok=True)
        image = image.unsqueeze(0).to(device)
        query = model(image)
        query = torch.flatten(query, start_dim=1)
        res = find_similarity(query)
        for r in res[:3]:
            (found_img, _), _ = database[r[0]]
            found_imgs.append(
                denormalize(found_img, (0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)))
            print(r[0], r[1])
        found_imgs = torch.stack(found_imgs)
        found_imgs = torchvision.utils.make_grid(found_imgs)
        found_imgs = TF.to_pil_image(found_imgs)
        found_imgs.save(save_path / ("{}_{}".format(j, classes[label])+".jpg"))


if __name__ == "__main__":
    app.run(main)
