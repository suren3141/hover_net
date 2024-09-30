# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch import nn, is_tensor

# from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from config_monuseg import Config
from dataloader.train_loader import FileLoader, MoNuSegDataset
from dataloader.utils import get_file_list

from models.hovernet.targets import gen_targets
from models.hovernet.opt import get_config


from PIL import Image
import numpy as np
import os
import re
from run_train import TrainManager

from tqdm import tqdm

from typing import Union

IMG_WIDTH, IMG_HEIGHT = (64, 64)

def get_class(path):
    match = re.findall('gmm_\d/\d', path)
    if match is not None and len(match) > 0 :
        cls = os.path.split(match[0])[-1]
    else:
        cls = os.path.splitext(os.path.split(path)[-1])[0][:-4]
    
    return cls

def resize_images(img):
    if is_tensor(img[0]):
        return [to_pil_image(i.permute(2, 0, 1)).resize((IMG_WIDTH, IMG_HEIGHT)) for i in img]
    elif isinstance(img[0], np.ndarray):
        return [Image.fromarray(i).resize((IMG_WIDTH, IMG_HEIGHT)) for i in img]


def get_emb_model(model_name):
    
    if model_name == "ResNet101":
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
    elif model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == "color":
        raise NotImplementedError()        
    else:
        raise NotImplementedError()

    model_emb = nn.Sequential(*list(model.children())[:-1]) # strips off last linear layer

    model_emb.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    return model_emb, preprocess



def get_file_path(path):
    if isinstance(path, list):
        img_path, ann_path = path
    elif isinstance(path, str):
        img_path = path
        ann_path = None
    else:
        raise ValueError(type(path))

    is_batch = True if isinstance(img_path, tuple) else False

    return img_path, ann_path

def get_images_labels_features(dataloader, model, preprocess, PREPROCESS_IMG=None):            


    images = []
    labels = []
    features = []
    img_paths = []
    ann_paths = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        # Step 3: Apply inference preprocessing transforms
        img = batch['img']
        path = batch['path']

        if PREPROCESS_IMG:
            img = PREPROCESS_IMG(img)


        img_path, ann_path = get_file_path(path)

        assert img.ndim == 4, "Missing batches of RGB"

        feature = extract_features(img.permute(0, 3, 1, 2), model, preprocess)

        # print(feature.shape)
        # img = load_and_resize_image(img_path, IMG_WIDTH, IMG_HEIGHT)
        cls = [get_class(p) for p in img_path]

        images.extend(resize_images(img))
        labels.extend(cls)
        features.extend(feature)
        img_paths.extend(img_path)
        ann_paths.extend(ann_path)

    return images, labels, features, (img_paths, ann_paths)

def get_images(paths):
    images = []
    for path in paths:
        img_path, ann_path = get_file_path(path)
        img = np.array(Image.open(img_path).convert('RGB'))
        images.extend(resize_images([img]))

    return images




def get_images(file_names):

    images = []

    for file in file_names:
        with Image.open(file) as img:
            img = img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
            images.append(img)

    return images



def extract_features(img, model, preprocess):

    batch = preprocess(img)
    # batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    emb = model(batch).squeeze().detach().numpy()
 
    return emb

if __name__ == "__main__":

    MODEL_NAME = "ResNet50"
    run_mode = "train"

    config = Config()

    dir_list = config.train_dir_list

    file_list = get_file_list(dir_list, config.file_type)

    print("Dataset %s: %d" % (run_mode, len(file_list)))
    input_dataset = MoNuSegDataset(
        file_list,
        file_type=config.file_type,
        mode=run_mode,
        with_type=config.type_classification,
        target_gen=(gen_targets, {}),
        **config.shape_info[run_mode]
    )

    dataloader = DataLoader(
        input_dataset,
        num_workers= 8,
        batch_size= 8,
        shuffle= run_mode == "train",
        drop_last= run_mode == "train",
    )


    # train_manager = TrainManager()
    # train_manager.nr_gpus = 1
    # phase_list = train_manager.model_config["phase_list"][0]
    # target_info = phase_list["target_info"]
    # mode = "train"
    # dataloader = train_manager._get_datagen(1, mode, target_info["gen"])

        

    # img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    if MODEL_NAME == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif MODEL_NAME == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        raise NotImplementedError()

    model_emb = nn.Sequential(*list(model.children())[:-1]) # strips off last linear layer

    model_emb.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    images, labels, features, paths = get_images_labels_features(dataloader, model_emb, preprocess)


    LOG_DIR = os.path.join('./logs_clustered', MODEL_NAME)
    write_embedding(LOG_DIR, images, features, labels)

