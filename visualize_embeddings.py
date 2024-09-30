import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, os
import pandas as pd

# currentdir = os.path.abspath(os.getcwd())
# parentdir = os.path.dirname(currentdir)
# print(parentdir)
# sys.path.insert(0, parentdir) 

# from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from get_embedding import get_images_labels_features, write_embedding

from dataloader.train_loader import MoNuSegDataset
from dataloader.utils import get_file_list


from sklearn.preprocessing import StandardScaler
import umap
from sklearn import metrics

from pathlib import Path

import joblib

from get_embedding import get_images_labels_features, get_emb_model
import glob, os

if __name__ == "__main__":
    EMB_MODEL_NAME = "ResNet50"
    EMB_TRANSFORM = ""


    LOG_DIR = os.path.join('./logs_vis/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4', EMB_MODEL_NAME)

    file_list = []
    file_labels = []

    gt_dirs = {
        "train": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTrainingData"],
        "test": ["/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/4/MoNuSegTestData"],
    }


    for k, v in gt_dirs.items():
        f = get_file_list(v, ".png")
        file_list.extend(f)
        file_labels.extend([f"{k}"] * len(f))


    syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_1000st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/4/"
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "*")))
    syn_dirs = {os.path.split(x)[-1] : x for x in syn_dirs}

    for k, v in syn_dirs.items():
        f = get_file_list(v, ".png", img_path="samples", ann_path="labels")
        file_list.extend(f)
        file_labels.extend([f"4_{k}"] * len(f))


    syn_pardir = "/mnt/dataset/MoNuSeg/out_sdm/monuseg_patches_128.64CH_1000st_1e-4lr_8bs_hv_ResNet18_kmeans_10_v1.1_4/ResNet18_kmeans_10_v1.1/2/"
    syn_dirs = sorted(glob.glob(os.path.join(syn_pardir, "*")))
    syn_dirs = {os.path.split(x)[-1] : x for x in syn_dirs}

    for k, v in syn_dirs.items():
        f = get_file_list(v, ".png", img_path="samples", ann_path="labels")
        file_list.extend(f)
        file_labels.extend([f"2_{k}"] * len(f))

    # print("Dataset %s: %d" % (run_mode, len(file_list)))
    dataset = MoNuSegDataset(
        file_list, file_type=".png", mode="test", with_type=False, 
        target_gen=(None, None), input_shape=(256,256), mask_shape=(256,256))
    dataloader = DataLoader(dataset, num_workers= 8, batch_size= 8, shuffle=False, drop_last=False, )


    model_emb, preprocess = get_emb_model(EMB_MODEL_NAME)

    ## Feature extraction
    images, labels, features, file_names = get_images_labels_features(dataloader, model_emb, preprocess)

    # if EMB_TRANSFORM == "ss":
    #     SS = StandardScaler()
    #     scaled_features = SS.fit_transform(features)
    # elif EMB_TRANSFORM == "umap":
    #     reducer = umap.UMAP(
    #         # n_neighbors=30,
    #         # min_dist=0.0,
    #         n_components=3,
    #         random_state=42,
    #     )
        
    #     scaled_features = reducer.fit_transform(features)

    # else:
    #     scaled_features = features



    if LOG_DIR is not None:
        write_embedding(Path(LOG_DIR), images, features, file_labels, paths=None)
