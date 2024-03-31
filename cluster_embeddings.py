import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, os

# currentdir = os.path.abspath(os.getcwd())
# parentdir = os.path.dirname(currentdir)
# print(parentdir)
# sys.path.insert(0, parentdir) 

from config_monuseg import Config

# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch import nn

# from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from get_embedding import get_images_labels_features, write_embedding

from dataloader.train_loader import MoNuSegDataset
from dataloader.utils import get_file_list

from models.hovernet.targets import gen_targets
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans, SpectralClustering
from pathlib import Path


import joblib

    
def plot_clusters(X, labels, cmap=True, ax=None):
    ax = ax or plt.gca()
    if cmap:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)

    n_labels = np.unique(labels)
        
    plt.title("KMeans with %d components"%len(n_labels), fontsize=(20))
    plt.xlabel("U.A.")
    plt.ylabel("U.A.")

def get_emb_model(model_name):
    
    if model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        raise NotImplementedError()

    model_emb = nn.Sequential(*list(model.children())[:-1]) # strips off last linear layer

    model_emb.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    return model_emb, preprocess

def get_cluster_model(cluster_model, n_clusters, **kwargs):
    if cluster_model == "gmm":
        f = GMM
    elif cluster_model == "kmeans":
        f = KMeans
    if cluster_model == "spectral":
        f = SpectralClustering
    
    return f(n_clusters, **kwargs)

def save_cluster_model(model, out_name):

    assert Path(out_name).stem == "joblib", "save models with extension joblib"

    joblib.dump(model, out_name)

def exp_cluster(cluster_model_name, n_clusters, scaled_train_features, scaled_val_features):

    bst_sil=[]
    models = []
    mx_size = []
    mn_size = []

    for it in tqdm(range(10)):
        cluster_model = get_cluster_model(cluster_model_name, n_clusters, n_init=3).fit(scaled_train_features) 
        labels = cluster_model.predict(scaled_val_features)
        unique, counts = np.unique(labels, return_counts=True)
        try:
            sil=metrics.silhouette_score(scaled_val_features, labels, metric='euclidean')
            bst_sil.append(sil)
            models.append(cluster_model)
            mx_size.append(np.max(counts)/sum(counts))
            mn_size.append(np.min(counts)/sum(counts))

        except Exception as e:
            print(e)

    ind = np.argsort(bst_sil)[::-1]
    print(bst_sil[ind])
    print(mx_size[ind])
    print(mn_size[ind])

    return models[ind[0]]

import json

def cluster_to_json(file_names, labels, out_path, json_name):
    if not os.path.exists(out_path): os.mkdir(out_path)

    img_dic = {}
    mask_dic = {}

    for file_name, label in zip(file_names, labels):
        img_path, ann_path = file_name[0], file_name[1]
        # label = get_class(img_path)

        img_dic[img_path] = label
        mask_dic[ann_path] = label

    out_file = os.path.join(out_path, json_name)
    with open(out_file, "w+") as f:
        json.dump({"images":img_dic, "bin_masks":mask_dic}, f)



if __name__ == "__main__":
    EMB_MODEL_NAME = "ResNet50"
    CLUSTER_MODEL_NAME = "gmm"
    N_CLUSTERS = 5

    LOG_DIR = os.path.join('./logs_clustered', EMB_MODEL_NAME)

    out_path = f"/mnt/dataset/MoNuSeg/patches_256x256_128x128/{CLUSTER_MODEL_NAME}_{N_CLUSTERS}"


    config = Config()
    log_path = None

    training_file_list = get_file_list(config.train_dir_list, config.file_type)
    valid_file_list = get_file_list(config.valid_dir_list, config.file_type)

    # print("Dataset %s: %d" % (run_mode, len(file_list)))
    train_dataset = MoNuSegDataset(
        training_file_list, file_type=config.file_type, mode="train", with_type=config.type_classification, 
        target_gen=(gen_targets, {}), **config.shape_info["train"])

    val_dataset = MoNuSegDataset(
        valid_file_list, file_type=config.file_type, mode="valid", with_type=config.type_classification, 
        target_gen=(gen_targets, {}), **config.shape_info["valid"])

    train_dataloader = DataLoader(train_dataset, num_workers= 8, batch_size= 8, shuffle=True, drop_last=True, )
    val_dataloader = DataLoader(val_dataset, num_workers= 8, batch_size= 8, shuffle=False, drop_last=False, )



    model_emb, preprocess = get_emb_model(EMB_MODEL_NAME)

    train_images, train_labels, train_features, train_file_names = get_images_labels_features(train_dataloader, model_emb, preprocess)
    val_images, val_labels, val_features, val_file_names = get_images_labels_features(val_dataloader, model_emb, preprocess)


    SS = StandardScaler()
    scaled_train_features = SS.fit_transform(train_features)
    scaled_val_features = SS.transform(val_features)

    best_model = exp_cluster(CLUSTER_MODEL_NAME, N_CLUSTERS, scaled_train_features, scaled_val_features)

    train_clusters = best_model.predict(scaled_train_features)
    val_clusters = best_model.predict(scaled_val_features)

    if LOG_DIR is not None:
        write_embedding(LOG_DIR, train_images, train_features, [f"{x}_train" for x in train_clusters])
        write_embedding(LOG_DIR, val_images, val_features, [f"{x}_train" for x in val_clusters])

    if out_path is not None:

        cluster_to_json(train_file_names, train_clusters, out_path, "train.json")
        cluster_to_json(val_file_names, val_clusters, out_path, "train.json")

        
