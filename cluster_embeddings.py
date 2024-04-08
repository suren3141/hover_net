import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, os
import pandas as pd

# currentdir = os.path.abspath(os.getcwd())
# parentdir = os.path.dirname(currentdir)
# print(parentdir)
# sys.path.insert(0, parentdir) 

from config_monuseg import Config

# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models import resnet101, ResNet101_Weights
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
    
    if model_name == "ResNet101":
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
    elif model_name == "ResNet50":
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

    assert os.path.splitext(out_name)[-1] == ".joblib", "save models with extension joblib"

    joblib.dump(model, out_name)


def exp_cluster(cluster_model_name, n_clusters, scaled_train_features, scaled_val_features, iterations=10):

    bst_sil=[]
    models = []
    mx_size = []
    mn_size = []

    for it in tqdm(range(iterations)):
        cluster_model = get_cluster_model(cluster_model_name, n_clusters, n_init=3).fit(scaled_train_features) 
        labels = cluster_model.predict(scaled_val_features)
        unique, counts = np.unique(cluster_model.predict(scaled_train_features), return_counts=True)
        try:
            sil=metrics.silhouette_score(scaled_val_features, labels, metric='euclidean')
            bst_sil.append(sil)
            models.append(cluster_model)
            mx_size.append(np.max(counts)/sum(counts))
            mn_size.append(np.min(counts)/sum(counts))

        except Exception as e:
            print(e)

    ind = np.argsort(bst_sil)[::-1]
    out = {
        "model" : models[ind[0]],
        "sil" : np.array(bst_sil)[ind].tolist(),
        "mx_size" : np.array(mx_size)[ind].tolist(),
        "mn_size" : np.array(mn_size)[ind].tolist(),
    }

    print(out['sil'])
    print(out['mx_size'])
    print(out['mn_size'])

    return out

import json

def cluster_to_json(file_names, labels, out_path, json_name):
    if not os.path.exists(out_path): os.mkdir(out_path)

    img_dic = {}
    mask_dic = {}

    if len(file_names) == 2:
        img_paths, ann_paths = file_names
        for img_path, ann_path, label in zip(img_paths, ann_paths, labels):

            img_dic[img_path] = str(label)
            mask_dic[ann_path] = str(label)


    else:
        for file_name, label in zip(file_names, labels):
            img_path, ann_path = file_name
            # label = get_class(img_path)

            img_dic[img_path] = str(label)
            mask_dic[ann_path] = str(label)

    out_file = os.path.join(out_path, json_name)
    with open(out_file, "w+") as f:
        json.dump({"images":img_dic, "bin_masks":mask_dic}, f)



if __name__ == "__main__":
    EMB_MODEL_NAME = "ResNet18"
    CLUSTER_MODEL_NAME = "kmeans"
    N_CLUSTERS = 10
    exp_name = "v1.1"

    LOG_DIR = os.path.join('./logs_clustered', EMB_MODEL_NAME)

    out_path = f"/mnt/dataset/MoNuSeg/patches_256x256_128x128/{EMB_MODEL_NAME}_{CLUSTER_MODEL_NAME}_{N_CLUSTERS}_{exp_name}"


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

    ## Feature extraction
    if LOG_DIR is not None and os.path.exists(Path(LOG_DIR)/'train'):
        train_labels = pd.read_csv(Path(LOG_DIR)/'train'/'metadata.tsv' ,sep='\t', header=None)[0].to_list()
        train_features = pd.read_csv(Path(LOG_DIR)/'train'/'features.tsv' ,sep='\t', header=None).to_numpy()
        train_file_names = pd.read_csv(Path(LOG_DIR)/'train'/'paths.tsv' ,sep='\t', header=None)
        train_file_names = train_file_names[0].to_list(), train_file_names[1].to_list()
        train_images = None
    else:
        train_images, train_labels, train_features, train_file_names = get_images_labels_features(train_dataloader, model_emb, preprocess)

    if LOG_DIR is not None and os.path.exists(Path(LOG_DIR)/'valid'):
        val_labels = pd.read_csv(Path(LOG_DIR)/'valid'/'metadata.tsv' ,sep='\t', header=None)[0].to_list()
        val_features = pd.read_csv(Path(LOG_DIR)/'valid'/'features.tsv' ,sep='\t', header=None).to_numpy()
        val_file_names = pd.read_csv(Path(LOG_DIR)/'valid'/'paths.tsv' ,sep='\t', header=None)
        val_file_names = val_file_names[0].to_list(), val_file_names[1].to_list()
        val_images = None
    else:
        val_images, val_labels, val_features, val_file_names = get_images_labels_features(val_dataloader, model_emb, preprocess)


    SS = StandardScaler()
    scaled_train_features = SS.fit_transform(train_features)
    scaled_val_features = SS.transform(val_features)

    ## Clustering
    if out_path is not None and os.path.exists(os.path.join(out_path, "model.joblib")):

        best_model = joblib.load(os.path.join(out_path, "model.joblib"))
        exp_out = None

        train_clusters = best_model.predict(scaled_train_features)
        val_clusters = best_model.predict(scaled_val_features)

    else:
        exp_out = exp_cluster(CLUSTER_MODEL_NAME, N_CLUSTERS, scaled_train_features, scaled_val_features)
        best_model = exp_out.pop('model')
        train_clusters = best_model.predict(scaled_train_features)
        val_clusters = best_model.predict(scaled_val_features)

    if LOG_DIR is not None:
        write_embedding(Path(LOG_DIR)/'train', train_images, train_features, [f"{x}_train" for x in train_clusters], paths=train_file_names)
        write_embedding(Path(LOG_DIR)/'valid', val_images, val_features, [f"{x}_val" for x in val_clusters], paths=val_file_names)


    if out_path is not None:
        cluster_to_json(train_file_names, train_clusters, out_path, "train.json")
        cluster_to_json(val_file_names, val_clusters, out_path, "valid.json")
        save_cluster_model(best_model, os.path.join(out_path, "model.joblib"))

        if exp_out is not None:

            with open(os.path.join(out_path, "exp.json"), "w+") as f:
                json.dump(exp_out, f)



        
