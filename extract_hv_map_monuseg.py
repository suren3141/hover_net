import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, os

# currentdir = os.path.abspath(os.getcwd())
# parentdir = os.path.dirname(currentdir)
# print(parentdir)
# sys.path.insert(0, parentdir) 

from config_monuseg import Config

from torch.utils.data import DataLoader

from dataloader.train_loader import MoNuSegDataset
from dataloader.utils import get_file_list

from models.hovernet.targets import gen_targets, prep_sample
from tqdm import tqdm
from PIL import Image

def extract_hv_map(dataloader):

    for batch in tqdm(dataloader):
        path = batch.pop('path')
        if isinstance(path, list): img_path, label_path = path
        if isinstance(img_path, tuple): img_path = img_path[0]
        for k in batch.keys():
            batch[k] = batch[k].numpy()
        out = prep_sample(batch, is_batch=True)

        file_path, file_name = os.path.split(img_path)
        out_path = os.path.join(os.path.dirname(file_path), "maps")
        if not os.path.exists(out_path): os.mkdir(out_path)

        out_pil = Image.fromarray(out)
        out_pil.save(os.path.join(out_path, file_name))


if __name__ == "__main__":

    config = Config()

    training_file_list = get_file_list(config.train_dir_list, config.file_type)
    valid_file_list = get_file_list(config.valid_dir_list, config.file_type)

    # print("Dataset %s: %d" % (run_mode, len(file_list)))
    train_dataset = MoNuSegDataset(
        training_file_list, file_type=config.file_type, mode="train", with_type=config.type_classification, 
        target_gen=(gen_targets, {}), input_shape=(256,256), mask_shape=(256,256))
    train_dataloader = DataLoader(train_dataset, num_workers= 8, batch_size= 1, shuffle=True, drop_last=True, )

    val_dataset = MoNuSegDataset(
        valid_file_list, file_type=config.file_type, mode="valid", with_type=config.type_classification, 
        target_gen=(gen_targets, {}), input_shape=(256,256), mask_shape=(256,256))
    val_dataloader = DataLoader(val_dataset, num_workers= 8, batch_size= 1, shuffle=False, drop_last=False, )

    extract_hv_map(train_dataloader)
    extract_hv_map(val_dataloader)