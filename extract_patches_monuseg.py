"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import os.path as osp
import tqdm
import pathlib

import numpy as np
import cv2

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

from PIL import Image
import blobfile as bf

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False
    out_format = '.png'

    win_size = [256, 256]
    step_size = [128, 128]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "monuseg"
    save_root = "/mnt/dataset/MoNuSeg/patches/"

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "MoNuSegTrainingData": {
            "img": (".tif", "/mnt/dataset/MoNuSeg/dataset/MoNuSegTrainingData/images/"),
            "ann": (".png", "/mnt/dataset/MoNuSeg/dataset/MoNuSegTrainingData/bin_masks/"),
        },
        "MoNuSegTestData": {
            "img": (".tif", "/mnt/dataset/MoNuSeg/dataset/MoNuSegTestData/images/"),
            "ann": (".png", "/mnt/dataset/MoNuSeg/dataset/MoNuSegTestData/bin_masks/"),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )

        print(out_dir)
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)
        if not osp.exists(osp.join(out_dir, "images")): os.makedirs(osp.join(out_dir, "images"))
        if not osp.exists(osp.join(out_dir, "bin_masks")): os.makedirs(osp.join(out_dir, "bin_masks"))

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                if out_format == '.png':
                    img_patch, ann_patch = patch[:,:,:-1], patch[:,:,-1]
                    Image.fromarray(img_patch).save(f"{out_dir}/images/{base_name}_{idx:03d}.png")
                    Image.fromarray(ann_patch).save(f"{out_dir}/bin_masks/{base_name}_{idx:03d}.png")

                elif out_format == '.npy':
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
