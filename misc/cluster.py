import os
from glob import glob

import json
from pathlib import Path



def link(src, dst, remove_existing=False):
    if not os.path.exists(dst):
        os.symlink(src, dst)
    elif remove_existing:
        os.unlink(dst)
        os.symlink(src, dst)
    else:
        raise ValueError(f"path {dst} already exists")

def json_to_symlink(
        json_file:str,
        out_path:str,
        par_dir:str = "MoNuSegTrainingData",
        remove_existing = False,
        ):
    """
    creates symlink from json file.
    """

    with open(json_file, "r") as f:
        js = json.load(f)
        # print(js)

        

        for path, label in js['images'].items():

            os.makedirs(os.path.join(out_path, f"{label}", par_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_path, f"{label}", par_dir, "bin_masks"), exist_ok=True)
            os.makedirs(os.path.join(out_path, f"{label}", par_dir, "inst_masks"), exist_ok=True)

            file_name = os.path.basename(path)
            # file_name = Path(os.path.basename(img_path)).stem

            src = path
            dst = os.path.join(out_path, f"{label}", par_dir, "images", file_name)

            out_dir = os.path.join(out_path, f"{label}", par_dir, "images")
            if not os.path.exists(out_dir): os.makedirs(out_dir)

            if not os.path.exists(dst):
                os.symlink(src, dst)
            elif remove_existing:
                os.unlink(dst)
                os.symlink(src, dst)
            else:
                raise ValueError(f"path {dst} already exists")
            # print(dst)


        for path, label in js['bin_masks'].items():

            if not os.path.exists(os.path.join(out_path, label, par_dir)):
                os.makedirs(os.path.join(out_path, f"{label}", par_dir, "images"))
                os.makedirs(os.path.join(out_path, f"{label}", par_dir, "bin_masks"))            

            file_name = os.path.basename(path)
            # file_name = Path(os.path.basename(img_path)).stem

            src = path
            dst = os.path.join(out_path, f"{label}", par_dir, "bin_masks", file_name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            elif remove_existing:
                os.unlink(dst)
                os.symlink(src, dst)
            else:
                raise ValueError(f"path {dst} already exists")

            # print(dst)

        # instance masks : .tif files
        for path, label in js['images'].items():

            par_path = Path(path).parent.parent
            file_name = os.path.basename(path)
            src = os.path.join(par_path, 'inst_masks', file_name[:-3] + 'tif')
            if os.path.exists(src):
                dst = os.path.join(out_path, f"{label}", par_dir, "inst_masks", file_name[:-3] + 'tif')
                link(src, dst, remove_existing)



def symlink_to_json(
        out_path:str,
        json_file:str,
        par_folder:str = "MoNuSegTrainingData",
        extension:str = "png"
        ):
    """
    TODO : (incomplete)
    recovers json file from sylink. 
    """

    # Files should be saved in path : out_path/label/par_folder/
    img_file_names = sorted(glob(os.path.join(out_path, "*", par_folder, "images", f"*.{extension}")))
    mask_file_names = sorted(glob(os.path.join(out_path, "*", par_folder, "bin_masks", f"*.{extension}")))

    assert len(img_file_names) == len(mask_file_names)

    img_dic = {}
    mask_dic = {}

    for img_name in img_file_names:
        label = os.path.normpath(img_name).split(os.path.sep)[-4]
        img_dic[img_name] = label

    for mask_name in mask_file_names:
        label = os.path.normpath(mask_name).split(os.path.sep)[-4]
        mask_dic[mask_name] = label

    out_dic = {'images':img_dic, "bin_masks":mask_dic}

    return out_dic



if __name__ == "__main__":
    # out_path = "/mnt/dataset/MoNuSeg/patches_256x256_128x128/ResNet18_kmeans_10_v1.1/"
    out_path = "/mnt/dataset/MoNuSeg/patches_valid_inst_256x256_128x128/color_rand_ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2"

    modes = {
        "train.json" : "MoNuSegTrainingData",
        "valid.json" : "MoNuSegTestData",
    }

    for k, v in modes.items():
        js_name = os.path.join(out_path, k)
        par_dir = v

        json_to_symlink(js_name, out_path, par_dir, remove_existing=True)
