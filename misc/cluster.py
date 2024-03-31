import os
from glob import glob

import json

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

            if not os.path.exists(os.path.join(out_path, label)):
                os.makedirs(os.path.join(out_path, f"{label}", par_dir, "images"))
                os.makedirs(os.path.join(out_path, f"{label}", par_dir, "bin_masks"))            

            file_name = os.path.basename(path)
            # file_name = Path(os.path.basename(img_path)).stem

            src = path
            dst = os.path.join(out_path, f"{label}", par_dir, "images", file_name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            elif remove_existing:
                os.unlink(dst)
                os.symlink(src, dst)
            else:
                raise ValueError(f"path {dst} already exists")
            # print(dst)


        for path, label in js['bin_masks'].items():

            if not os.path.exists(os.path.join(out_path, label)):
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
    out_path = "/mnt/dataset/MoNuSeg/patches_256x256_128x128/names_37_14"

    js_name = "/mnt/dataset/MoNuSeg/patches_256x256_128x128/names_37_14/train.json"
    par_dir = "MoNuSegTrainingData"

    json_to_symlink(js_name, out_path, par_dir, remove_existing=True)

    js_name = "/mnt/dataset/MoNuSeg/patches_256x256_128x128/names_37_14/valid.json"
    par_dir = "MoNuSegTestData"

    json_to_symlink(js_name, out_path, par_dir, remove_existing=True)    