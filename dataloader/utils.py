import glob, os, sys
from pathlib import Path
import numpy as np

def get_file_list(data_dir_list, file_type, img_path="images", ann_path='bin_masks', inst_path=None):
    """
    """

    if isinstance(data_dir_list, str): data_dir_list = [data_dir_list]

    file_list = []

    if file_type == '.png':
        for dir_path in data_dir_list:
            image_files =  sorted(glob.glob(os.path.join(dir_path, img_path, '*.png')))
            ann_files =  sorted(glob.glob(os.path.join(dir_path, ann_path, '*.png')))
            if inst_path is not None:
                inst_files =  sorted(glob.glob(os.path.join(dir_path, inst_path, '*.tif')))
                files = list(zip(image_files, ann_files, inst_files))
            else:
                inst_files = None
                files = list(zip(image_files, ann_files))


            file_list.extend(files)
            # file_list.extend(image_files)

        file_list = sorted(file_list, key=lambda x:x[0])       
        # file_list.sort()  # to always ensure same input ordering
    
    else:
        raise NotImplementedError()
    
    # Make sure all file names are the same
    for f in file_list:
        x = [Path(i).stem for i in f]
        assert len(set(x)) == 1 and x[0] != ''

    return file_list

