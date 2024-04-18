import glob, os, sys


def get_file_list(data_dir_list, file_type, img_path="images", ann_path='bin_masks'):
    """
    """

    if isinstance(data_dir_list, str): data_dir_list = [data_dir_list]

    file_list = []

    if file_type == '.npy':
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))

        file_list.sort()  # to always ensure same input ordering

    elif file_type == '.png':
        for dir_path in data_dir_list:
            image_files =  glob.glob(os.path.join(dir_path, img_path, '*.png'))
            ann_files =  glob.glob(os.path.join(dir_path, ann_path, '*.png'))

            file_list.extend(list(zip(image_files, ann_files)))
            # file_list.extend(image_files)

        file_list = sorted(file_list, key=lambda x:x[0])       
        # file_list.sort()  # to always ensure same input ordering
    
    else:
        raise NotImplementedError()

    return file_list

