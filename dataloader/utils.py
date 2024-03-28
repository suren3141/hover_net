import glob, os, sys


def get_file_list(data_dir_list, file_type, ):

    file_list = []

    if file_type == '.npy':
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))

        file_list.sort()  # to always ensure same input ordering

    elif file_type == '.png':
        for dir_path in data_dir_list:
            image_files =  glob.glob(os.path.join(dir_path, 'images', '*.png'))
            ann_files =  glob.glob(os.path.join(dir_path, 'bin_masks', '*.png'))

            file_list.extend(list(zip(image_files, ann_files)))
            # file_list.extend(image_files)

        file_list = sorted(file_list, key=lambda x:x[0])       
        # file_list.sort()  # to always ensure same input ordering
    
    else:
        raise NotImplementedError()

    return file_list

