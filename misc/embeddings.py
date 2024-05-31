import os
from tensorboard.plugins import projector
import numpy as np
from torch import nn, is_tensor
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

def overwrite_embedding_classes(log_dir, labels):
    metadata_filename = "metadata.tsv"

    print("writing labels...")
    with open(os.path.join(log_dir, metadata_filename), "w") as f:
        f.write("{}\n".format('\n'.join([str(l) for l in labels])))



def create_sprite_image(pil_images, save_path):
    # Assuming all images have the same width and height
    img_width, img_height = pil_images[0].size
 
    # create a master square images
    row_coln_count = int(np.ceil(np.sqrt(len(pil_images))))
    master_img_width = img_width * row_coln_count
    master_img_height = img_height * row_coln_count
 
    master_image = Image.new(
        mode = 'RGBA',
        size = (master_img_width, master_img_height),
        color = (0, 0, 0, 0)
    )
 
    for i, img in tqdm(enumerate(pil_images)):
        div, mod = divmod(i, row_coln_count)
        w_loc = img_width * mod
        h_loc = img_height * div
        master_image.paste(img, (w_loc, h_loc))
 
    master_image.convert('RGB').save(save_path, transparency=0)
    return

def write_embedding(log_dir, pil_images, features, labels, paths=None):
    """Writes embedding data and projector configuration to the logdir."""
    metadata_filename = "metadata.tsv"
    path_filename = "paths.tsv"
    tensor_filename = "features.tsv"
    npy_filename = "features.npy"
    sprite_image_filename = "sprite.jpg"

    os.makedirs(log_dir, exist_ok=True)
 
    print("writing labels...")
    with open(os.path.join(log_dir, metadata_filename), "w") as f:
        f.write("{}\n".format('\n'.join([str(l) for l in labels])))

    if paths is not None and len(paths) == 2:
        print("writing paths...")
        with open(os.path.join(log_dir, path_filename), "w") as f:
            img_files, ann_files = paths
            for i, a in zip(img_files, ann_files):
                f.write(f"{i}\t{a}\n")

    print("writing embeddings...")
    np.save(os.path.join(log_dir, npy_filename), np.array(features))
    with open(os.path.join(log_dir, tensor_filename), "w") as f:
        for tensor in tqdm(features):
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))

    print("writing images...")
    sprite_image_path = os.path.join(log_dir, sprite_image_filename)
    if pil_images is None:
        if os.path.exists(sprite_image_path):
            img_width, img_height = 64, 64
        else:
            sprite_image_filename = None
    else:
        if is_tensor(pil_images[0]):
            pil_images = [to_pil_image(t.permute(2, 0, 1)) for t in pil_images]
        create_sprite_image(pil_images, sprite_image_path)
        # Specify the width and height of a single thumbnail.
        img_width, img_height = pil_images[0].size
 
 
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Label info.
    embedding.metadata_path = metadata_filename
    # Features info.
    embedding.tensor_path = tensor_filename
    # Image info.
    if sprite_image_filename is not None:
        embedding.sprite.image_path = sprite_image_filename
        embedding.sprite.single_image_dim.extend([img_width, img_height])
    # Create the configuration file.
    projector.visualize_embeddings(log_dir, config)
     
    return


