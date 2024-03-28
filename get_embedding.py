# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

# from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader
from torch import nn, is_tensor

from torchvision.transforms.functional import to_pil_image

from config_monuseg import Config
from dataloader.train_loader import FileLoader, MoNuSegDataset
from dataloader.utils import get_file_list

from models.hovernet.targets import gen_targets
from models.hovernet.opt import get_config


from PIL import Image
import numpy as np
import os
import re
from run_train import TrainManager

from tensorboard.plugins import projector
from tqdm import tqdm

def get_class(path):
    match = re.findall('gmm_\d/\d', path)
    if match is not None:
        cls = os.path.split(match[0])[-1]
        return cls
    
    return None

def resize_images(img):
    return [to_pil_image(i.permute(2, 0, 1)).resize((IMG_WIDTH, IMG_HEIGHT)) for i in img]


def get_images_labels_features(dataloader, model, preprocess):

    images = []
    labels = []
    features = []
    paths = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        # Step 3: Apply inference preprocessing transforms
        img = batch['img']
        path = batch['path']
        assert img.ndim == 4, "Missing batches of RGB"

        feature = extract_features(img.permute(0, 3, 1, 2), model, preprocess)

        # img = load_and_resize_image(img_path, IMG_WIDTH, IMG_HEIGHT)
        cls = [get_class(p) for p in path]

        images.extend(resize_images(img))
        labels.extend(cls)
        features.extend(feature)
        paths.extend(path)

    return images, labels, features, paths


def extract_features(img, model, preprocess):

    batch = preprocess(img)
    # batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    emb = model(batch).squeeze().detach().numpy()
 
    return emb

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

def write_embedding(log_dir, pil_images, features, labels):
    """Writes embedding data and projector configuration to the logdir."""
    metadata_filename = "metadata.tsv"
    tensor_filename = "features.tsv"
    npy_filename = "features.npy"
    sprite_image_filename = "sprite.jpg"
 
    img = pil_images[0]
    if is_tensor(img):
        pil_images = [to_pil_image(t.permute(2, 0, 1)) for t in pil_images]
 
    print("writing labels...")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, metadata_filename), "w") as f:
        f.write("{}\n".format('\n'.join([str(l) for l in labels])))
    print("writing embeddings...")
    np.save(os.path.join(log_dir, npy_filename), np.array(features))
    with open(os.path.join(log_dir, tensor_filename), "w") as f:
        for tensor in tqdm(features):
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))
 
    sprite_image_path = os.path.join(log_dir, sprite_image_filename)
 
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Label info.
    embedding.metadata_path = metadata_filename
    # Features info.
    embedding.tensor_path = tensor_filename
    # Image info.
    create_sprite_image(pil_images, sprite_image_path)
    embedding.sprite.image_path = sprite_image_filename
    # Specify the width and height of a single thumbnail.
    img_width, img_height = pil_images[0].size
    embedding.sprite.single_image_dim.extend([img_width, img_height])
    # Create the configuration file.
    projector.visualize_embeddings(log_dir, config)
     
    return



if __name__ == "__main__":

    IMG_WIDTH, IMG_HEIGHT = (64, 64)
    MODEL_NAME = "ResNet50"
    run_mode = "train"

    config = Config()

    dir_list = config.train_dir_list

    file_list = get_file_list(dir_list, config.file_type)

    print("Dataset %s: %d" % (run_mode, len(file_list)))
    input_dataset = MoNuSegDataset(
        file_list,
        file_type=config.file_type,
        mode=run_mode,
        with_type=config.type_classification,
        target_gen=(gen_targets, {}),
        **config.shape_info[run_mode]
    )

    dataloader = DataLoader(
        input_dataset,
        num_workers= 8,
        batch_size= 8,
        shuffle= run_mode == "train",
        drop_last= run_mode == "train",
    )


    # train_manager = TrainManager()
    # train_manager.nr_gpus = 1
    # phase_list = train_manager.model_config["phase_list"][0]
    # target_info = phase_list["target_info"]
    # mode = "train"
    # dataloader = train_manager._get_datagen(1, mode, target_info["gen"])

        

    # img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    if MODEL_NAME == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif MODEL_NAME == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        raise NotImplementedError()

    model_emb = nn.Sequential(*list(model.children())[:-1]) # strips off last linear layer

    model_emb.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    images, labels, features, paths = get_images_labels_features(dataloader, model_emb, preprocess)


    LOG_DIR = os.path.join('./logs_clustered', MODEL_NAME)
    write_embedding(LOG_DIR, images, features, labels)

