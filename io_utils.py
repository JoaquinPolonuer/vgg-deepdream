import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # visualizations
import yaml

from config import OUT_IMAGES_PATH, SupportedModels, SupportedPretrainedWeights


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["img_width"] = int(config["img_width"])
    config["pyramid_size"] = int(config["pyramid_size"])
    config["num_gradient_ascent_iterations"] = int(config["num_gradient_ascent_iterations"])
    config["spatial_shift_size"] = float(config["spatial_shift_size"])
    config["smoothing_coefficient"] = float(config["smoothing_coefficient"])
    config["lr"] = float(config["lr"])

    # Convert string enum names to actual enum values
    config["model_name"] = getattr(SupportedModels, config["model_name"]).name
    config["pretrained_weights"] = getattr(
        SupportedPretrainedWeights, config["pretrained_weights"]
    ).name

    # Add dump_dir and normalize input path
    config["dump_dir"] = os.path.join(
        OUT_IMAGES_PATH, f'{config["model_name"]}_{config["pretrained_weights"]}'
    )
    config["input"] = os.path.basename(config["input"])  # handle absolute and relative paths

    return config


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f"Path does not exist: {img_path}")
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if (
            isinstance(target_shape, int) and target_shape != -1
        ):  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # This need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


# config is just a shared dictionary that you'll be seeing used everywhere, but we'll define it a bit later.
# For the time being think of it as an oracle - whatever the function needs - config provides ^^
def save_and_maybe_display_image(config, dump_img, name_modifier=None):
    assert isinstance(dump_img, np.ndarray), f"Expected numpy array got {type(dump_img)}."

    # Step 1: figure out the dump dir location
    dump_dir = config["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)

    # Step 2: define the output image name
    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + ".jpg"
    else:
        dump_img_name = build_image_name(config)

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img * 255).astype(np.uint8)

    # Step 3: write image to the file system
    # ::-1 because opencv expects BGR (and not RGB) format...
    dump_path = os.path.join(dump_dir, dump_img_name)
    cv.imwrite(dump_path, dump_img[:, :, ::-1])

    # Step 4: potentially display/plot the image
    if config["should_display"]:
        fig = plt.figure(
            figsize=(7.5, 5), dpi=100
        )  # otherwise plots are really small in Jupyter Notebook
        plt.imshow(dump_img)
        plt.show()

    return dump_path


# This function makes sure we can later reconstruct the image using the information encoded into the filename!
# Again don't worry about all the arguments we'll define them later
def build_image_name(config):
    input_name = "rand_noise" if config["use_noise"] else config["input"].split(".")[0]
    layers = "_".join(config["layers_to_use"])
    # Looks awful but makes the creation process transparent for other creators
    img_name = f'{input_name}_width_{config["img_width"]}_model_{config["model_name"]}_{config["pretrained_weights"]}_{layers}_pyrsize_{config["pyramid_size"]}_pyrratio_{config["pyramid_ratio"]}_iter_{config["num_gradient_ascent_iterations"]}_lr_{config["lr"]}_shift_{config["spatial_shift_size"]}_smooth_{config["smoothing_coefficient"]}.jpg'
    return img_name
