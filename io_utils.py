import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  # visualizations
import yaml

from config import RUN_CONFIG

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


def save_and_maybe_display_image(dump_img, display=False):
    assert isinstance(dump_img, np.ndarray), f"Expected numpy array got {type(dump_img)}."

    # Step 1: figure out the dump dir location
    dump_dir = RUN_CONFIG["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)

    # Step 2: define the output image name
    dump_img_name = build_image_name()

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img * 255).astype(np.uint8)

    # Step 3: write image to the file system
    # ::-1 because opencv expects BGR (and not RGB) format...
    dump_path = os.path.join(dump_dir, dump_img_name)
    cv.imwrite(dump_path, dump_img[:, :, ::-1])

    # Step 4: potentially display/plot the image
    if display:
        fig = plt.figure(
            figsize=(7.5, 5), dpi=100
        )  # otherwise plots are really small in Jupyter Notebook
        plt.imshow(dump_img)
        plt.show()

    return dump_path


# This function makes sure we can later reconstruct the image using the information encoded into the filename!
# Again don't worry about all the arguments we'll define them later
def build_image_name():
    input_name = "rand_noise" if RUN_CONFIG["use_noise"] else RUN_CONFIG["input"].split(".")[0]
    layers = "_".join(RUN_CONFIG["layers_to_use"])
    # Looks awful but makes the creation process transparent for other creators
    img_name = f'{input_name}_width_{RUN_CONFIG["img_width"]}_model_{RUN_CONFIG["model_name"]}_{RUN_CONFIG["pretrained_weights"]}_{layers}_pyrsize_{RUN_CONFIG["pyramid_size"]}_pyrratio_{RUN_CONFIG["pyramid_ratio"]}_iter_{RUN_CONFIG["num_gradient_ascent_iterations"]}_lr_{RUN_CONFIG["lr"]}_shift_{RUN_CONFIG["spatial_shift_size"]}_smooth_{RUN_CONFIG["smoothing_coefficient"]}_seed_{RUN_CONFIG["seed"]}.jpg'
    return img_name
