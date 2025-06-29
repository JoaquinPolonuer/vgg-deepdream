# I always like to structure my imports into Python's native libs,
# stuff I installed via conda/pip and local file imports (but we don't have those here)

# Python native libs
import os

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from torchvision import models
from config import DEVICE, IMAGENET_MEAN_1, IMAGENET_STD_1, INPUT_DATA_PATH, RUN_CONFIG
from vgg_experimental import Vgg16Experimental
from smoothing import CascadeGaussianSmoothing
from io_utils import load_image
from processing_utils import (
    pre_process_numpy_img,
    post_process_numpy_img,
    pytorch_input_adapter,
    pytorch_output_adapter,
    get_new_shape,
)


def deep_dream_static_image(img=None):
    model = Vgg16Experimental(
        RUN_CONFIG["pretrained_weights"], requires_grad=False, show_progress=True
    ).to(DEVICE)
    if RUN_CONFIG["layers_to_use"].get("linear3"):
        linear3_indices = RUN_CONFIG["layers_to_use"]["linear3"]

        # You can see the list of classes here: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
        # NOTE: Esta un poco hardcodeado esto, porque nada asegura que "seleccionemos" ese modelo
        classes_to_maximize = [
            models.VGG16_Weights.IMAGENET1K_V1.meta["categories"][i] for i in linear3_indices
        ]
        classes_to_maximize = [f"{class_name}s" for class_name in classes_to_maximize]
        print(f"Dreaming about {', '.join(classes_to_maximize)}")

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = os.path.join(INPUT_DATA_PATH, RUN_CONFIG["input"])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = load_image(img_path, target_shape=RUN_CONFIG["img_width"])
        if RUN_CONFIG["use_noise"]:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = pre_process_numpy_img(img)
    original_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(RUN_CONFIG["pyramid_size"]):
        new_shape = get_new_shape(RUN_CONFIG, original_shape, pyramid_level)
        img = cv.resize(
            img, (new_shape[1], new_shape[0])
        )  # resize depending on the current pyramid level
        input_tensor = pytorch_input_adapter(img)  # convert to trainable tensor

        for iteration in range(RUN_CONFIG["num_gradient_ascent_iterations"]):
            # This is where the magic happens, treat it as a black box until the next cell
            gradient_ascent(model, input_tensor, iteration)

        img = pytorch_output_adapter(input_tensor)

    return post_process_numpy_img(img)


LOWER_IMAGE_BOUND = torch.tensor((-IMAGENET_MEAN_1 / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(
    DEVICE
)
UPPER_IMAGE_BOUND = torch.tensor(((1 - IMAGENET_MEAN_1) / IMAGENET_STD_1).reshape(1, -1, 1, 1)).to(
    DEVICE
)


def calculate_loss(model_input, model_output):

    # Step 1: Grab activations/feature maps of interest
    activations = []
    for layer_name, neuron_indices in RUN_CONFIG["layers_to_use"].items():
        layer_activation = model_output[layer_name]
        # If specific neurons are specified, select only those
        if neuron_indices is not None:
            if layer_activation.dim() == 4:  # Convolutional layer (B, C, H, W)
                layer_activation = layer_activation[:, neuron_indices, :, :]
            elif layer_activation.dim() == 2:  # Fully connected layer (B, N)
                layer_activation = layer_activation[:, neuron_indices]
            else:
                print(
                    f"Warning: Unsupported activation shape for layer {layer_name}: {layer_activation.shape}"
                )
        activations.append(layer_activation)
    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction="mean")(
            layer_activation, torch.zeros_like(layer_activation)
        )
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))

    return loss


def gradient_ascent(model, input_tensor, iteration):
    # Step 0: Feed forward pass
    model_output = model(input_tensor)

    loss = calculate_loss(input_tensor, model_output)

    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization, more an art then a science)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # We'll see the details of this one in the next cell and that's all, you now understand DeepDream!
    sigma = ((iteration + 1) / RUN_CONFIG["num_gradient_ascent_iterations"]) * 2.0 + RUN_CONFIG[
        "smoothing_coefficient"
    ]
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(
        grad
    )  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += RUN_CONFIG["lr"] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)
