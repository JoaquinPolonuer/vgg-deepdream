import os
import enum
import torch
import numpy as np
import random
import yaml


# The 2 datasets we'll be leveraging
class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1


# The 2 models we'll be using
class SupportedModels(enum.Enum):
    VGG16_EXPERIMENTAL = (0,)
    RESNET50 = 1


# Commonly used paths, let's define them here as constants
DATA_DIR_PATH = os.path.join(os.getcwd(), "data")
INPUT_DATA_PATH = os.path.join(DATA_DIR_PATH, "input")
BINARIES_PATH = os.path.join(os.getcwd(), "models", "binaries")
OUT_IMAGES_PATH = os.path.join(DATA_DIR_PATH, "out-images")

# Make sure these exist as the rest of the code relies on it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(OUT_IMAGES_PATH, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # checking whether you have a GPU

# Images will be normalized using these, because the CNNs were trained with normalized images as well!
IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# RUN RUN_CONFIG
with open("config.yaml", "r") as file:
    RUN_CONFIG = yaml.safe_load(file)

RUN_CONFIG["seed"] = int(RUN_CONFIG["seed"])
RUN_CONFIG["img_width"] = int(RUN_CONFIG["img_width"])
RUN_CONFIG["pyramid_size"] = int(RUN_CONFIG["pyramid_size"])
RUN_CONFIG["num_gradient_ascent_iterations"] = int(RUN_CONFIG["num_gradient_ascent_iterations"])
RUN_CONFIG["spatial_shift_size"] = float(RUN_CONFIG["spatial_shift_size"])
RUN_CONFIG["smoothing_coefficient"] = float(RUN_CONFIG["smoothing_coefficient"])
RUN_CONFIG["lr"] = float(RUN_CONFIG["lr"])

# Convert string enum names to actual enum values
RUN_CONFIG["model_name"] = getattr(SupportedModels, RUN_CONFIG["model_name"]).name
RUN_CONFIG["pretrained_weights"] = getattr(
    SupportedPretrainedWeights, RUN_CONFIG["pretrained_weights"]
).name

# Add dump_dir and normalize input path
RUN_CONFIG["dump_dir"] = os.path.join(
    OUT_IMAGES_PATH, f'{RUN_CONFIG["model_name"]}_{RUN_CONFIG["pretrained_weights"]}'
)
RUN_CONFIG["input"] = os.path.basename(RUN_CONFIG["input"])  # handle absolute and relative paths

# SEED
seed = RUN_CONFIG["seed"]

random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

# Additional PyTorch settings for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For MPS (Apple Silicon) reproducibility
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
