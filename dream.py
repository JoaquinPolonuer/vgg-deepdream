import os

from config import OUT_IMAGES_PATH

from deepdream_core import deep_dream_static_image
from io_utils import save_and_maybe_display_image
from io_utils import load_config

# Load configuration from YAML file
config = load_config()

img = deep_dream_static_image(config)  # yep a single liner




config["should_display"] = True
dump_path = save_and_maybe_display_image(config, img)
# print(f"Saved DeepDream static image to: {os.path.relpath(dump_path)}\n")
