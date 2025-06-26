import os

from deepdream_core import deep_dream_static_image
from io_utils import save_and_maybe_display_image

img = deep_dream_static_image()  # yep a single liner

dump_path = save_and_maybe_display_image(img, display=True)
# print(f"Saved DeepDream static image to: {os.path.relpath(dump_path)}\n")
