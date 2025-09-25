import math
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from cccv.util.device import DEFAULT_DEVICE

print(f"PyTorch version: {torch.__version__}")
torch_2_4: bool = torch.__version__.startswith("2.4")

ASSETS_PATH = Path(__file__).resolve().parent.parent.absolute() / "assets"

# normal test image
TEST_IMG_PATH = ASSETS_PATH / "test.jpg"

# vfi test image
TEST_IMG_PATH_0 = ASSETS_PATH / "vfi" / "test_i0.jpg"
TEST_IMG_PATH_1 = ASSETS_PATH / "vfi" / "test_i1.jpg"
TEST_IMG_PATH_2 = ASSETS_PATH / "vfi" / "test_i2.jpg"

EVAL_IMG_PATH_RIFE = ASSETS_PATH / "vfi" / "test_out_rife.jpg"

EVAL_IMG_PATH_DRBA_0 = ASSETS_PATH / "vfi" / "test_out_drba_0.jpg"
EVAL_IMG_PATH_DRBA_1 = ASSETS_PATH / "vfi" / "test_out_drba_1.jpg"
EVAL_IMG_PATH_DRBA_2 = ASSETS_PATH / "vfi" / "test_out_drba_2.jpg"
EVAL_IMG_PATH_DRBA_3 = ASSETS_PATH / "vfi" / "test_out_drba_3.jpg"
EVAL_IMG_PATH_DRBA_4 = ASSETS_PATH / "vfi" / "test_out_drba_4.jpg"

CI_ENV = os.environ.get("GITHUB_ACTIONS") == "true"
CCCV_FP16 = True if not CI_ENV else False
CCCV_TILE = None if not CI_ENV else (64, 64)
CCCV_DEVICE = DEFAULT_DEVICE if not CI_ENV else torch.device("cpu")


# load normal test image
def load_image(img_path: Path = TEST_IMG_PATH) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


# load vfi test images
def load_images() -> List[np.ndarray]:
    return [load_image(k) for k in [TEST_IMG_PATH_0, TEST_IMG_PATH_1, TEST_IMG_PATH_2]]


def load_eval_images() -> List[np.ndarray]:
    return [
        load_image(k)
        for k in [
            EVAL_IMG_PATH_DRBA_0,
            EVAL_IMG_PATH_DRBA_1,
            EVAL_IMG_PATH_DRBA_2,
            EVAL_IMG_PATH_DRBA_3,
            EVAL_IMG_PATH_DRBA_4,
        ]
    ]


def load_eval_image() -> np.ndarray:
    return load_image(EVAL_IMG_PATH_RIFE)


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray, similarity: float = 0.85) -> bool:
    """
    calculate image similarity, check SR is correct

    :param image1: original image
    :param image2: upscale image
    :param similarity: similarity threshold
    :return:
    """
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, _) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return score > similarity


def compare_image_size(image1: np.ndarray, image2: np.ndarray, scale: int) -> bool:
    """
    compare original image size and upscale image size, check targetscale is correct

    :param image1: original image
    :param image2: upscale image
    :param scale: upscale ratio
    :return:
    """
    target_size = (math.ceil(image1.shape[0] * scale), math.ceil(image1.shape[1] * scale))

    return image2.shape[0] == target_size[0] and image2.shape[1] == target_size[1]
