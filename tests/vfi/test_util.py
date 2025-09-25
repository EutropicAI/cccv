import cv2
import pytest
import torch
from ccvfi.util.color import rgb_to_yuv, yuv_to_rgb
from ccvfi.util.device import DEFAULT_DEVICE
from ccvfi.util.misc import (
    TMapper,
    check_scene,
    create_window_3d,
    de_resize,
    distance_calculator,
    gaussian,
    resize,
    ssim_matlab,
)
from torchvision import transforms

from .util import calculate_image_similarity, load_images


def test_device() -> None:
    print(DEFAULT_DEVICE)


def test_color() -> None:
    with pytest.raises(TypeError):
        rgb_to_yuv(1)
    with pytest.raises(TypeError):
        yuv_to_rgb(1)

    with pytest.raises(ValueError):
        rgb_to_yuv(torch.zeros(1, 1))
    with pytest.raises(ValueError):
        yuv_to_rgb(torch.zeros(1, 1))

    img = load_images()[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transforms.ToTensor()(img).unsqueeze(0).to("cpu")

    img = rgb_to_yuv(img)
    img = yuv_to_rgb(img)

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    assert calculate_image_similarity(img, load_images()[0])


def test_resize() -> None:
    img = torch.randn(1, 3, 64, 64)  # 创建一个随机的 4D 张量
    scale = 0.5
    resized_img = resize(img, scale)
    assert resized_img.shape[2] % 64 == 0  # 检查高度是否能被 64 整除
    assert resized_img.shape[3] % 64 == 0  # 检查宽度是否能被 64 整除


def test_de_resize() -> None:
    img = torch.randn(1, 3, 128, 128)
    ori_h, ori_w = 64, 64
    de_resized_img = de_resize(img, ori_h, ori_w)
    assert de_resized_img.shape[2] == ori_h
    assert de_resized_img.shape[3] == ori_w


def test_distance_calculator() -> None:
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # 创建一个 4D 张量
    distance = distance_calculator(x)
    expected_distance = torch.sqrt(torch.tensor([1.0**2 + 2.0**2, 3.0**2 + 4.0**2]))
    assert torch.allclose(distance, expected_distance)


def test_TMapper() -> None:
    mapper = TMapper(src=1.0, dst=2.0)
    timestamps = mapper.get_range_timestamps(0.0, 1.0, normalize=True)
    assert len(timestamps) > 0
    assert all(0.0 <= t <= 1.0 for t in timestamps)


def test_gaussian() -> None:
    window_size = 5
    sigma = 1.5
    gauss = gaussian(window_size, sigma)
    assert gauss.shape == (window_size,)
    assert torch.allclose(gauss.sum(), torch.tensor(1.0))


def test_create_window_3d() -> None:
    window_size = 5
    channel = 1
    window = create_window_3d(window_size, channel)
    assert window.shape == (1, channel, window_size, window_size, window_size)


def test_ssim_matlab() -> None:
    img1 = torch.randn(1, 3, 64, 64)
    img2 = torch.randn(1, 3, 64, 64)
    ssim_value = ssim_matlab(img1, img2)
    assert isinstance(ssim_value, torch.Tensor)
    assert 0.0 <= ssim_value.item() <= 1.0


class Test_Check_Scene:
    def test_5d(self) -> None:
        x1 = torch.randn(1, 1, 3, 64, 64)
        x2 = torch.randn(1, 1, 3, 64, 64)

        # 测试 enable_scdet 为 False 的情况
        result = check_scene(x1, x2, enable_scdet=False, scdet_threshold=0.5)
        assert result is False  # 当 enable_scdet 为 False 时，应返回 False

        # 测试 enable_scdet 为 True 的情况
        result = check_scene(x1, x2, enable_scdet=True, scdet_threshold=0.5)
        assert isinstance(result, bool)

    def test_4d(self) -> None:
        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)

        result = check_scene(x1, x2, enable_scdet=True, scdet_threshold=0.5)
        assert isinstance(result, bool)

    def test_3d(self) -> None:
        x1 = torch.randn(3, 64, 64)
        x2 = torch.randn(3, 64, 64)

        result = check_scene(x1, x2, enable_scdet=True, scdet_threshold=0.5)
        assert isinstance(result, bool)
