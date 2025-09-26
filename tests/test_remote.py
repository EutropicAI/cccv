import pytest

from cccv import CONFIG_REGISTRY, ConfigType
from cccv.util.remote import get_cache_dir, git_clone, load_file_from_url
from tests.util import CI_ENV


def test_cache_models() -> None:
    load_file_from_url(CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x))


@pytest.mark.skipif(CI_ENV, reason="Skip test on CI environment to save the provider's bandwidth")
def test_cache_models_with_gh_proxy() -> None:
    load_file_from_url(
        config=CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x),
        force_download=True,
        gh_proxy="https://github.abskoop.workers.dev/",
    )
    load_file_from_url(
        config=CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x),
        force_download=True,
        gh_proxy="https://github.abskoop.workers.dev",
    )


def test_git_clone() -> None:
    clone_dir = git_clone("https://github.com/EutropicAI/cccv_demo_remote_model")
    assert clone_dir == get_cache_dir() / "cccv_demo_remote_model"
