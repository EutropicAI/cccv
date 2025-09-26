import hashlib
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Union

from tenacity import retry, stop_after_attempt, stop_after_delay, wait_random
from torch.hub import download_url_to_file

from cccv.config import BaseConfig

if getattr(sys, "frozen", False):
    # frozen
    _IS_FROZEN_ = True
    CACHE_PATH = Path(sys.executable).parent.absolute() / "cache_models"
else:
    # unfrozen
    _IS_FROZEN_ = False
    CACHE_PATH = Path(__file__).resolve().parent.parent.absolute() / "cache_models"


CCCV_CACHE_MODEL_DIR = os.environ.get("CCCV_CACHE_MODEL_DIR", str(CACHE_PATH))

CCCV_REMOTE_MODEL_ZOO = os.environ.get(
    "CCCV_REMOTE_MODEL_ZOO", "https://github.com/EutropicAI/cccv/releases/download/model_zoo/"
)


def get_cache_dir(model_dir: Optional[Union[Path, str]] = None) -> Path:
    if model_dir is None or str(model_dir) == "":
        model_dir = str(CCCV_CACHE_MODEL_DIR)
        print(
            f"[CCCV] Using default cache model path {model_dir}, override it by setting environment variable CCCV_CACHE_MODEL_DIR"
        )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return Path(model_dir)


def get_file_sha256(file_path: Union[Path, str], blocksize: int = 1 << 20) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(blocksize)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def load_file_from_url(
    config: BaseConfig,
    force_download: bool = False,
    progress: bool = True,
    model_dir: Optional[Union[Path, str]] = None,
    gh_proxy: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """
    Load file form http url, will download models if necessary.

    Reference: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    :param config: The config object.
    :param force_download: Whether to force download the file.
    :param progress: Whether to show the download progress.
    :param model_dir: The path to save the downloaded model. Should be a full path. If None, use default cache path.
    :param gh_proxy: The proxy for downloading from github release. Example: https://github.abskoop.workers.dev/
    :return:
    """
    model_dir = get_cache_dir(model_dir)
    cached_file_path = model_dir / config.name

    if config.url is not None:
        _url: str = str(config.url)
    else:
        remote_zoo = CCCV_REMOTE_MODEL_ZOO
        print(
            f"[CCCV] Fetching models from {remote_zoo}, override it by setting environment variable CCCV_REMOTE_MODEL_ZOO"
        )
        if not remote_zoo.endswith("/"):
            remote_zoo += "/"
        _url = remote_zoo + config.name

    _gh_proxy = gh_proxy
    if _gh_proxy is not None and _url.startswith("https://github.com"):
        if not _gh_proxy.endswith("/"):
            _gh_proxy += "/"
        _url = _gh_proxy + _url

    if not cached_file_path.exists() or force_download:
        if _gh_proxy is not None:
            print(f"[CCCV] Using github proxy: {_gh_proxy}")
        print(f"[CCCV] Downloading: {_url} to {cached_file_path}\n")

        @retry(wait=wait_random(min=3, max=5), stop=stop_after_delay(10) | stop_after_attempt(30))
        def _download() -> None:
            try:
                download_url_to_file(url=_url, dst=str(cached_file_path), hash_prefix=None, progress=progress)
            except Exception as e:
                warnings.warn(f"[CCCV] Download failed: {e}, retrying...", stacklevel=2)
                raise e

        _download()

    if config.hash is not None:
        get_hash = get_file_sha256(cached_file_path)
        if get_hash != config.hash:
            raise ValueError(
                f"[CCCV] File {cached_file_path} hash mismatched with config hash {config.hash}, compare with {get_hash}"
            )

    return cached_file_path


def git_clone(git_url: str, model_dir: Optional[Union[Path, str]] = None, **kwargs: Any) -> Path:
    """
    Clone or update a git repository. We suggest use HuggingFace repo instead of GitHub repo for larger models.

    :param git_url: GitHub repository URL
    :param model_dir: Directory to clone into
    :param **kwargs: Additional git options (branch, commit_hash, etc.)
    :return: Path to the cloned repository
    """
    if not shutil.which("git"):
        warnings.warn(
            "[CCCV] git is not installed or not in the system's PATH. "
            "Please install git to use models from remote git repositories.",
            stacklevel=2,
        )

    model_dir = get_cache_dir(model_dir)
    # get repo name from url
    repo_name = git_url.split("/")[-1].replace(".git", "")
    clone_dir = model_dir / repo_name

    if clone_dir.exists() and (clone_dir / ".git").exists():
        print(f"[CCCV] Repository exists, updating: {clone_dir}")
        subprocess.run(["git", "-C", str(clone_dir), "pull"], check=True)

        # if branch or commit_hash is specified, checkout to that
        if "branch" in kwargs:
            subprocess.run(["git", "-C", str(clone_dir), "checkout", kwargs["branch"]], check=True)
        if "commit_hash" in kwargs:
            subprocess.run(["git", "-C", str(clone_dir), "reset", "--hard", kwargs["commit_hash"]], check=True)
    else:
        # clone the repo if not exists
        print(f"[CCCV] Cloning repository: {git_url} -> {clone_dir}")
        command = ["git", "clone", git_url, str(clone_dir)]

        if "branch" in kwargs:
            command.extend(["--branch", kwargs["branch"]])

        subprocess.run(command, check=True)

        if "commit_hash" in kwargs:
            subprocess.run(["git", "-C", str(clone_dir), "reset", "--hard", kwargs["commit_hash"]], check=True)

    return clone_dir


if __name__ == "__main__":
    # get all model files sha256
    for root, _, files in os.walk(get_cache_dir()):
        for file in files:
            if not file.endswith(".pth") and not file.endswith(".pt"):
                continue
            file_path = os.path.join(root, file)
            name = os.path.basename(file_path)
            print(f"[CCCV] {name}: {get_file_sha256(file_path)}")
