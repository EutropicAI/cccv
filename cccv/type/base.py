from abc import ABC, abstractmethod
from typing import Any


class BaseModelInterface(ABC):
    @abstractmethod
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def inference_video(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
