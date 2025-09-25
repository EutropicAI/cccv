from typing import Any, Union

from cccv.config import CONFIG_REGISTRY
from cccv.type import ConfigType


class AutoConfig:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: Union[ConfigType, str],
        **kwargs: Any,
    ) -> Any:
        """
        Get a config instance of a pretrained model configuration.

        :param pretrained_model_name_or_path: The name or path of the pretrained model configuration
        :return:
        """
        if "pretrained_model_name" in kwargs:
            print(
                "[CCCV] warning: 'pretrained_model_name' is deprecated, please use 'pretrained_model_name_or_path' instead."
            )
            pretrained_model_name_or_path = kwargs.pop("pretrained_model_name")

        return CONFIG_REGISTRY.get(pretrained_model_name_or_path)
