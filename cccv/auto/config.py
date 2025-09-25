from typing import Any, Optional, Union

from cccv.config import CONFIG_REGISTRY, BaseConfig
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

    @staticmethod
    def register(config: Union[BaseConfig, Any], name: Optional[str] = None) -> None:
        """
        Register the given config class instance under the name BaseConfig.name or the given name.
        Can be used as a function call. See docstring of this class for usage.

        :param config: The config class instance to register
        :param name: The name to register the config class instance under. If None, use BaseConfig.name
        :return:
        """
        # used as a function call
        CONFIG_REGISTRY.register(obj=config, name=name)
