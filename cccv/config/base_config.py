from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, FilePath, HttpUrl

from cccv.type.arch import ArchType
from cccv.type.model import ModelType


class BaseConfig(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    path: Optional[FilePath] = None
    hash: Optional[str] = None
    arch: Union[ArchType, str]
    model: Union[ModelType, str]


class AutoBaseConfig(BaseConfig):
    model_config = ConfigDict(extra="allow")


class AuxiliaryBaseConfig(BaseConfig):
    pass


class SRBaseConfig(BaseConfig):
    scale: int


class VSRBaseConfig(SRBaseConfig):
    num_frame: int


class VFIBaseConfig(BaseConfig):
    num_frame: int
