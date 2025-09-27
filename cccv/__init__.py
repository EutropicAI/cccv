"""
MIT License

Copyright (c) 2025 Eutropic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "0.0.4"

from cccv.arch import ARCH_REGISTRY
from cccv.auto import AutoConfig, AutoModel
from cccv.config import CONFIG_REGISTRY, AutoBaseConfig, BaseConfig, SRBaseConfig, VFIBaseConfig, VSRBaseConfig
from cccv.model import MODEL_REGISTRY, AuxiliaryBaseModel, CCBaseModel, SRBaseModel, VFIBaseModel, VSRBaseModel
from cccv.type import ArchType, BaseModelInterface, ConfigType, ModelType
