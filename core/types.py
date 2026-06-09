from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class AdapterType(str, enum.Enum):
    """支持的图像生成适配器类型。"""

    GEMINI = "gemini"
    OPENAI_CHAT = "openai_chat"
    OPENAI = "openai"
    SILICONFLOW = "siliconflow_adapter"
    VOLCENGINE_ARK = "volcengine_ark"
    GITEE_AI = "gitee_ai"
    AGNES_AI = "agnes_ai"
    JIMENG2API = "jimeng2api"
    GROK = "grok"
    CUSTOM_HTTP = "custom_http"


class ImageCapability(enum.Flag):
    """图像生成适配器支持的功能。"""

    NONE = 0
    TEXT_TO_IMAGE = enum.auto()  # 文生图
    IMAGE_TO_IMAGE = enum.auto()  # 图生图
    RESOLUTION = enum.auto()  # 指定分辨率
    ASPECT_RATIO = enum.auto()  # 指定宽高比


@dataclass
class AdapterMetadata:
    """关于适配器能力的元数据。"""

    name: str
    capabilities: ImageCapability = ImageCapability.TEXT_TO_IMAGE


@dataclass
class AdapterConfig:
    """构造适配器所需的配置。"""

    type: AdapterType = AdapterType.GEMINI
    name: str = ""  # 供应商展示名称
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 3
    debug_request_logging: bool = False
    non_retryable_status_codes: list[int] = field(default_factory=list)
    non_retryable_error_keywords: list[str] = field(default_factory=list)
    safety_settings: str | None = None
    capability_options: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # 适配器特有配置


@dataclass
class ImageData:
    """带有 MIME 类型和可选来源 URL 的图像数据。"""

    data: bytes
    mime_type: str
    source_url: str | None = None


@dataclass
class GenerationRequest:
    """用户生图请求。"""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None
    batch_index: int = 1
    batch_count: int = 1
    retry_status_callback: Callable[[int, int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass
class GenerationResult:
    """生图尝试的结果。"""

    images: list[bytes] | None = None
    error: str | None = None
