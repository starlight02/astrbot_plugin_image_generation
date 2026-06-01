from __future__ import annotations

from astrbot.api import logger

from ..adapter import (
    AgnesAIAdapter,
    GeminiAdapter,
    GeminiOpenAIAdapter,
    GiteeAIAdapter,
    GrokAdapter,
    Jimeng2APIAdapter,
    OpenAIAdapter,
    SiliconFlowAdapter,
    VolcengineArkAdapter,
)
from .types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageData,
)
from .logging_utils import format_cn_log_fields, log_prefix, safe_log_text
from .utils import convert_images_batch


class ImageGenerator:
    """适配器编排器，负责分发生图请求。"""

    def __init__(self, adapter_config: AdapterConfig):
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    def _create_adapter(self, config: AdapterConfig):
        """根据配置创建对应的适配器。"""
        adapter_map: dict[AdapterType, type] = {
            AdapterType.GEMINI: GeminiAdapter,
            AdapterType.GEMINI_OPENAI: GeminiOpenAIAdapter,
            AdapterType.OPENAI: OpenAIAdapter,
            AdapterType.SILICONFLOW: SiliconFlowAdapter,
            AdapterType.VOLCENGINE_ARK: VolcengineArkAdapter,
            AdapterType.GITEE_AI: GiteeAIAdapter,
            AdapterType.AGNES_AI: AgnesAIAdapter,
            AdapterType.JIMENG2API: Jimeng2APIAdapter,
            AdapterType.GROK: GrokAdapter,
        }

        adapter_cls = adapter_map.get(config.type)
        if not adapter_cls:
            raise ValueError(f"不支持的适配器类型: {config.type}")
        return adapter_cls(config)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行生图逻辑。"""
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        converted_images = await self.convert_reference_images(request.images)
        return await self.generate_preconverted(request, images=converted_images)

    async def convert_reference_images(
        self, images: list[ImageData]
    ) -> list[ImageData]:
        """Convert reference images into adapter-compatible formats once."""
        if not images:
            return []
        return await convert_images_batch(images)

    def build_generation_request(
        self,
        request: GenerationRequest,
        *,
        images: list[ImageData],
    ) -> GenerationRequest:
        """Build a request with already converted reference images."""
        return GenerationRequest(
            prompt=request.prompt,
            images=images,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            task_id=request.task_id,
            batch_index=request.batch_index,
            batch_count=request.batch_count,
            retry_status_callback=request.retry_status_callback,
        )

    async def generate_preconverted(
        self,
        request: GenerationRequest,
        *,
        images: list[ImageData],
    ) -> GenerationResult:
        """Generate an image request with preconverted reference images."""
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        patched_request = self.build_generation_request(request, images=images)
        logger.debug(
            f"{log_prefix('Generator', request.task_id)} 分发生图请求: "
            + format_cn_log_fields(
                适配器=self.adapter.__class__.__name__,
                模型=self.adapter.model,
                进度=f"{request.batch_index}/{request.batch_count}",
                参考图=f"{len(images)}张",
                宽高比=request.aspect_ratio,
                分辨率=request.resolution,
            )
        )

        try:
            return await self.adapter.generate(patched_request)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{log_prefix('Generator', request.task_id)} 生成失败: {safe_log_text(exc, 200)}",
                exc_info=True,
            )
            return GenerationResult(images=None, error=str(exc))

    def update_model(self, model: str) -> None:
        """更新适配器使用的模型。"""
        if self.adapter:
            self.adapter.update_model(model)

    async def update_adapter(self, adapter_config: AdapterConfig) -> None:
        """更新适配器配置并重新创建适配器。

        注意: 此方法会关闭旧适配器以释放资源。
        """
        if self.adapter:
            await self.adapter.close()
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    async def close(self) -> None:
        """关闭适配器。"""
        if self.adapter:
            await self.adapter.close()
