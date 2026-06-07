"""
插件配置管理模块
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig

from .constants import (
    ALL_LLM_TOOLS,
    ALL_RESULT_INFO_ITEMS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
    DEFAULT_DAILY_LIMIT_COUNT,
    DEFAULT_GENERATION_IMAGE_COUNT,
    DEFAULT_IMAGE_AUDIT_PROMPT,
    DEFAULT_MAX_GENERATION_IMAGE_COUNT,
    DEFAULT_MAX_IMAGES_PER_MESSAGE,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MAX_IMAGE_SIZE_MB,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS,
    DEFAULT_NON_RETRYABLE_STATUS_CODES,
    DEFAULT_PROMPT_AUDIT_PROMPT,
    DEFAULT_RESULT_INFO_ITEMS,
    DEFAULT_RATE_LIMIT_SECONDS,
    DEFAULT_RESOLUTION,
    DEFAULT_TIMEOUT,
    LLM_TOOL_IMAGE_GENERATION,
    LLM_TOOL_PRESET_EDIT,
    LLM_TOOL_PRESET_QUERY,
    LLM_TOOL_TASK_MANAGEMENT,
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_TASK_ID,
    RESULT_INFO_USAGE,
)
from .config_validator import ConfigValidator
from .logging_utils import log_prefix, safe_log_text
from .types import AdapterConfig, AdapterType


__all__ = (
    "ConfigManager",
    "GenerationSettings",
    "ImageAuditSettings",
    "LLM_TOOL_IMAGE_GENERATION",
    "LLM_TOOL_PRESET_EDIT",
    "LLM_TOOL_PRESET_QUERY",
    "LLM_TOOL_TASK_MANAGEMENT",
    "PersonaTemplate",
    "PluginConfig",
    "PromptAuditSettings",
    "RESULT_INFO_COUNT",
    "RESULT_INFO_DURATION",
    "RESULT_INFO_MODEL",
    "RESULT_INFO_TASK_ID",
    "RESULT_INFO_USAGE",
    "SafetyAuditSettings",
    "UsageSettings",
)


PROVIDER_COMMON_FIELDS = frozenset(
    {
        "__template_key",
        "name",
        "base_url",
        "proxy",
        "api_keys",
        "available_models",
        "capability_options",
        "timeout",
        "max_retry_attempts",
    }
)

ADAPTER_EXTRA_DEFAULTS: dict[AdapterType, dict[str, Any]] = {
    AdapterType.OPENAI_CHAT: {
        "prompt_prefix": "Generate an image: ",
        "modalities": ["image", "text"],
    },
    AdapterType.OPENAI: {"model_family": "auto"},
}
LOG = log_prefix("Config")


@dataclass
class UsageSettings:
    """用户使用限制设置。"""

    rate_limit_seconds: int = DEFAULT_RATE_LIMIT_SECONDS
    enable_daily_limit: bool = False
    daily_limit_count: int = DEFAULT_DAILY_LIMIT_COUNT
    max_image_size_mb: int = DEFAULT_MAX_IMAGE_SIZE_MB
    umo_blacklist: list[str] = field(default_factory=list)
    admin_bypass_limits: bool = True
    umo_whitelist: list[str] = field(default_factory=list)
    blacklist_block_message: str = "❌ 当前会话已被加入黑名单，无法使用生图功能"


@dataclass
class GenerationSettings:
    """生成设置。"""

    default_aspect_ratio: str = DEFAULT_ASPECT_RATIO
    default_resolution: str = DEFAULT_RESOLUTION
    default_image_count: int = DEFAULT_GENERATION_IMAGE_COUNT
    max_image_count: int = DEFAULT_MAX_GENERATION_IMAGE_COUNT
    max_images_per_message: int = DEFAULT_MAX_IMAGES_PER_MESSAGE
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    non_retryable_status_codes: list[int] = field(
        default_factory=lambda: list(DEFAULT_NON_RETRYABLE_STATUS_CODES)
    )
    non_retryable_error_keywords: list[str] = field(
        default_factory=lambda: list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS)
    )
    result_info_items: set[str] = field(
        default_factory=lambda: set(DEFAULT_RESULT_INFO_ITEMS)
    )
    start_task_message_template: str = "已开始生图任务{reference_images_block}{preset_block}{persona_block}{image_count_block} [任务ID: {task_id}]"


@dataclass
class PersonaTemplate:
    """生图人设模板。"""

    name: str
    prompt: str
    image: str = ""


@dataclass
class PromptAuditSettings:
    """生图前提示词审核设置。"""

    blocked_words: list[str] = field(default_factory=list)
    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    max_retry_attempts: int = DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS
    ai_prompt: str = DEFAULT_PROMPT_AUDIT_PROMPT


@dataclass
class ImageAuditSettings:
    """生图后图片审核设置。"""

    enable_ai_audit: bool = False
    ai_provider_id: str = ""
    max_retry_attempts: int = DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS
    ai_prompt: str = DEFAULT_IMAGE_AUDIT_PROMPT


@dataclass
class SafetyAuditSettings:
    """安全审核总设置。"""

    umo_whitelist: list[str] = field(default_factory=list)
    prompt_audit: PromptAuditSettings = field(default_factory=PromptAuditSettings)
    image_audit: ImageAuditSettings = field(default_factory=ImageAuditSettings)


@dataclass
class PluginConfig:
    """完整的插件配置。"""

    adapter_config: AdapterConfig | None = None
    usage_settings: UsageSettings = field(default_factory=UsageSettings)
    generation_settings: GenerationSettings = field(default_factory=GenerationSettings)
    safety_audit_settings: SafetyAuditSettings = field(
        default_factory=SafetyAuditSettings
    )
    presets: dict[str, Any] = field(default_factory=dict)
    personas: dict[str, PersonaTemplate] = field(default_factory=dict)
    enabled_llm_tools: set[str] = field(default_factory=lambda: set(ALL_LLM_TOOLS))


class ConfigManager:
    """插件配置管理器。"""

    def __init__(self, config: AstrBotConfig):
        self._config = config
        self._config_validator = ConfigValidator(getattr(config, "schema", None))
        self._plugin_config: PluginConfig = PluginConfig()
        self._all_provider_configs: list[AdapterConfig] = []  # 保存所有供应商配置
        self.load()

    def load(self) -> PluginConfig:
        """加载并解析插件配置。"""
        self._validate_config_values()

        gen_cfg = self._get_config_section("generation")
        user_limits_cfg = self._get_config_section("user_limits")
        safety_cfg = self._get_config_section("safety_audit")
        prompt_templates_cfg = self._get_config_section("prompt_templates")
        api_providers_raw = self._config.get("api_providers", [])

        all_provider_configs = self._load_provider_configs(api_providers_raw, gen_cfg)
        self._all_provider_configs = all_provider_configs

        self._plugin_config = PluginConfig(
            adapter_config=self._select_adapter_config(
                all_provider_configs,
                self._get_str(gen_cfg, "model", ""),
            ),
            usage_settings=self._parse_usage_settings(user_limits_cfg),
            generation_settings=self._parse_generation_settings(gen_cfg),
            safety_audit_settings=self._parse_safety_audit_settings(safety_cfg),
            presets=self._load_presets(prompt_templates_cfg.get("presets", [])),
            personas=self._load_personas(prompt_templates_cfg.get("personas", [])),
            enabled_llm_tools=set(
                self._parse_enabled_llm_tools(
                    self._config.get("enable_llm_tool", list(ALL_LLM_TOOLS))
                )
            ),
        )

        return self._plugin_config

    def _parse_usage_settings(self, cfg: dict[str, Any]) -> UsageSettings:
        """Parse user limit settings from normalized config."""
        return UsageSettings(
            rate_limit_seconds=self._get_int(
                cfg,
                "rate_limit_seconds",
                DEFAULT_RATE_LIMIT_SECONDS,
                min_value=0,
            ),
            enable_daily_limit=self._get_bool(cfg, "enable_daily_limit", False),
            daily_limit_count=self._get_int(
                cfg,
                "daily_limit_count",
                DEFAULT_DAILY_LIMIT_COUNT,
                min_value=1,
            ),
            max_image_size_mb=self._get_int(
                cfg,
                "max_image_size_mb",
                DEFAULT_MAX_IMAGE_SIZE_MB,
                min_value=1,
            ),
            umo_blacklist=self._parse_string_list(cfg.get("umo_blacklist", [])),
            admin_bypass_limits=self._get_bool(cfg, "admin_bypass_limits", True),
            umo_whitelist=self._parse_string_list(cfg.get("umo_whitelist", [])),
            blacklist_block_message=self._get_str(
                cfg,
                "blacklist_block_message",
                UsageSettings.blacklist_block_message,
            ),
        )

    def _parse_generation_settings(self, cfg: dict[str, Any]) -> GenerationSettings:
        """Parse image generation behavior settings."""
        return GenerationSettings(
            default_aspect_ratio=self._get_str(
                cfg,
                "default_aspect_ratio",
                DEFAULT_ASPECT_RATIO,
            ),
            default_resolution=self._get_str(
                cfg,
                "default_resolution",
                DEFAULT_RESOLUTION,
            ),
            default_image_count=self._get_int(
                cfg,
                "default_image_count",
                DEFAULT_GENERATION_IMAGE_COUNT,
                min_value=1,
            ),
            max_image_count=self._get_int(
                cfg,
                "max_image_count",
                DEFAULT_MAX_GENERATION_IMAGE_COUNT,
                min_value=1,
            ),
            max_images_per_message=self._get_int(
                cfg,
                "max_images_per_message",
                DEFAULT_MAX_IMAGES_PER_MESSAGE,
                min_value=1,
            ),
            max_concurrent_tasks=self._get_int(
                cfg,
                "max_concurrent_tasks",
                DEFAULT_MAX_CONCURRENT_TASKS,
                min_value=1,
            ),
            non_retryable_status_codes=self._parse_int_list(
                cfg.get(
                    "non_retryable_status_codes",
                    list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
                ),
                list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
            ),
            non_retryable_error_keywords=self._parse_string_list_config(
                cfg.get(
                    "non_retryable_error_keywords",
                    list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
                ),
                list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
            ),
            result_info_items=self._parse_result_info_items(cfg),
            start_task_message_template=self._get_str(
                cfg,
                "start_task_message_template",
                GenerationSettings.start_task_message_template,
            ),
        )

    def _parse_result_info_items(self, cfg: dict[str, Any]) -> set[str]:
        """Parse selected result information items."""
        selected = self._parse_string_list(
            cfg.get("result_info_items", list(DEFAULT_RESULT_INFO_ITEMS))
        )
        valid_items = set(ALL_RESULT_INFO_ITEMS)
        return {item for item in selected if item in valid_items}

    def _parse_safety_audit_settings(self, cfg: dict[str, Any]) -> SafetyAuditSettings:
        """Parse prompt and image audit settings."""
        prompt_audit_cfg = self._get_nested_section(cfg, "prompt_audit")
        image_audit_cfg = self._get_nested_section(cfg, "image_audit")

        return SafetyAuditSettings(
            umo_whitelist=self._parse_string_list(cfg.get("umo_whitelist", [])),
            prompt_audit=self._parse_prompt_audit_settings(prompt_audit_cfg),
            image_audit=self._parse_image_audit_settings(image_audit_cfg),
        )

    def _parse_prompt_audit_settings(self, cfg: dict[str, Any]) -> PromptAuditSettings:
        """Parse prompt audit settings."""
        return PromptAuditSettings(
            blocked_words=self._parse_string_list(cfg.get("blocked_words", [])),
            enable_ai_audit=self._get_bool(cfg, "enable_ai_audit", False),
            ai_provider_id=self._get_str(cfg, "ai_provider_id", ""),
            max_retry_attempts=self._get_int(
                cfg,
                "max_retry_attempts",
                DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
                min_value=1,
            ),
            ai_prompt=self._get_str(cfg, "ai_prompt", PromptAuditSettings.ai_prompt),
        )

    def _parse_image_audit_settings(self, cfg: dict[str, Any]) -> ImageAuditSettings:
        """Parse image audit settings."""
        return ImageAuditSettings(
            enable_ai_audit=self._get_bool(cfg, "enable_ai_audit", False),
            ai_provider_id=self._get_str(cfg, "ai_provider_id", ""),
            max_retry_attempts=self._get_int(
                cfg,
                "max_retry_attempts",
                DEFAULT_AUDIT_MAX_RETRY_ATTEMPTS,
                min_value=1,
            ),
            ai_prompt=self._get_str(cfg, "ai_prompt", ImageAuditSettings.ai_prompt),
        )

    def reload(self) -> PluginConfig:
        """重新加载配置。"""
        return self.load()

    def _validate_config_values(self) -> None:
        """Validate config values and persist corrected values."""
        changed = self._config_validator.validate(self._config)
        if not changed:
            return
        self._config.save_config()

    def _get_config_section(self, name: str) -> dict[str, Any]:
        """Return a dictionary config section, falling back to an empty dict."""
        value = self._config.get(name, {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} 配置项 {safe_log_text(name)} 格式错误，已按空对象处理")
        return {}

    def _get_nested_section(self, cfg: dict[str, Any], key: str) -> dict[str, Any]:
        """Return a nested dictionary section, falling back to an empty dict."""
        value = cfg.get(key, {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} 配置项 {key} 格式错误，已按空对象处理")
        return {}

    def _get_str(
        self,
        cfg: dict[str, Any],
        key: str,
        default: str,
        *,
        strip: bool = True,
    ) -> str:
        """Read a config value as string."""
        value = cfg.get(key, default)
        if value is None:
            value = default
        parsed = str(value)
        return parsed.strip() if strip else parsed

    def _get_bool(self, cfg: dict[str, Any], key: str, default: bool) -> bool:
        """Read a config value as bool without treating arbitrary strings as true."""
        value = cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return default

    def _get_int(
        self,
        cfg: dict[str, Any],
        key: str,
        default: int,
        *,
        min_value: int,
    ) -> int:
        """Read a config value as int and clamp it."""
        return self._coerce_int(cfg.get(key, default), default, min_value=min_value)

    def _parse_enabled_llm_tools(self, raw: Any) -> list[str]:
        """Parse enabled LLM tool names from list config."""
        if isinstance(raw, bool):
            return list(ALL_LLM_TOOLS) if raw else []

        if not isinstance(raw, list):
            logger.warning(f"{LOG} enable_llm_tool 配置格式错误，已按空列表处理")
            return []

        selected: list[str] = []
        for item in raw:
            tool_name = str(item).strip()
            if tool_name in ALL_LLM_TOOLS and tool_name not in selected:
                selected.append(tool_name)
        return selected

    def _load_provider_configs(
        self, raw_providers: Any, gen_cfg: dict[str, Any]
    ) -> list[AdapterConfig]:
        """Parse all provider templates into normalized adapter configs."""
        if not isinstance(raw_providers, list):
            logger.warning(f"{LOG} api_providers 配置格式错误，已按空列表处理")
            return []

        provider_configs: list[AdapterConfig] = []
        for provider_item in raw_providers:
            if not isinstance(provider_item, dict):
                continue
            if parsed := self._parse_provider_config(provider_item, gen_cfg):
                provider_configs.append(parsed)
        return provider_configs

    def _parse_provider_config(
        self,
        provider_item: dict[str, Any],
        gen_cfg: dict[str, Any],
    ) -> AdapterConfig | None:
        """Parse one provider item with global fallback and provider overrides."""
        adapter_type = self._parse_adapter_type(provider_item)
        if not adapter_type:
            return None

        base_url = str(provider_item.get("base_url") or "").strip()
        proxy = str(provider_item.get("proxy") or "").strip() or None

        return AdapterConfig(
            type=adapter_type,
            name=str(provider_item.get("name", "")).strip(),
            base_url=self._clean_base_url(
                base_url,
                preserve_version_path=adapter_type == AdapterType.CUSTOM_HTTP,
            ),
            api_keys=self._parse_string_list(provider_item.get("api_keys", [])),
            available_models=self._parse_string_list(
                provider_item.get("available_models", [])
            ),
            proxy=proxy,
            timeout=self._get_provider_int_override(
                provider_item,
                gen_cfg,
                "timeout",
                DEFAULT_TIMEOUT,
                min_value=1,
            ),
            max_retry_attempts=self._get_provider_int_override(
                provider_item,
                gen_cfg,
                "max_retry_attempts",
                DEFAULT_MAX_RETRY_ATTEMPTS,
                min_value=0,
            ),
            non_retryable_status_codes=self._parse_int_list(
                gen_cfg.get(
                    "non_retryable_status_codes",
                    list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
                ),
                list(DEFAULT_NON_RETRYABLE_STATUS_CODES),
            ),
            non_retryable_error_keywords=self._parse_string_list_config(
                gen_cfg.get(
                    "non_retryable_error_keywords",
                    list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
                ),
                list(DEFAULT_NON_RETRYABLE_ERROR_KEYWORDS),
            ),
            capability_options=self._parse_capability_options(provider_item),
            extra=self._parse_provider_extra(adapter_type, provider_item),
        )

    def _parse_adapter_type(self, provider_item: dict[str, Any]) -> AdapterType | None:
        """Parse and validate the provider template key."""
        adapter_type_str = str(provider_item.get("__template_key") or "").strip()
        if not adapter_type_str:
            return None

        try:
            return AdapterType(adapter_type_str)
        except ValueError:
            logger.warning(
                f"{LOG} 忽略未知适配器类型: {safe_log_text(adapter_type_str)}"
            )
            return None

    def _get_provider_int_override(
        self,
        provider_item: dict[str, Any],
        gen_cfg: dict[str, Any],
        key: str,
        default: int,
        *,
        min_value: int,
    ) -> int:
        """Resolve an integer provider override, using global config by default."""
        global_value = self._coerce_int(
            gen_cfg.get(key, default), default, min_value=min_value
        )
        if key not in provider_item:
            return global_value

        raw_value = provider_item.get(key)
        if raw_value in (None, ""):
            return global_value

        provider_value = self._coerce_int(raw_value, global_value, min_value=0)
        if provider_value <= 0:
            return global_value
        return max(min_value, provider_value)

    def _coerce_int(self, value: Any, default: int, *, min_value: int) -> int:
        """Safely coerce a value to int and clamp it."""
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, parsed)

    def _parse_int_list(self, raw: Any, default: list[int]) -> list[int]:
        """Parse a list-like config value into unique integers."""
        if not isinstance(raw, list):
            return list(default)

        result: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                continue
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in result:
                result.append(value)
        return result

    def _parse_provider_extra(
        self,
        adapter_type: AdapterType,
        provider_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect adapter-specific settings without changing parser code later."""
        extra = dict(ADAPTER_EXTRA_DEFAULTS.get(adapter_type, {}))

        for key in extra:
            if key in provider_item:
                extra[key] = self._normalize_extra_value(provider_item[key])

        for key, value in provider_item.items():
            if key in PROVIDER_COMMON_FIELDS or key.startswith("__"):
                continue
            extra.setdefault(key, self._normalize_extra_value(value))

        return extra

    def _normalize_extra_value(self, value: Any) -> Any:
        """Normalize adapter-specific values before storing them in extra."""
        if isinstance(value, str):
            return value.strip()
        return value

    def _parse_string_list(self, raw: Any) -> list[str]:
        """Parse a list-like config value into non-empty strings."""
        if not isinstance(raw, list):
            return []
        return [item for item in (str(v).strip() for v in raw) if item]

    def _parse_string_list_config(self, raw: Any, default: list[str]) -> list[str]:
        """Parse a string list config value while preserving explicit empty lists."""
        if not isinstance(raw, list):
            return list(default)
        return self._parse_string_list(raw)

    def _select_adapter_config(
        self, provider_configs: list[AdapterConfig], model_setting: str
    ) -> AdapterConfig | None:
        """Select active provider config and attach full model choices."""
        matched_config: AdapterConfig | None = None
        current_model = ""

        if "/" in model_setting:
            target_provider_name, target_model = model_setting.split("/", 1)
            for cfg in provider_configs:
                if cfg.name == target_provider_name:
                    matched_config = cfg
                    current_model = target_model
                    break

        if not matched_config and provider_configs:
            matched_config = provider_configs[0]
            current_model = (
                matched_config.available_models[0]
                if matched_config.available_models
                else ""
            )
            logger.info(
                f"{LOG} 未匹配到当前模型配置，默认使用: {safe_log_text(matched_config.name)}/{safe_log_text(current_model)}"
            )

        if not matched_config:
            logger.error(f"{LOG} 未找到任何有效的生图模型配置")
            return None

        return replace(
            matched_config,
            model=current_model,
            available_models=self._build_model_choices(provider_configs),
        )

    def _build_model_choices(self, provider_configs: list[AdapterConfig]) -> list[str]:
        """Build display model choices in provider/model format."""
        choices: list[str] = []
        for cfg in provider_configs:
            choices.extend(f"{cfg.name}/{model}" for model in cfg.available_models)
        return choices

    def _parse_capability_options(
        self, provider_item: dict[str, Any]
    ) -> dict[str, bool]:
        """解析供应商能力配置（完全由配置驱动）。"""
        raw = provider_item.get("capability_options", [])

        supported_keys = (
            "text_to_image",
            "image_to_image",
            "aspect_ratio",
            "resolution",
        )

        if not isinstance(raw, list):
            logger.warning(f"{LOG} capability_options 配置格式错误，已按空列表处理")
            raw = []

        capability_alias_map = {
            "文生图": "text_to_image",
            "图生图": "image_to_image",
            "宽高比": "aspect_ratio",
            "分辨率": "resolution",
            # 允许英文值，便于手动配置文件时兼容
            "text_to_image": "text_to_image",
            "image_to_image": "image_to_image",
            "aspect_ratio": "aspect_ratio",
            "resolution": "resolution",
        }

        selected: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            key = capability_alias_map.get(item.strip())
            if key:
                selected.add(key)

        return {key: key in selected for key in supported_keys}

    def _clean_base_url(self, url: str, *, preserve_version_path: bool = False) -> str:
        """清理 Base URL，移除末尾的 /v1*"""
        if not url:
            return ""
        url = url.rstrip("/")
        if preserve_version_path:
            return url
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_presets(self, presets_config: list[Any]) -> dict[str, Any]:
        """加载预设配置。"""
        presets: dict[str, Any] = {}
        if not isinstance(presets_config, list):
            return presets

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets[name.strip()] = prompt.strip()
        return presets

    def _get_writable_prompt_templates_config(self) -> dict[str, Any]:
        """Return the grouped prompt-template config for command-side updates."""
        value = self._config.setdefault("prompt_templates", {})
        if isinstance(value, dict):
            return value
        logger.warning(f"{LOG} prompt_templates 配置格式错误，已重置为空对象")
        value = {}
        self._config["prompt_templates"] = value
        return value

    def _save_presets_config(self) -> None:
        prompt_templates_cfg = self._get_writable_prompt_templates_config()
        prompt_templates_cfg["presets"] = [
            f"{k}:{v}" for k, v in self._plugin_config.presets.items()
        ]
        self._config.save_config()

    def _load_personas(self, personas_config: Any) -> dict[str, PersonaTemplate]:
        """加载人设模板配置。"""
        personas: dict[str, PersonaTemplate] = {}
        if not isinstance(personas_config, list):
            return personas

        for item in personas_config:
            if not isinstance(item, dict):
                continue

            name = str(item.get("persona_name") or item.get("name") or "").strip()
            prompt = str(item.get("persona_prompt") or item.get("prompt") or "").strip()
            image = self._parse_file_value(
                item.get("persona_image")
                or item.get("image")
                or item.get("reference_image")
            )
            if name and (prompt or image):
                personas[name] = PersonaTemplate(name=name, prompt=prompt, image=image)
        return personas

    def _parse_file_value(self, raw: Any) -> str:
        """从 file 配置值中提取首个可用文件路径或 URL。"""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            for item in raw:
                if parsed := self._parse_file_value(item):
                    return parsed
            return ""
        if isinstance(raw, dict):
            for key in ("path", "file", "url", "name"):
                if parsed := self._parse_file_value(raw.get(key)):
                    return parsed
        return ""

    def save_model_setting(self, model: str) -> None:
        """保存模型设置。"""
        self._config.setdefault("generation", {})["model"] = model
        self._config.save_config()

    def save_preset(self, name: str, content: str) -> None:
        """保存预设。"""
        self._plugin_config.presets[name] = content
        self._save_presets_config()

    def delete_preset(self, name: str) -> bool:
        """删除预设，返回是否成功。"""
        if name in self._plugin_config.presets:
            del self._plugin_config.presets[name]
            self._save_presets_config()
            return True
        return False

    # ---------------------- 便捷属性访问 ----------------------
    @property
    def adapter_config(self) -> AdapterConfig | None:
        """获取适配器配置。"""
        return self._plugin_config.adapter_config

    @property
    def presets(self) -> dict[str, Any]:
        """获取预设字典。"""
        return self._plugin_config.presets

    @property
    def personas(self) -> dict[str, PersonaTemplate]:
        """获取人设模板字典。"""
        return self._plugin_config.personas

    def is_llm_tool_enabled(self, tool_name: str) -> bool:
        """检查指定 LLM 工具是否启用。"""
        return tool_name in self._plugin_config.enabled_llm_tools

    @property
    def default_aspect_ratio(self) -> str:
        """默认宽高比。"""
        return self._plugin_config.generation_settings.default_aspect_ratio

    @property
    def default_resolution(self) -> str:
        """默认分辨率。"""
        return self._plugin_config.generation_settings.default_resolution

    @property
    def default_image_count(self) -> int:
        """默认单次生成图片数量。"""
        return min(
            self._plugin_config.generation_settings.default_image_count,
            self.max_image_count,
        )

    @property
    def max_image_count(self) -> int:
        """单次最大生成图片数量。"""
        return self._plugin_config.generation_settings.max_image_count

    @property
    def max_images_per_message(self) -> int:
        """单条消息最多发送的图片数量。"""
        return self._plugin_config.generation_settings.max_images_per_message

    @property
    def max_concurrent_tasks(self) -> int:
        """最大并发生图请求数。"""
        return self._plugin_config.generation_settings.max_concurrent_tasks

    @property
    def result_info_items(self) -> set[str]:
        """生图成功后要展示的结果信息项。"""
        return self._plugin_config.generation_settings.result_info_items

    def should_show_result_info(self, item: str) -> bool:
        """检查指定结果信息项是否启用。"""
        return item in self.result_info_items

    @property
    def start_task_message_template(self) -> str:
        """开始生图任务提示模板。"""
        return self._plugin_config.generation_settings.start_task_message_template

    @property
    def usage_settings(self) -> UsageSettings:
        """用户使用限制设置。"""
        return self._plugin_config.usage_settings

    @property
    def safety_audit_settings(self) -> SafetyAuditSettings:
        """安全审核设置。"""
        return self._plugin_config.safety_audit_settings

    # ---------------------- 供应商查询方法 ----------------------
    def get_provider_config(self, adapter_type: AdapterType) -> AdapterConfig | None:
        """获取指定类型的供应商配置。

        Args:
            adapter_type: 要获取的适配器类型。

        Returns:
            匹配的供应商配置，如果没有则返回 None。
        """
        for cfg in self._all_provider_configs:
            if cfg.type == adapter_type:
                return cfg
        return None
