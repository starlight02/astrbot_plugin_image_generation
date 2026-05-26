"""Configuration migration and schema normalization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .config_defaults import (
    ALL_LLM_TOOLS,
    ALL_RESULT_INFO_ITEMS,
    DEFAULT_IMAGE_AUDIT_PROMPT,
    DEFAULT_PROMPT_AUDIT_PROMPT,
    DEFAULT_RESULT_INFO_ITEMS,
    LEGACY_IMAGE_AUDIT_PROMPTS,
    LEGACY_PROMPT_AUDIT_PROMPTS,
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
)
from .constants import LEGACY_AUTO_OPTION, UNSPECIFIED_OPTION


SCHEMA_DEFAULT_FACTORIES: dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "string": str,
    "text": str,
    "list": list,
    "file": list,
    "template_list": list,
}


class ConfigMigrator:
    """Migrate legacy config, then normalize it using schema metadata."""

    TEMPLATE_KEY_FIELD = "__template_key"
    LEGACY_TEMPLATE_KEY_FIELD = "template"
    TEMPLATE_KEY_ALIASES: dict[str, str] = {"z_image_gitee": "gitee_ai"}
    FIELD_ALIASES: dict[str, dict[str, str]] = {
        "prompt_templates.personas[]": {
            "name": "persona_name",
            "prompt": "persona_prompt",
            "image": "persona_image",
            "reference_image": "persona_image",
        },
    }
    VALUE_ALIASES: dict[str, dict[Any, Any]] = {
        "generation.default_aspect_ratio": {LEGACY_AUTO_OPTION: UNSPECIFIED_OPTION},
        "generation.default_resolution": {LEGACY_AUTO_OPTION: UNSPECIFIED_OPTION},
    }
    LIST_ADDITIONS_ON_TEMPLATE_MIGRATION: dict[str, dict[str, list[Any]]] = {
        "z_image_gitee": {"capability_options": ["图生图"]},
    }
    _SENTINEL = object()

    def __init__(self, schema: Mapping[str, Any] | None):
        self._schema = schema if isinstance(schema, Mapping) else {}

    @classmethod
    def normalize_template_key(cls, value: Any) -> str:
        template_key = str(value or "").strip()
        return cls.TEMPLATE_KEY_ALIASES.get(template_key, template_key)

    def migrate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        changed = False
        messages: list[str] = []

        changed |= self._migrate_enable_llm_tool(config, messages)
        changed |= self._move_legacy_prompt_templates(config, messages)
        changed |= self._migrate_legacy_safety_audit_prompts(config, messages)
        changed |= self._migrate_generation_result_info_items(config, messages)

        if not self._schema:
            return changed, messages

        normalized, normalize_changed, normalize_messages = self._normalize_object(
            config,
            self._schema,
            path="",
        )
        changed |= normalize_changed
        messages.extend(normalize_messages)
        if normalize_changed:
            config.clear()
            config.update(normalized)
        return changed, messages

    def _migrate_enable_llm_tool(
        self, config: dict[str, Any], messages: list[str]
    ) -> bool:
        value = config.get("enable_llm_tool")
        if not isinstance(value, bool):
            return False

        config["enable_llm_tool"] = list(ALL_LLM_TOOLS) if value else []
        messages.append("enable_llm_tool: bool -> list")
        return True

    def _move_legacy_prompt_templates(
        self, config: dict[str, Any], messages: list[str]
    ) -> bool:
        legacy_presets = self._pop_if_present(config, "presets")
        legacy_personas = self._pop_if_present(config, "personas")
        if legacy_presets is self._SENTINEL and legacy_personas is self._SENTINEL:
            return False

        prompt_templates = config.setdefault("prompt_templates", {})
        if not isinstance(prompt_templates, dict):
            prompt_templates = {}
            config["prompt_templates"] = prompt_templates

        if legacy_presets is not self._SENTINEL and "presets" not in prompt_templates:
            prompt_templates["presets"] = legacy_presets
            messages.append("presets -> prompt_templates.presets")

        if legacy_personas is not self._SENTINEL and "personas" not in prompt_templates:
            prompt_templates["personas"] = legacy_personas
            messages.append("personas -> prompt_templates.personas")

        return True

    def _migrate_legacy_safety_audit_prompts(
        self, config: dict[str, Any], messages: list[str]
    ) -> bool:
        """Replace old built-in safety prompts with the current defaults."""
        safety_cfg = config.get("safety_audit")
        if not isinstance(safety_cfg, dict):
            return False

        replacements = (
            (
                "prompt_audit",
                LEGACY_PROMPT_AUDIT_PROMPTS,
                DEFAULT_PROMPT_AUDIT_PROMPT,
            ),
            (
                "image_audit",
                LEGACY_IMAGE_AUDIT_PROMPTS,
                DEFAULT_IMAGE_AUDIT_PROMPT,
            ),
        )
        changed = False
        for section_name, old_prompts, new_prompt in replacements:
            section = safety_cfg.get(section_name)
            if not isinstance(section, dict):
                continue
            if section.get("ai_prompt") not in old_prompts:
                continue
            section["ai_prompt"] = new_prompt
            messages.append(
                f"safety_audit.{section_name}.ai_prompt: updated built-in prompt"
            )
            changed = True
        return changed

    def _migrate_generation_result_info_items(
        self, config: dict[str, Any], messages: list[str]
    ) -> bool:
        """Convert legacy result-info switches to the selectable item list."""
        generation_cfg = config.get("generation")
        if not isinstance(generation_cfg, dict):
            return False

        if "result_info_items" in generation_cfg:
            return False

        has_generation_info = "show_generation_info" in generation_cfg
        has_model_info = "show_model_info" in generation_cfg
        if not has_generation_info and not has_model_info:
            return False

        selected = set(DEFAULT_RESULT_INFO_ITEMS)
        if self._coerce_bool(generation_cfg.get("show_generation_info"), False):
            selected.update((RESULT_INFO_DURATION, RESULT_INFO_COUNT))
        if self._coerce_bool(generation_cfg.get("show_model_info"), False):
            selected.add(RESULT_INFO_MODEL)

        generation_cfg["result_info_items"] = [
            item for item in ALL_RESULT_INFO_ITEMS if item in selected
        ]
        messages.append(
            "generation.show_generation_info/show_model_info -> generation.result_info_items"
        )
        return True

    def _normalize_object(
        self,
        raw: Any,
        schema: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[dict[str, Any], bool, list[str]]:
        messages: list[str] = []
        changed = False

        if isinstance(raw, Mapping):
            raw_mapping, alias_changed, alias_messages = self._apply_field_aliases(
                raw, path=path
            )
            changed |= alias_changed
            messages.extend(alias_messages)
        else:
            raw_mapping = {}
            changed = True
            messages.append(f"{path or '<root>'}: reset to object")

        normalized: dict[str, Any] = {}
        for key, meta in schema.items():
            key_path = self._join_path(path, key)
            if key in raw_mapping and raw_mapping[key] is not None:
                value, value_changed, value_messages = self._normalize_value(
                    raw_mapping[key],
                    meta,
                    path=key_path,
                )
                normalized[key] = value
                changed |= value_changed
                messages.extend(value_messages)
            else:
                normalized[key] = self._schema_default(meta)
                changed = True
                messages.append(f"{key_path}: add default")

        for key in raw_mapping:
            if key not in schema:
                changed = True
                messages.append(
                    f"{self._join_path(path, str(key))}: removed obsolete key"
                )

        if list(raw_mapping.keys()) != list(normalized.keys()):
            changed = True
            if set(raw_mapping.keys()) == set(normalized.keys()):
                messages.append(f"{path or '<root>'}: fixed key order")

        return normalized, changed, messages

    def _normalize_value(
        self,
        raw: Any,
        meta: Any,
        *,
        path: str,
    ) -> tuple[Any, bool, list[str]]:
        if not isinstance(meta, Mapping):
            return copy.deepcopy(raw), False, []

        meta_type = meta.get("type")
        if meta_type == "object":
            items = meta.get("items")
            if not isinstance(items, Mapping):
                return (
                    self._schema_default(meta),
                    raw is not None,
                    [f"{path}: reset to object"],
                )
            return self._normalize_object(raw, items, path=path)

        if meta_type == "template_list":
            return self._normalize_template_list(raw, meta, path=path)

        return self._normalize_leaf_value(raw, meta, path=path)

    def _normalize_template_list(
        self,
        raw: Any,
        meta: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[list[Any], bool, list[str]]:
        if not isinstance(raw, list):
            return self._schema_default(meta), True, [f"{path}: reset to list"]

        templates = meta.get("templates")
        if not isinstance(templates, Mapping):
            templates = {}

        normalized_items: list[Any] = []
        changed = False
        messages: list[str] = []

        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"
            if not isinstance(item, Mapping):
                changed = True
                messages.append(f"{item_path}: removed non-object item")
                continue

            template_key, old_template_key, key_changed, key_messages = (
                self._normalize_template_key(
                    item,
                    templates,
                    item_path=item_path,
                )
            )
            changed |= key_changed
            messages.extend(key_messages)
            if not template_key:
                changed = True
                messages.append(f"{item_path}: removed item without template")
                continue

            template_meta = templates.get(template_key)
            if not isinstance(template_meta, Mapping):
                changed = True
                messages.append(
                    f"{item_path}: removed unknown template {template_key!r}"
                )
                continue

            item_schema = template_meta.get("items")
            if not isinstance(item_schema, Mapping):
                item_schema = {}

            child_raw = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    self.TEMPLATE_KEY_FIELD,
                    self.LEGACY_TEMPLATE_KEY_FIELD,
                }
            }
            changed |= self._ensure_list_values(
                child_raw,
                self.LIST_ADDITIONS_ON_TEMPLATE_MIGRATION.get(old_template_key, {}),
                messages,
                label=item_path,
            )
            child_normalized, child_changed, child_messages = self._normalize_object(
                child_raw,
                item_schema,
                path=item_path,
            )
            normalized_item = {self.TEMPLATE_KEY_FIELD: template_key}
            normalized_item.update(child_normalized)

            if dict(item) != normalized_item:
                changed = True
            changed |= child_changed
            messages.extend(child_messages)
            normalized_items.append(normalized_item)

        if len(normalized_items) != len(raw):
            changed = True

        return normalized_items, changed, messages

    def _normalize_template_key(
        self,
        item: Mapping[str, Any],
        templates: Mapping[str, Any],
        *,
        item_path: str,
    ) -> tuple[str, str, bool, list[str]]:
        messages: list[str] = []
        changed = False

        raw_key = item.get(self.TEMPLATE_KEY_FIELD)
        legacy_key = item.get(self.LEGACY_TEMPLATE_KEY_FIELD)
        if legacy_key not in (None, ""):
            changed = True
            messages.append(
                f"{item_path}.{self.LEGACY_TEMPLATE_KEY_FIELD}: removed legacy key"
            )
            if raw_key in (None, ""):
                raw_key = legacy_key
                messages.append(
                    f"{item_path}.{self.LEGACY_TEMPLATE_KEY_FIELD} -> {self.TEMPLATE_KEY_FIELD}"
                )

        old_template_key = str(raw_key).strip() if raw_key not in (None, "") else ""
        template_key = self.normalize_template_key(old_template_key)
        if template_key != old_template_key:
            messages.append(
                f"{item_path}.{self.TEMPLATE_KEY_FIELD}: {old_template_key!r} -> {template_key!r}"
            )
            changed = True

        if not template_key and len(templates) == 1:
            template_key = next(iter(templates))
            messages.append(
                f"{item_path}.{self.TEMPLATE_KEY_FIELD}: add default {template_key!r}"
            )
            changed = True

        if item.get(self.TEMPLATE_KEY_FIELD) != template_key:
            changed = True

        return template_key, old_template_key, changed, messages

    def _apply_field_aliases(
        self,
        raw: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[dict[str, Any], bool, list[str]]:
        aliases = self._field_aliases_for_path(path)
        if not aliases:
            return dict(raw), False, []

        changed = False
        messages: list[str] = []
        normalized = dict(raw)
        for legacy_key, current_key in aliases.items():
            if legacy_key not in normalized:
                continue
            if current_key not in normalized:
                normalized[current_key] = normalized[legacy_key]
                messages.append(f"{path}.{legacy_key} -> {current_key}")
            normalized.pop(legacy_key, None)
            changed = True
        return normalized, changed, messages

    def _field_aliases_for_path(self, path: str) -> dict[str, str]:
        if path.startswith("prompt_templates.personas["):
            return self.FIELD_ALIASES["prompt_templates.personas[]"]
        return {}

    def _ensure_list_values(
        self,
        target: dict[str, Any],
        additions: dict[str, list[Any]],
        messages: list[str],
        *,
        label: str,
    ) -> bool:
        changed = False
        for key, values in additions.items():
            current = target.get(key)
            if not isinstance(current, list):
                continue
            for value in values:
                if value not in current:
                    current.append(value)
                    changed = True
                    messages.append(f"{label}.{key}: add {value!r}")
        return changed

    def _normalize_leaf_value(
        self,
        raw: Any,
        meta: Mapping[str, Any],
        *,
        path: str,
    ) -> tuple[Any, bool, list[str]]:
        meta_type = str(meta.get("type") or "")
        default = self._schema_default(meta)
        value = self._coerce_schema_value(raw, meta_type, default)
        changed = value != raw

        value, alias_changed = self._apply_value_aliases(value, path=path)
        changed |= alias_changed

        value, options_changed = self._normalize_options(value, meta)
        changed |= options_changed
        return value, changed, [f"{path}: normalized by schema"] if changed else []

    def _apply_value_aliases(self, value: Any, *, path: str) -> tuple[Any, bool]:
        aliases = self._value_aliases_for_path(path)
        if not aliases:
            return value, False

        if isinstance(value, list):
            normalized = [aliases.get(item, item) for item in value]
            return normalized, normalized != value

        normalized = aliases.get(value, value)
        return normalized, normalized != value

    def _value_aliases_for_path(self, path: str) -> dict[Any, Any]:
        return self.VALUE_ALIASES.get(path, {})

    def _coerce_schema_value(self, raw: Any, meta_type: str, default: Any) -> Any:
        """Coerce a scalar schema value without applying option filters."""
        if meta_type == "int":
            return self._coerce_number(raw, default, int)
        if meta_type == "float":
            return self._coerce_number(raw, default, float)
        if meta_type == "bool":
            return self._coerce_bool(raw, default)
        if meta_type in {"string", "text"}:
            return raw if isinstance(raw, str) else default if raw is None else str(raw)
        if meta_type == "file":
            return self._coerce_file_value(raw, default)
        if meta_type == "list":
            return copy.deepcopy(raw) if isinstance(raw, list) else default
        return copy.deepcopy(raw)

    def _coerce_number(self, raw: Any, default: Any, parser: Any) -> Any:
        """Coerce int or float config values."""
        if isinstance(raw, bool):
            return default
        try:
            return parser(raw)
        except (TypeError, ValueError):
            return default

    def _coerce_bool(self, raw: Any, default: bool) -> bool:
        """Coerce bool config values with explicit string support."""
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return default

    def _coerce_file_value(self, raw: Any, default: Any) -> list[Any]:
        """Coerce file config values to AstrBot file-list shape."""
        if isinstance(raw, list):
            return copy.deepcopy(raw)
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return default

    def _normalize_options(
        self,
        value: Any,
        meta: Mapping[str, Any],
    ) -> tuple[Any, bool]:
        options = meta.get("options")
        if not isinstance(options, list):
            return value, False

        option_set = set(options)
        if isinstance(value, list):
            normalized = [item for item in value if item in option_set]
            return normalized, normalized != value

        if value in option_set:
            return value, False

        return self._schema_default(meta), True

    def _schema_default(self, meta: Any) -> Any:
        if not isinstance(meta, Mapping):
            return None

        meta_type = str(meta.get("type") or "")
        if meta_type == "object":
            items = meta.get("items")
            if not isinstance(items, Mapping):
                return {}
            return {key: self._schema_default(value) for key, value in items.items()}

        if "default" in meta:
            return copy.deepcopy(meta["default"])

        default_factory = SCHEMA_DEFAULT_FACTORIES.get(meta_type)
        return default_factory() if default_factory else None

    def _join_path(self, base: str, key: str) -> str:
        return f"{base}.{key}" if base else key

    def _pop_if_present(self, target: dict[str, Any], key: str) -> Any:
        if key not in target:
            return self._SENTINEL
        return target.pop(key)
