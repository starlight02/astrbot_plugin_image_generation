"""Prompt template helpers for presets and personas."""

from __future__ import annotations

import json
from typing import Any

from .reference_collector import normalize_string_items


def normalize_name_items(raw: Any) -> list[str]:
    """Normalize one or many preset/persona names from tool arguments."""
    names: list[str] = []
    seen: set[str] = set()
    for item in normalize_string_items(raw):
        for name in item.split():
            normalized = name.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(normalized)
    return names


def find_named_entry(entries: dict[str, Any], token: str) -> str | None:
    """Find an entry by exact or case-insensitive name."""
    if token in entries:
        return token
    lowered_token = token.lower()
    for name in entries:
        if name.lower() == lowered_token:
            return name
    return None


def parse_preset_prompt(
    preset_content: Any,
    aspect_ratio: str,
    resolution: str,
) -> tuple[str, str, str]:
    """Parse a preset prompt and optional generation overrides."""
    preset_prompt = str(preset_content or "").strip()
    if not preset_prompt.startswith("{"):
        return preset_prompt, aspect_ratio, resolution

    try:
        preset_data = json.loads(preset_prompt)
    except json.JSONDecodeError:
        return preset_prompt, aspect_ratio, resolution

    if not isinstance(preset_data, dict):
        return preset_prompt, aspect_ratio, resolution

    preset_prompt = str(preset_data.get("prompt", "") or "").strip()
    aspect_ratio = str(preset_data.get("aspect_ratio") or aspect_ratio)
    resolution = str(preset_data.get("resolution") or resolution)
    return preset_prompt, aspect_ratio, resolution


def format_template_summary(
    matched_presets: list[str],
    matched_personas: list[str],
) -> tuple[str | None, str]:
    """Format matched preset/persona names for task metadata."""
    if matched_presets and matched_personas:
        return (
            "；".join(
                (
                    f"预设: {'、'.join(matched_presets)}",
                    f"人设: {'、'.join(matched_personas)}",
                )
            ),
            "预设/人设",
        )
    if matched_presets:
        return "、".join(matched_presets), "预设"
    if matched_personas:
        return "、".join(matched_personas), "人设"
    return None, "预设"
