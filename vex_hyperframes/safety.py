from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any


@dataclass(frozen=True)
class AuthoredHtmlSafetyReport:
    safe: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SafetyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.tags.append(
            (
                tag.lower(),
                {str(key).lower(): str(value or "") for key, value in attrs},
            )
        )


_FORBIDDEN_TAGS = {
    "applet",
    "audio",
    "base",
    "embed",
    "form",
    "iframe",
    "input",
    "link",
    "meta",
    "object",
    "script",
    "source",
    "video",
}
_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"https?://", "remote_url"),
    (r"\bjavascript\s*:", "javascript_url"),
    (r"\bdata\s*:\s*text/html", "executable_data_url"),
    (r"\b(?:eval|Function)\s*\(", "dynamic_code_execution"),
    (r"\b(?:fetch|WebSocket|XMLHttpRequest|EventSource)\b", "network_api"),
    (r"\b(?:Worker|SharedWorker|ServiceWorker)\b", "worker_api"),
    (r"\bimport\s*\(", "dynamic_import"),
    (r"\b(?:document\.cookie|localStorage|sessionStorage|indexedDB)\b", "browser_storage"),
    (
        r"(?:\brequire\s*\(|\bprocess\s*\.|\bchild_process\b|\bDeno\s*\.)",
        "host_runtime_api",
    ),
    (r"\b(?:requestAnimationFrame|setInterval|setTimeout)\b", "wall_clock_api"),
    (r"@import\b", "css_import"),
    (r"url\s*\(", "css_external_resource"),
)


def validate_authored_html_safety(
    html: str,
    *,
    max_chars: int = 120_000,
) -> AuthoredHtmlSafetyReport:
    source = str(html or "")
    errors: list[str] = []
    warnings: list[str] = []
    if not source.strip():
        return AuthoredHtmlSafetyReport(False, ["authored_html_is_empty"])
    if len(source) > max_chars:
        errors.append("authored_html_exceeds_size_limit")
    parser = _SafetyParser()
    try:
        parser.feed(source)
    except Exception:
        errors.append("authored_html_parse_failed")
        return AuthoredHtmlSafetyReport(False, errors)
    for tag, attrs in parser.tags:
        if tag in _FORBIDDEN_TAGS:
            errors.append(f"forbidden_html_tag:{tag}")
        for name, value in attrs.items():
            if name.startswith("on"):
                errors.append(f"event_handler_attribute:{name}")
            if name in {"href", "src", "srcdoc", "action", "formaction"} and value:
                errors.append(f"external_resource_attribute:{name}")
            if name == "style" and "url(" in value.lower():
                errors.append("inline_style_external_resource")
    for pattern, code in _FORBIDDEN_PATTERNS:
        if re.search(pattern, source, flags=re.IGNORECASE):
            errors.append(code)
    if "<style" not in source.lower():
        warnings.append("authored_fragment_has_no_scoped_style")
    return AuthoredHtmlSafetyReport(
        safe=not errors,
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "AuthoredHtmlSafetyReport",
    "validate_authored_html_safety",
]
