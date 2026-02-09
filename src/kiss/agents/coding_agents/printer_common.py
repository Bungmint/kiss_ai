"""Shared constants and utilities for printer modules."""

from pathlib import Path

LANG_MAP = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "rb": "ruby", "rs": "rust", "go": "go",
    "java": "java", "c": "c", "cpp": "cpp", "h": "c",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "xml": "xml", "html": "html",
    "css": "css", "sql": "sql", "md": "markdown",
}

MAX_RESULT_LEN = 3000

KNOWN_KEYS = {"file_path", "path", "content", "command", "old_string", "new_string", "description"}


def lang_for_path(path: str) -> str:
    ext = Path(path).suffix.lstrip(".")
    return LANG_MAP.get(ext, ext or "text")


def truncate_result(content: str) -> str:
    if len(content) <= MAX_RESULT_LEN:
        return content
    half = MAX_RESULT_LEN // 2
    return content[:half] + "\n... (truncated) ...\n" + content[-half:]


def extract_path_and_lang(tool_input: dict) -> tuple[str, str]:
    file_path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    lang = lang_for_path(file_path) if file_path else "text"
    return file_path, lang


def extract_extras(tool_input: dict) -> dict[str, str]:
    extras: dict[str, str] = {}
    for k, v in tool_input.items():
        if k not in KNOWN_KEYS:
            val = str(v)
            if len(val) > 200:
                val = val[:200] + "..."
            extras[k] = val
    return extras
