"""Structured console logging with a per-request correlation id."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

request_id_ctx: contextvars.ContextVar = contextvars.ContextVar("request_id", default="-")

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "neo4j",
    "neo4j.pool",
    "neo4j.io",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "qdrant_client",
    "google_genai",
    "google_genai.models",
    "PIL",
    "fastembed",
    "filelock",
)


class ConsoleFormatter(logging.Formatter):
    """One readable line per record: time LEVEL logger [request] message {fields}"""

    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        colour = self.COLORS.get(record.levelname, "") if self.use_color else ""
        reset = self.RESET if colour else ""
        rid = request_id_ctx.get()
        rid_part = " [" + rid[:8] + "]" if rid and rid != "-" else ""
        message = record.getMessage()
        fields = getattr(record, "extra_fields", None)
        if fields:
            message = message + "  " + json.dumps(fields, default=str, ensure_ascii=False)
        line = "{} {}{:<8}{} {:<30}{} {}".format(
            stamp, colour, record.levelname, reset, record.name, rid_part, message
        )
        if record.exc_info:
            line = line + "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ConsoleFormatter(use_color=sys.stdout.isatty()))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a message with structured key/value context appended to the line."""
    logger.info(message, extra={"extra_fields": fields})
