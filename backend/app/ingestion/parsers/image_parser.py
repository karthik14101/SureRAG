"""Image ingestion: OCR text, a vision-model caption, and EXIF metadata.

An image with no text attached is invisible to a text retriever, so we generate
something searchable through three independent routes and index whatever we get:

  1. OCR (pytesseract) -- needs the Tesseract BINARY, not just the pip package.
     Auto-detected; absent means this step is skipped, not an error.
  2. Vision captioning -- Gemini only. GroqCloud ships no vision model today.
  3. Filename and EXIF -- always available, and often genuinely informative.
"""
from __future__ import annotations

import io
import pathlib
import threading

from app.config import settings
from app.ingestion.parsers.base import (
    ExtractedImage,
    ParsedDocument,
    ParserError,
    TextBlock,
    clean_text,
    guess_mime,
)
from app.logging_conf import get_logger

logger = get_logger(__name__)

_ocr_checked = False
_ocr_available = False
_ocr_lock = threading.Lock()

CAPTION_PROMPT = (
    "Describe this image for a document search index. State what it depicts, and "
    "transcribe any visible text, labels, axis titles or numbers exactly. If it is "
    "a chart or diagram, describe what it shows and the relationship it conveys. "
    "Be factual and specific. Do not speculate. Maximum 120 words."
)

MAX_VISION_BYTES = 4 * 1024 * 1024


def ocr_available() -> bool:
    """Probe for the Tesseract binary once per process."""
    global _ocr_checked, _ocr_available
    if _ocr_checked:
        return _ocr_available
    with _ocr_lock:
        if _ocr_checked:
            return _ocr_available
        _ocr_checked = True
        if not settings.ocr_enabled:
            _ocr_available = False
            return False
        try:
            import pytesseract

            if settings.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            version = pytesseract.get_tesseract_version()
            _ocr_available = True
            logger.info("OCR enabled (Tesseract %s)", str(version).split("\n")[0])
        except Exception as exc:  # noqa: BLE001
            _ocr_available = False
            logger.info(
                "OCR disabled -- the Tesseract binary was not found (%s). Images will "
                "be indexed by caption and filename only. See README step 0.4.",
                str(exc)[:120],
            )
    return _ocr_available


def run_ocr(data: bytes) -> str:
    if not ocr_available():
        return ""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            # Tiny images OCR poorly; upscaling helps materially.
            if img.width < 800:
                scale = min(3.0, 800 / max(1, img.width))
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            text = pytesseract.image_to_string(img)
        cleaned = clean_text(text)
        # Tesseract on a photo returns punctuation soup; require real words.
        words = [w for w in cleaned.split() if len(w) > 2 and any(c.isalnum() for c in w)]
        return cleaned if len(words) >= 3 else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("OCR failed: %s", str(exc)[:160])
        return ""


async def caption_image(data: bytes, mime: str) -> str | None:
    """Vision caption via the active provider, or None when unsupported."""
    if len(data) > MAX_VISION_BYTES:
        return None
    try:
        from app.llm.factory import get_llm

        llm = get_llm()
        if not llm.supports_vision:
            return None
        caption = await llm.describe_image(data, mime, CAPTION_PROMPT)
        return clean_text(caption) if caption else None
    except Exception as exc:  # noqa: BLE001 - captioning is best-effort
        logger.debug("Vision captioning unavailable: %s", str(exc)[:160])
        return None


def _read_metadata(data: bytes) -> tuple[int, int, dict]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            width, height = img.width, img.height
            meta: dict = {"format": img.format or "", "mode": img.mode}
            try:
                exif = img.getexif()
                if exif:
                    interesting = {
                        271: "camera_make",
                        272: "camera_model",
                        306: "datetime",
                        270: "description",
                        315: "artist",
                    }
                    for tag, name in interesting.items():
                        value = exif.get(tag)
                        if value:
                            meta[name] = str(value).strip()[:200]
            except Exception:  # noqa: BLE001 - EXIF is optional
                pass
            return width, height, meta
    except Exception as exc:  # noqa: BLE001
        raise ParserError("Could not read the image: {}".format(str(exc)[:160])) from exc


def _normalise_for_vision(data: bytes, mime: str) -> tuple[bytes, str]:
    """Convert exotic formats to PNG so the vision API accepts them."""
    if mime in ("image/png", "image/jpeg", "image/webp"):
        return data, mime
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
    except Exception:  # noqa: BLE001
        return data, mime


async def parse_image(path, filename: str) -> ParsedDocument:
    path = pathlib.Path(path)
    try:
        data = path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        raise ParserError("Could not read '{}': {}".format(filename, str(exc)[:160])) from exc

    if not data:
        raise ParserError("'{}' is empty.".format(filename))

    mime = guess_mime(filename, "image/png")
    width, height, meta = _read_metadata(data)

    result = ParsedDocument()
    result.metadata = {"width": width, "height": height, **meta}

    ocr_text = run_ocr(data)
    vision_data, vision_mime = _normalise_for_vision(data, mime)
    caption = await caption_image(vision_data, vision_mime)

    image = ExtractedImage(
        ref="upload_0",
        data=data,
        mime=mime,
        width=width,
        height=height,
        page_no=None,
        caption=caption,
        ocr_text=ocr_text or None,
        origin="uploaded",
    )
    result.images.append(image)

    # Build one searchable block from every signal we managed to collect.
    readable_name = clean_text(path.stem.replace("_", " ").replace("-", " "))
    parts: list[str] = ["Image file: {}".format(filename)]
    if readable_name and readable_name.casefold() != path.stem.casefold():
        parts.append("Title: {}".format(readable_name))
    if caption:
        parts.append("Description: {}".format(caption))
    if ocr_text:
        parts.append("Text in image: {}".format(ocr_text))
    for key in ("description", "camera_model", "datetime", "artist"):
        if meta.get(key):
            parts.append("{}: {}".format(key.replace("_", " ").title(), meta[key]))
    parts.append("Dimensions: {} x {} pixels".format(width, height))

    result.blocks.append(
        TextBlock(
            text=clean_text("\n".join(parts)),
            page_no=None,
            section=readable_name or filename,
            modality="ocr" if ocr_text else "caption",
            media_refs=[image.ref],
            order=1,
        )
    )

    if not caption and not ocr_text:
        result.warnings.append(
            "Only the filename and metadata could be indexed for '{}'. Enable OCR "
            "(README step 0.4) or use LLM_PROVIDER=gemini for image "
            "descriptions.".format(filename)
        )

    return result
