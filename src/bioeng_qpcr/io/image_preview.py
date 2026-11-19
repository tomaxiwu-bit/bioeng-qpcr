"""Multi-format image inspection and preview for lab imagery.

The images a bioengineering lab actually works with are not all holiday
photos: alongside JPEG snapshots of a gel tray there are 16-bit single-channel
fluorescence TIFFs, multi-frame time series, multi-channel stacks that have to
be merged into a pseudo-colour composite, and vector figures.  This module
provides one preview path that handles all of them and, crucially, **tells the
user what it found instead of quietly converting it away**.

Two reading backends
--------------------
* :mod:`Pillow` when it is installed -- the full path, giving pixel access,
  thumbnailing, EXIF, multi-page frames, and PNG re-encoding for data URIs.
* A built-in header sniffer when it is not -- PNG, JPEG, GIF, BMP, TIFF and
  WebP are parsed straight from their headers so metadata (format, size, bit
  depth, channel count) is still reported.  Pixel-dependent features report
  ``preview_available = False`` together with the reason, rather than failing.

That split matters because Pillow lives in the optional ``gui`` extra, and a
command-line user running only MIQE validation should still be able to ask
"what is this file?" without installing a imaging stack.

Image *types* recognised
------------------------
======================  ==================================================
Single-channel 8-bit    gel documentation, scanned blots
Single-channel 16-bit   quantitative fluorescence (kept at full depth)
RGB / RGBA              colour photographs, annotated figures
Multi-frame             TIFF stacks and animated GIFs (frame index exposed)
Multi-channel stack     separate single-channel files merged into a composite
Vector                  SVG (metadata only; no rasteriser is assumed)
======================  ==================================================
"""

from __future__ import annotations

import base64
import io
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CHANNEL_PALETTES",
    "ImageInfo",
    "ImagePreview",
    "compose_channels",
    "inspect_image",
    "list_supported_formats",
    "load_preview",
    "save_preview_png",
    "split_channels",
    "to_data_uri",
]

try:  # pragma: no cover - exercised implicitly by whichever branch is present
    from PIL import Image

    _PILLOW_AVAILABLE = True
    _PILLOW_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    Image = None  # type: ignore[assignment]
    _PILLOW_AVAILABLE = False
    _PILLOW_ERROR = f"{type(exc).__name__}: {exc}"

#: Pixel decoders may need several bytes per pixel even when the compressed
#: upload is small.  Larger images remain inspectable, but are not rendered.
_MAX_PREVIEW_PIXELS = 25_000_000


#: Pseudo-colour lookup tables offered for single-channel images.
#:
#: Each palette maps a normalised channel value in ``[0, 1]`` to an RGB triple
#: by linear interpolation between the listed stops, so a 16-bit fluorescence
#: channel can be displayed in the colour of the fluorophore it represents.
CHANNEL_PALETTES: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "gray": ((0.0, (0, 0, 0)), (1.0, (255, 255, 255))),
    "green": ((0.0, (0, 0, 0)), (0.5, (0, 160, 60)), (1.0, (170, 255, 170))),
    "red": ((0.0, (0, 0, 0)), (0.5, (180, 30, 30)), (1.0, (255, 170, 150))),
    "blue": ((0.0, (0, 0, 0)), (0.5, (40, 90, 200)), (1.0, (170, 200, 255))),
    "magenta": ((0.0, (0, 0, 0)), (0.5, (170, 40, 170)), (1.0, (255, 170, 255))),
    "cyan": ((0.0, (0, 0, 0)), (0.5, (40, 170, 180)), (1.0, (170, 250, 255))),
    "yellow": ((0.0, (0, 0, 0)), (0.5, (180, 160, 20)), (1.0, (255, 250, 170))),
    "fire": (
        (0.0, (0, 0, 0)),
        (0.33, (150, 20, 0)),
        (0.66, (240, 140, 0)),
        (1.0, (255, 255, 210)),
    ),
}

#: Formats Pillow is asked to open, with a short description for the UI.
_PILLOW_FORMATS: dict[str, str] = {
    "PNG": "无损位图，适合凝胶与截图",
    "JPEG": "有损照片格式",
    "TIFF": "科研影像常用容器，支持多页/16 位",
    "BMP": "未压缩位图",
    "GIF": "支持动画与调色板",
    "WEBP": "现代有损/无损格式",
    "ICO": "图标容器",
    "PSD": "Photoshop 文档（合并图层预览）",
    "TGA": "Targa 位图",
    "PPM": "Portable Pixmap",
    "JPEG2000": "JPEG 2000",
    "AVIF": "AV1 图片格式",
    "DDS": "DirectDraw Surface",
    "PCX": "PC Paintbrush",
    "SGI": "Silicon Graphics 位图",
    "FITS": "天文/科学 FITS",
    "EPS": "Encapsulated PostScript（栅格化预览）",
    "PDF": "矢量/多页文档（首屏栅格化预览）",
}

#: Extensions handled by the header sniffer when Pillow is unavailable, with
#: the magic-byte test used to recognise them.
_HEADER_SNIFFERS: dict[str, str] = {
    "PNG": "\\x89PNG",
    "JPEG": "\\xff\\xd8",
    "GIF": "GIF8",
    "BMP": "BM",
    "TIFF": "II* or MM*",
    "WEBP": "RIFF....WEBP",
    "SVG": "<?xml / <svg",
}


# ---------------------------------------------------------------------------
# Pure-Python header parsing
# ---------------------------------------------------------------------------


def _sniff_png(data: bytes) -> dict[str, Any]:
    """Read width, height, bit depth and colour type from a PNG IHDR."""
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return {}
    width, height = struct.unpack(">II", data[16:24])
    depth, colour_type = data[24], data[25]
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour_type, 0)
    return {
        "width": width,
        "height": height,
        "bits_per_channel": depth,
        "channels": channels,
        "has_alpha": colour_type in (4, 6),
        "mode": "RGBA" if colour_type == 6 else "RGB" if colour_type == 2 else "L",
    }


def _sniff_jpeg(data: bytes) -> dict[str, Any]:
    """Walk JPEG segments to the first SOFn marker and read the frame size."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return {}
    index = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if index + 4 > len(data):
            break
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if marker in sof_markers:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            components = data[index + 9]
            return {
                "width": width,
                "height": height,
                "channels": components,
                "bits_per_channel": data[index + 4],
                "has_alpha": False,
                "mode": "L" if components == 1 else "RGB",
            }
        index += 2 + length
    return {}


def _sniff_gif(data: bytes) -> dict[str, Any]:
    """Read GIF logical-screen dimensions, plus a frame count when cheap."""
    if len(data) < 10 or data[:3] != b"GIF":
        return {}
    width, height = struct.unpack("<HH", data[6:10])
    # Counting image separators gives the frame count; the whole file is
    # already in memory, so this is a linear scan rather than a parse.
    frames = max(data.count(b"\x2c"), 1)
    return {
        "width": width,
        "height": height,
        "channels": 3,
        "bits_per_channel": 8,
        "has_alpha": True,
        "mode": "P",
        "n_frames": frames,
        "is_animated": frames > 1,
    }


def _sniff_bmp(data: bytes) -> dict[str, Any]:
    """Read BMP dimensions and bit depth from the DIB header."""
    if len(data) < 30 or data[:2] != b"BM":
        return {}
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size == 12:  # BITMAPCOREHEADER
        width, height = struct.unpack("<HH", data[18:22])
        depth = struct.unpack("<H", data[24:26])[0]
    else:
        width, height = struct.unpack("<ii", data[18:26])
        depth = struct.unpack("<H", data[28:30])[0]
    height = abs(height)
    channels = 4 if depth == 32 else 3 if depth == 24 else 1
    return {
        "width": width,
        "height": height,
        "bits_per_channel": depth // 8 if depth >= 8 else depth,
        "channels": channels,
        "has_alpha": depth == 32,
        "mode": "L" if channels == 1 else "RGB",
    }


def _sniff_webp(data: bytes) -> dict[str, Any]:
    """Read WebP dimensions from the VP8 / VP8L / VP8X chunk."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return {}
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return {
            "width": width,
            "height": height,
            "channels": 4,
            "bits_per_channel": 8,
            "has_alpha": bool(data[20] & 0x10),
            "mode": "RGBA",
        }
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return {
            "width": (bits & 0x3FFF) + 1,
            "height": ((bits >> 14) & 0x3FFF) + 1,
            "channels": 4,
            "bits_per_channel": 8,
            "has_alpha": True,
            "mode": "RGBA",
        }
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return {
            "width": width & 0x3FFF,
            "height": height & 0x3FFF,
            "channels": 3,
            "bits_per_channel": 8,
            "has_alpha": False,
            "mode": "RGB",
        }
    return {}


def _sniff_tiff(data: bytes) -> dict[str, Any]:
    """Parse the first TIFF IFD for size, depth, channels and page count."""
    if len(data) < 8:
        return {}
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return {}
    try:
        if struct.unpack(endian + "H", data[2:4])[0] != 42:
            return {}
        offset = struct.unpack(endian + "I", data[4:8])[0]
        if offset + 2 > len(data):
            return {}
        entries = struct.unpack(endian + "H", data[offset : offset + 2])[0]
        tags: dict[int, int] = {}
        for index in range(entries):
            base = offset + 2 + index * 12
            if base + 12 > len(data):
                break
            tag, kind = struct.unpack(endian + "HH", data[base : base + 4])
            if kind == 3:  # SHORT
                tags[tag] = struct.unpack(endian + "H", data[base + 8 : base + 10])[0]
            elif kind == 4:  # LONG
                tags[tag] = struct.unpack(endian + "I", data[base + 8 : base + 12])[0]
    except struct.error:
        return {}

    width = tags.get(256)
    height = tags.get(257)
    if not width or not height:
        return {}
    samples = tags.get(277, 1)
    depth = tags.get(258, 8)
    next_ifd = None
    try:
        next_offset_position = offset + 2 + entries * 12
        if next_offset_position + 4 <= len(data):
            next_ifd = struct.unpack(
                endian + "I", data[next_offset_position : next_offset_position + 4]
            )[0]
    except struct.error:
        next_ifd = None
    return {
        "width": width,
        "height": height,
        "channels": samples,
        "bits_per_channel": depth,
        "has_alpha": False,
        "mode": "L" if samples == 1 else "RGB",
        "has_more_pages": bool(next_ifd),
    }


def _sniff_svg(data: bytes) -> dict[str, Any]:
    """Read the width/height (or viewBox) of an SVG document."""
    text = data[:4096].decode("utf-8", errors="replace")
    if "<svg" not in text:
        return {}
    viewbox = re.search(
        r'viewBox\s*=\s*["\']\s*([-\d.eE]+)[\s,]+([-\d.eE]+)[\s,]+'
        r"([-\d.eE]+)[\s,]+([-\d.eE]+)",
        text,
    )
    width = re.search(r'width\s*=\s*["\']([\d.]+)', text)
    height = re.search(r'height\s*=\s*["\']([\d.]+)', text)
    if width and height:
        return {
            "width": int(float(width.group(1))),
            "height": int(float(height.group(1))),
            "channels": 0,
            "bits_per_channel": 0,
            "mode": "SVG",
            "vector": True,
        }
    if viewbox:
        return {
            "width": int(float(viewbox.group(3))),
            "height": int(float(viewbox.group(4))),
            "channels": 0,
            "bits_per_channel": 0,
            "mode": "SVG",
            "vector": True,
        }
    return {"width": 0, "height": 0, "mode": "SVG", "vector": True}


_HEADER_READERS = {
    "PNG": _sniff_png,
    "JPEG": _sniff_jpeg,
    "GIF": _sniff_gif,
    "BMP": _sniff_bmp,
    "WEBP": _sniff_webp,
    "TIFF": _sniff_tiff,
    "SVG": _sniff_svg,
}


def _detect_format_from_bytes(data: bytes) -> str:
    """Identify the container format from the first bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:2] == b"\xff\xd8":
        return "JPEG"
    if data[:3] == b"GIF":
        return "GIF"
    if data[:2] == b"BM":
        return "BMP"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[:5] == b"%PDF-":
        return "PDF"
    if data[:4] == b"8BPS":
        return "PSD"
    head = data[:512].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<svg"):
        return "SVG"
    if data[:6] == b"GRIB":
        return "GRIB"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageInfo:
    """What a file is, independent of whether it could be rendered.

    Attributes
        path:
            Source path.
        format:
            Container format (``"PNG"``, ``"TIFF"``, ``"SVG"`` ...).
        mode:
            Pillow mode (``"L"``, ``"RGB"``, ``"I;16"`` ...) or a best guess
            from the header sniffer.
        width, height:
            Pixel dimensions.
        n_frames:
            Number of frames/pages (``1`` for ordinary images).
        is_animated:
            ``True`` for multi-frame GIF/TIFF or an ``has_more_pages`` TIFF.
        bits_per_channel:
            Bit depth per channel, when known.
        channels:
            Number of colour channels, when known.
        dpi:
            Resolution as a ``(x, y)`` tuple when recorded.
        has_alpha, has_icc_profile, has_exif:
            Format feature flags.
        is_vector:
            ``True`` for SVG, which has no intrinsic raster resolution.
        file_size, sha256:
            Size and content hash for provenance.
        renderer:
            ``"pillow"`` or ``"header"`` -- how this information was obtained.
        preview_available:
            Whether pixel-level preview can be produced.
        warnings:
            Anything that limits interpretation.
    """

    path: str
    format: str
    mode: str
    width: int
    height: int
    n_frames: int
    is_animated: bool
    bits_per_channel: int | None
    channels: int | None
    dpi: tuple[float, float] | None
    has_alpha: bool
    has_icc_profile: bool
    has_exif: bool
    is_vector: bool
    file_size: int
    sha256: str
    renderer: str
    preview_available: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        """Image area in megapixels."""
        return self.width * self.height / 1e6

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height (``0.0`` when the height is unknown)."""
        return self.width / self.height if self.height else 0.0

    @property
    def is_multichannel_stack(self) -> bool:
        """Whether the file holds more than one channel of a single image."""
        return self.n_frames > 1 and (self.channels or 0) <= 2

    def to_text(self) -> str:
        """Render a human-readable description of the file."""
        lines = [
            f"文件:     {self.path}",
            f"格式:     {self.format}  (读取方式: {self.renderer})",
            f"尺寸:     {self.width} x {self.height} px  "
            f"({self.megapixels:.2f} MP, 宽高比 {self.aspect_ratio:.3f})",
            f"模式:     {self.mode}",
        ]
        if self.bits_per_channel:
            lines.append(
                f"位深:     {self.bits_per_channel} bit/通道"
                + (f"，{self.channels} 通道" if self.channels else "")
            )
        if self.n_frames > 1:
            suffix = "（动画/堆栈）" if self.is_animated else ""
            lines.append(f"帧数:     {self.n_frames}{suffix}")
        if self.dpi:
            lines.append(f"分辨率:   {self.dpi[0]:g} x {self.dpi[1]:g} dpi")
        features = []
        if self.has_alpha:
            features.append("透明通道")
        if self.has_icc_profile:
            features.append("ICC 色彩配置")
        if self.has_exif:
            features.append("EXIF 元数据")
        if self.is_vector:
            features.append("矢量图")
        lines.append("特性:     " + ("、".join(features) if features else "无"))
        lines.append(f"大小:     {self.file_size:,} bytes")
        lines.append(f"SHA256:   {self.sha256[:16]}…")
        if self.is_vector:
            lines.append(
                "提示:     矢量图没有固定像素分辨率，需要渲染器才能生成位图预览。"
            )
        if not self.preview_available:
            lines.append("提示:     当前环境无法生成像素预览，仅提供元数据。")
        for warning in self.warnings:
            lines.append(f"[警告] {warning}")
        return "\n".join(lines)


def _pillow_info(path: Path, data: bytes, digest: str) -> ImageInfo | None:
    """Build an :class:`ImageInfo` using Pillow, or return ``None``."""
    if not _PILLOW_AVAILABLE:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            n_frames = getattr(image, "n_frames", 1)
            dpi_raw = image.info.get("dpi")
            dpi = (
                (float(dpi_raw[0]), float(dpi_raw[1]))
                if isinstance(dpi_raw, (tuple, list)) and len(dpi_raw) == 2
                else None
            )
            exif: dict[str, Any] = {}
            try:
                exif = dict(image.getexif())
            except Exception:
                exif = {}
            mode = image.mode
            bits = _bits_for_mode(mode)
            return ImageInfo(
                path=str(path),
                format=(image.format or _detect_format_from_bytes(data)),
                mode=mode,
                width=int(image.width),
                height=int(image.height),
                n_frames=int(n_frames),
                is_animated=bool(getattr(image, "is_animated", False)) or n_frames > 1,
                bits_per_channel=bits,
                channels=len(image.getbands()),
                dpi=dpi,
                has_alpha=mode in ("RGBA", "LA", "PA") or "transparency" in image.info,
                has_icc_profile=bool(image.info.get("icc_profile")),
                has_exif=bool(exif),
                is_vector=False,
                file_size=len(data),
                sha256=digest,
                renderer="pillow",
                preview_available=True,
                warnings=[],
            )
    except Exception:
        # Any Pillow failure (unknown container, missing codec, truncated
        # file) falls through to the header sniffer, which can still report
        # the container and dimensions.
        return None


def _bits_for_mode(mode: str) -> int | None:
    """Return the bit depth implied by a Pillow mode."""
    mapping = {
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "CMYK": 8,
        "YCbCr": 8,
        "LAB": 8,
        "HSV": 8,
        "I": 32,
        "I;16": 16,
        "I;16B": 16,
        "I;16L": 16,
        "I;16N": 16,
        "F": 32,
        "LA": 8,
        "PA": 8,
        "La": 8,
    }
    return mapping.get(mode)


def inspect_image(path: str | Path) -> ImageInfo:
    """Describe an image file without decoding more than necessary.

    Tries Pillow first and falls back to header parsing, so the function
    succeeds for any file whose container can be recognised even when Pillow
    is absent or the specific codec is missing.

    Parameters
        path:
            Image file to inspect.

    Returns
        An :class:`ImageInfo`.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the file is empty or its container cannot be recognised.
    """
    import hashlib

    path = Path(path)
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Image file '{path}' is empty.")
    digest = hashlib.sha256(data).hexdigest()

    info = _pillow_info(path, data, digest)
    if info is not None:
        return info

    detected = _detect_format_from_bytes(data)
    reader = _HEADER_READERS.get(detected)
    if reader is None:
        raise ValueError(
            f"'{path}' is not a recognised image (detected container: "
            f"{detected}). Supported raster formats: "
            + ", ".join(sorted(name for name in _HEADER_SNIFFERS if name != "SVG"))
            + "; vector: SVG."
        )

    details = reader(data)
    warnings: list[str] = []
    if not _PILLOW_AVAILABLE:
        warnings.append(
            "Pillow 未安装，只读取了文件头元数据，无法生成像素预览"
            f"（{_PILLOW_ERROR}）。安装方式：pip install 'bioeng-qpcr[gui]'"
        )
    elif not details:
        warnings.append("文件头解析失败，尺寸信息可能不准确。")

    width = int(details.get("width", 0))
    height = int(details.get("height", 0))
    if width <= 0 or height <= 0:
        warnings.append("无法从文件头确定图像尺寸。")

    return ImageInfo(
        path=str(path),
        format=detected,
        mode=str(details.get("mode", "unknown")),
        width=width,
        height=height,
        n_frames=int(
            details.get(
                "n_frames",
                2 if detected == "TIFF" and details.get("has_more_pages") else 1,
            )
        ),
        is_animated=bool(
            details.get("is_animated")
            or (detected == "TIFF" and details.get("has_more_pages"))
        ),
        bits_per_channel=details.get("bits_per_channel"),
        channels=details.get("channels"),
        dpi=None,
        has_alpha=bool(details.get("has_alpha", False)),
        has_icc_profile=False,
        has_exif=False,
        is_vector=bool(details.get("vector", False)),
        file_size=len(data),
        sha256=digest,
        renderer="header",
        preview_available=False,
        warnings=warnings,
    )


def list_supported_formats() -> dict[str, str]:
    """Return every recognised format mapped to a short description.

    Formats from the built-in header sniffer are always listed; the extra
    formats Pillow can additionally decode are merged in when it is present.
    """
    formats = {
        name: description
        for name, description in _PILLOW_FORMATS.items()
        if _PILLOW_AVAILABLE
    }
    for name in _HEADER_SNIFFERS:
        formats.setdefault(name, "内置文件头解析（无像素访问）")
    return dict(sorted(formats.items()))


@dataclass(frozen=True)
class ImagePreview:
    """A rendered, display-ready thumbnail plus how it was produced.

    Attributes
        info:
            Metadata of the source file.
        thumbnail:
            ``uint8`` RGB array, shape ``(H, W, 3)``.
        thumbnail_size:
            ``(width, height)`` of :attr:`thumbnail`.
        data_uri:
            Base64 PNG data URI, or ``None`` when not requested.
        palette:
            Pseudo-colour table applied to single-channel data.
        frame:
            Frame index that was rendered.
        enhancements:
            Human-readable list of the transforms applied.
        warnings:
            Interpretation caveats.
    """

    info: ImageInfo
    thumbnail: np.ndarray
    thumbnail_size: tuple[int, int]
    data_uri: str | None
    palette: str
    frame: int
    enhancements: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Render the metadata plus the applied enhancement list."""
        lines = [self.info.to_text(), ""]
        width, height = self.thumbnail_size
        lines.append(f"预览:     {width} x {height} px")
        lines.append(f"帧:       {self.frame + 1} / {self.info.n_frames}")
        lines.append(
            "处理:     "
            + ("、".join(self.enhancements) if self.enhancements else "仅缩放")
        )
        for warning in self.warnings:
            lines.append(f"[提示] {warning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pixel processing
# ---------------------------------------------------------------------------


def _apply_palette(gray: np.ndarray, palette: str) -> np.ndarray:
    """Map a normalised single-channel image through a pseudo-colour LUT.

    Parameters
        gray:
            Float array in ``[0, 1]``.
        palette:
            Key into :data:`CHANNEL_PALETTES`.

    Returns
        ``uint8`` RGB array with the same leading shape.

    Raises
        ValueError:
            If *palette* is unknown.
    """
    if palette not in CHANNEL_PALETTES:
        raise ValueError(
            f"Unknown palette {palette!r}; available: "
            + ", ".join(sorted(CHANNEL_PALETTES))
        )
    stops = CHANNEL_PALETTES[palette]
    positions = np.array([stop[0] for stop in stops], dtype=float)
    colours = np.array([stop[1] for stop in stops], dtype=float)
    flat = np.clip(gray, 0.0, 1.0).ravel()
    rgb = np.empty((flat.size, 3), dtype=float)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, positions, colours[:, channel])
    return rgb.reshape((*gray.shape, 3)).astype(np.uint8)


def _to_gray_array(array: np.ndarray) -> np.ndarray:
    """Reduce an arbitrary image array to a float single-channel image.

    Keeps the full numeric range of 16-bit and float data instead of scaling
    to 8 bits, because normalisation happens later and doing it twice loses
    precision.
    """
    if array.ndim == 2:
        return array.astype(float)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0].astype(float)
    if array.ndim == 3 and array.shape[2] >= 3:
        weights = np.array([0.299, 0.587, 0.114])
        return array[:, :, :3].astype(float) @ weights
    return array.astype(float)


def _normalise(
    gray: np.ndarray,
    *,
    auto_contrast: bool,
    percentiles: tuple[float, float],
    gamma: float,
    invert: bool,
) -> np.ndarray:
    """Scale *gray* into ``[0, 1]`` with optional contrast clipping."""
    if auto_contrast and gray.size:
        low, high = np.percentile(gray, percentiles)
    else:
        low, high = (
            float(np.min(gray)) if gray.size else 0.0,
            float(np.max(gray)) if gray.size else 1.0,
        )
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = 0.0, max(float(np.max(gray)) if gray.size else 1.0, 1.0)
    scaled = np.clip((gray - low) / (high - low), 0.0, 1.0)
    if gamma != 1.0:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}.")
        scaled = np.power(scaled, 1.0 / gamma)
    if invert:
        scaled = 1.0 - scaled
    return scaled


def _array_from_pillow(path: Path, frame: int, max_size: tuple[int, int]) -> np.ndarray:
    """Load one frame of an image as a numpy array.

    JPEG files are decoded at a reduced scale when possible (``draft``), which
    turns a multi-second load of a 50 MP photograph into a fast one.
    """
    assert Image is not None
    if any(limit <= 0 for limit in max_size):
        raise ValueError(
            f"max_size must contain positive dimensions, got {max_size!r}."
        )
    with Image.open(path) as image:
        if frame:
            image.seek(frame)
        pixels = int(image.width) * int(image.height)
        if pixels > _MAX_PREVIEW_PIXELS:
            raise ValueError(
                f"Image has {pixels:,} pixels, exceeding the "
                f"{_MAX_PREVIEW_PIXELS:,}-pixel preview safety limit. "
                "Metadata is still available."
            )
        if image.format == "JPEG":
            image.draft("RGB", max_size)
        resampling = getattr(Image, "Resampling", Image).NEAREST
        image.thumbnail(max_size, resampling)
        converted = image.convert("I;16") if image.mode in ("I;16", "I") else image
        return np.asarray(converted).copy()


def load_preview(
    path: str | Path,
    *,
    max_size: tuple[int, int] = (640, 640),
    frame: int = 0,
    invert: bool = False,
    auto_contrast: bool = True,
    contrast_percentiles: tuple[float, float] = (0.5, 99.5),
    gamma: float = 1.0,
    palette: str = "gray",
    as_data_uri: bool = True,
) -> ImagePreview:
    """Produce a displayable preview of an image.

    Parameters
        path:
            Image file.
        max_size:
            Bounding box for the preview; the thumbnail is scaled down to fit
            while preserving the aspect ratio.  Never upscales.
        frame:
            Frame index for multi-page TIFFs and animated GIFs.
        invert:
            Invert intensities -- the standard trick for reading a
            fluorescence scan without wasting toner.
        auto_contrast:
            Clip to *contrast_percentiles* before scaling, which is what makes
            a dim gel band visible.
        contrast_percentiles:
            ``(low, high)`` percentiles used as the black and white points.
        gamma:
            Gamma applied after scaling (``> 1`` brightens mid-tones).
        palette:
            Pseudo-colour LUT for single-channel data; ignored for RGB.
        as_data_uri:
            Also return a base64 PNG ``data:`` URI for HTML embedding.

    Returns
        An :class:`ImagePreview`.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the file is not an image, the frame index is out of range, the
            palette is unknown, or no pixel backend is available.
    """
    path = Path(path)
    info = inspect_image(path)

    if palette not in CHANNEL_PALETTES:
        raise ValueError(
            f"Unknown palette {palette!r}; available: "
            + ", ".join(sorted(CHANNEL_PALETTES))
        )
    if info.is_vector:
        raise ValueError(
            f"'{path.name}' is a vector image ({info.format}); rasterising it "
            "requires a renderer, so no pixel preview is available. Metadata "
            "is still reported."
        )
    if not info.preview_available:
        raise ValueError(
            f"No pixel backend is available for '{path.name}': "
            + (" ".join(info.warnings) or "unknown reason")
        )
    if frame < 0 or frame >= info.n_frames:
        raise ValueError(
            f"frame {frame} is out of range; '{path.name}' has "
            f"{info.n_frames} frame(s)."
        )

    array = _array_from_pillow(path, frame, max_size)
    rendered_is_rgb = array.ndim == 3 and array.shape[2] >= 3 and not invert
    if rendered_is_rgb:
        normalised = _normalise(
            array[:, :, :3].astype(float),
            auto_contrast=auto_contrast,
            percentiles=contrast_percentiles,
            gamma=gamma,
            invert=False,
        )
        rgb = np.rint(normalised * 255).astype(np.uint8)
    else:
        normalised = _normalise(
            _to_gray_array(array),
            auto_contrast=auto_contrast,
            percentiles=contrast_percentiles,
            gamma=gamma,
            invert=invert,
        )
        rgb = _apply_palette(normalised, palette)

    thumbnail = _resize_nearest(rgb, max_size)
    enhancements: list[str] = []
    if invert:
        enhancements.append("invert")
    if auto_contrast:
        enhancements.append(
            f"auto_contrast({contrast_percentiles[0]:g}-{contrast_percentiles[1]:g}%)"
        )
    if gamma != 1.0:
        enhancements.append(f"gamma={gamma:g}")
    if not rendered_is_rgb:
        enhancements.append(f"palette={palette}")

    data_uri = to_data_uri(thumbnail) if as_data_uri else None
    warnings = list(info.warnings)
    if not rendered_is_rgb:
        warnings.append(
            "单通道数据已按伪彩表渲染；定量分析请使用原始像素值，不要从预览图反推强度。"
        )
    return ImagePreview(
        info=info,
        thumbnail=thumbnail,
        thumbnail_size=(thumbnail.shape[1], thumbnail.shape[0]),
        data_uri=data_uri,
        palette=palette,
        frame=frame,
        enhancements=tuple(enhancements),
        warnings=warnings,
    )


def _resize_nearest(rgb: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    """Downscale *rgb* to fit *max_size* using integer-strided sampling.

    Implemented with array slicing rather than Pillow's resampling so it works
    identically for 8-bit and 16-bit source data and has no dependency beyond
    NumPy.  Stride sampling (rather than interpolation) preserves the sharp
    edges that matter when inspecting a gel band.
    """
    height, width = rgb.shape[:2]
    limit_w, limit_h = max_size
    if width <= limit_w and height <= limit_h:
        return rgb
    scale = min(limit_w / width, limit_h / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    rows = (np.arange(new_h) * (height / new_h)).astype(int).clip(0, height - 1)
    cols = (np.arange(new_w) * (width / new_w)).astype(int).clip(0, width - 1)
    return rgb[rows][:, cols]


def to_data_uri(rgb: np.ndarray, *, fmt: str = "PNG") -> str:
    """Encode an ``uint8`` RGB array as a base64 ``data:`` URI.

    Parameters
        rgb:
            ``uint8`` array of shape ``(H, W, 3)`` or ``(H, W)``.
        fmt:
            Pillow format name.

    Returns
        A ``data:image/...;base64,...`` string.

    Raises
        ValueError:
            If Pillow is unavailable or the array is not a valid image shape.
    """
    if not _PILLOW_AVAILABLE:
        raise ValueError(
            "Encoding a data URI requires Pillow (pip install 'bioeng-qpcr[gui]')."
        )
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 array, got {array.dtype}.")
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}.")

    assert Image is not None
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "png" if fmt.upper() == "PNG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


def save_preview_png(preview: ImagePreview, path: str | Path) -> Path:
    """Write a preview thumbnail to a PNG file.

    Parameters
        preview:
            Preview to persist.
        path:
            Destination file.

    Returns
        The path written.

    Raises
        ValueError:
            If Pillow is unavailable.
    """
    if not _PILLOW_AVAILABLE:
        raise ValueError(
            "Saving a preview requires Pillow (pip install 'bioeng-qpcr[gui]')."
        )
    path = Path(path)
    assert Image is not None
    Image.fromarray(preview.thumbnail).save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Channel handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelImage:
    """One single-channel image ready to be merged into a composite.

    Attributes
        path:
            Source file.
        index:
            Channel index within a multi-frame file (``0`` for single images).
        label:
            Display name (defaults to the file stem).
        colour:
            Palette key used when rendering this channel.
    """

    path: str
    index: int = 0
    label: str = ""
    colour: str = "green"


def split_channels(
    path: str | Path, *, labels: Iterable[str] | None = None
) -> list[ChannelImage]:
    """Describe the channels of a multi-frame or multi-page image file.

    Parameters
        path:
            Source file (typically a multi-page TIFF).
        labels:
            Optional names for the channels, in frame order.

    Returns
        One :class:`ChannelImage` per frame, with colours assigned from a
        default fluorescence palette cycle.

    Raises
        FileNotFoundError:
            If *path* does not exist.
        ValueError:
            If the file has a single frame (nothing to split).
    """
    info = inspect_image(path)
    if info.n_frames < 2:
        raise ValueError(
            f"'{Path(path).name}' has a single frame; there is nothing to split."
        )
    cycle = ["green", "magenta", "cyan", "red", "blue", "yellow"]
    names = list(labels) if labels is not None else []
    channels: list[ChannelImage] = []
    for index in range(info.n_frames):
        label = names[index] if index < len(names) else f"channel{index + 1}"
        channels.append(
            ChannelImage(
                path=str(path),
                index=index,
                label=label,
                colour=cycle[index % len(cycle)],
            )
        )
    return channels


def compose_channels(
    channels: list[ChannelImage],
    *,
    max_size: tuple[int, int] = (640, 640),
    auto_contrast: bool = True,
    contrast_percentiles: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Merge single-channel images into one RGB composite.

    Each channel is normalised independently (so a dim channel is not lost
    under a bright one) and mapped through its own pseudo-colour table; the
    results are combined with a maximum projection, the usual display choice
    for multi-colour fluorescence.

    Parameters
        channels:
            Channels to merge, in draw order.
        max_size:
            Bounding box for the output composite.
        auto_contrast:
            Apply per-channel percentile stretching.
        contrast_percentiles:
            Percentiles used as the black and white points per channel.

    Returns
        A ``uint8`` RGB array.

    Raises
        ValueError:
            If *channels* is empty, the channels disagree on dimensions, or
            Pillow is unavailable.
    """
    if not channels:
        raise ValueError("At least one channel is required to compose a composite.")
    if not _PILLOW_AVAILABLE:
        raise ValueError(
            "Composing channels requires Pillow (pip install 'bioeng-qpcr[gui]')."
        )

    source_sizes: list[tuple[int, int]] = []
    planes: list[np.ndarray] = []
    for channel in channels:
        info = inspect_image(channel.path)
        source_sizes.append((info.width, info.height))
        array = _array_from_pillow(Path(channel.path), channel.index, max_size)
        gray = _to_gray_array(array)
        normalised = _normalise(
            gray,
            auto_contrast=auto_contrast,
            percentiles=contrast_percentiles,
            gamma=1.0,
            invert=False,
        )
        planes.append(_apply_palette(normalised, channel.colour).astype(np.uint8))

    if len(set(source_sizes)) != 1:
        raise ValueError(
            "All channels must have identical dimensions to be composed, got "
            f"{sorted(set(source_sizes))}."
        )
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError("Channels produced different preview dimensions.")
    composite = np.maximum.reduce(planes)
    return _resize_nearest(composite, max_size)
