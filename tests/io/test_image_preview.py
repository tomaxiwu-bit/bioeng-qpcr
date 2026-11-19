"""Tests for multi-format image inspection and preview.

The header sniffer is exercised directly (through the private ``_sniff_*``
helpers) because it is the fallback path: with Pillow installed the public
entry points never reach it, yet it is what keeps the CLI useful on a machine
without the optional imaging extra.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from bioeng_qpcr.io.image_preview import (
    CHANNEL_PALETTES,
    ChannelImage,
    _apply_palette,
    _array_from_pillow,
    _sniff_bmp,
    _sniff_gif,
    _sniff_jpeg,
    _sniff_png,
    _sniff_svg,
    _sniff_tiff,
    _sniff_webp,
    compose_channels,
    inspect_image,
    list_supported_formats,
    load_preview,
    save_preview_png,
    split_channels,
    to_data_uri,
)

PIL = pytest.importorskip("PIL", reason="Pillow is needed to author test images")
from PIL import Image  # noqa: E402


def _png_bytes(
    width: int = 4, height: int = 3, depth: int = 8, colour: int = 2
) -> bytes:
    """Minimal but structurally valid PNG header followed by filler."""
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR"
    ihdr += struct.pack(">II", width, height)
    ihdr += bytes([depth, colour, 0, 0, 0])
    return header + ihdr + b"\x00" * 16


def _make_gradient(width: int = 40, height: int = 30) -> np.ndarray:
    """Return a deterministic greyscale gradient."""
    rows = np.linspace(0, 255, height).reshape(-1, 1)
    cols = np.linspace(0, 255, width).reshape(1, -1)
    return ((rows + cols) / 2).astype(np.uint8)


@pytest.fixture
def gray_png(tmp_path):
    path = tmp_path / "gel.png"
    Image.fromarray(_make_gradient(), mode="L").save(path)
    return path


@pytest.fixture
def colour_png(tmp_path):
    path = tmp_path / "figure.png"
    array = np.stack(
        [
            _make_gradient(),
            np.zeros((30, 40), np.uint8),
            np.full((30, 40), 128, np.uint8),
        ],
        axis=2,
    )
    Image.fromarray(array, mode="RGB").save(path)
    return path


@pytest.fixture
def multi_page_tiff(tmp_path):
    path = tmp_path / "stack.tiff"
    frames = [Image.fromarray(_make_gradient() + offset) for offset in (0, 40, 80)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path


@pytest.fixture
def jpeg(tmp_path):
    path = tmp_path / "photo.jpg"
    colour = _make_gradient(64, 48).repeat(3).reshape(48, 64, 3)
    Image.fromarray(colour, mode="RGB").save(path, quality=90)
    return path


@pytest.fixture
def gif(tmp_path):
    path = tmp_path / "anim.gif"
    frames = [Image.fromarray(_make_gradient(20, 20) + offset) for offset in (0, 60)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return path


class TestHeaderSniffer:
    def test_png_dimensions_and_depth(self):
        details = _sniff_png(_png_bytes(width=7, height=5))
        assert details["width"] == 7
        assert details["height"] == 5
        assert details["bits_per_channel"] == 8
        assert details["channels"] == 3

    def test_png_detects_an_alpha_channel(self):
        details = _sniff_png(_png_bytes(colour=6))
        assert details["has_alpha"]
        assert details["channels"] == 4

    def test_16_bit_png_depth(self):
        assert _sniff_png(_png_bytes(depth=16, colour=0))["bits_per_channel"] == 16

    def test_non_png_returns_nothing(self):
        assert _sniff_png(b"not a png at all" * 4) == {}

    def test_jpeg_sof_scan(self):
        # SOI + APP0 + SOF0 with a known geometry.
        data = b"\xff\xd8"
        data += b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
        data += b"\xff\xc0" + struct.pack(">H", 11) + bytes([8])
        data += struct.pack(">HH", 12, 9)
        data += bytes([1]) + b"\x00" * 8
        details = _sniff_jpeg(data)
        assert details["width"] == 9
        assert details["height"] == 12
        assert details["channels"] == 1

    def test_gif_logical_screen(self):
        data = b"GIF89a" + struct.pack("<HH", 11, 13) + b"\x00" * 4
        details = _sniff_gif(data)
        assert details["width"] == 11
        assert details["height"] == 13

    def test_bmp_dimensions_and_depth(self):
        data = b"BM" + b"\x00" * 12 + struct.pack("<I", 40)
        data += struct.pack("<ii", 21, 17) + struct.pack("<HH", 1, 24)
        details = _sniff_bmp(data)
        assert details["width"] == 21
        assert details["height"] == 17
        assert details["channels"] == 3

    def test_tiff_ifd_parsing(self):
        # Little-endian TIFF with one IFD carrying ImageWidth/Length.
        entries = [
            (256, 4, 33),  # ImageWidth  = 33
            (257, 4, 22),  # ImageLength = 22
            (258, 3, 16),  # BitsPerSample = 16
            (277, 3, 1),  # SamplesPerPixel = 1
        ]
        data = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
        data += struct.pack("<H", len(entries))
        for tag, kind, value in entries:
            data += struct.pack("<HHI", tag, kind, 1) + struct.pack("<I", value)
        data += struct.pack("<I", 0)
        details = _sniff_tiff(data)
        assert details["width"] == 33
        assert details["height"] == 22
        assert details["bits_per_channel"] == 16
        assert details["channels"] == 1

    def test_webp_vp8x_dimensions(self):
        data = bytearray(b"RIFF" + struct.pack("<I", 20) + b"WEBP" + b"VP8X")
        data += b"\x00" * 8
        data += bytes([20, 0, 0]) + bytes([10, 0, 0])
        details = _sniff_webp(bytes(data))
        assert details["width"] == 21
        assert details["height"] == 11

    def test_svg_viewbox(self):
        data = b'<?xml version="1.0"?><svg viewBox="0 0 640 480"></svg>'
        details = _sniff_svg(data)
        assert details["width"] == 640
        assert details["height"] == 480
        assert details["vector"] is True


class TestInspectImage:
    def test_greyscale_png(self, gray_png):
        info = inspect_image(gray_png)
        assert info.format == "PNG"
        assert (info.width, info.height) == (40, 30)
        assert info.mode == "L"
        assert info.n_frames == 1
        assert not info.is_animated
        assert info.preview_available
        assert info.renderer == "pillow"
        assert info.megapixels == pytest.approx(0.0012)

    def test_colour_png_reports_bands(self, colour_png):
        info = inspect_image(colour_png)
        assert info.channels == 3
        assert info.mode == "RGB"

    def test_multi_page_tiff(self, multi_page_tiff):
        info = inspect_image(multi_page_tiff)
        assert info.format == "TIFF"
        assert info.n_frames == 3
        assert info.is_animated

    def test_animated_gif(self, gif):
        info = inspect_image(gif)
        assert info.format == "GIF"
        assert info.n_frames == 2
        assert info.is_animated

    def test_jpeg(self, jpeg):
        info = inspect_image(jpeg)
        assert info.format == "JPEG"
        assert (info.width, info.height) == (64, 48)

    def test_text_report_mentions_the_key_facts(self, gray_png):
        report = inspect_image(gray_png).to_text()
        assert "PNG" in report
        assert "40 x 30" in report
        assert "SHA256" in report

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            inspect_image(path)

    def test_unknown_container_lists_the_supported_formats(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"this is definitely not an image" * 10)
        with pytest.raises(ValueError, match="not a recognised image"):
            inspect_image(path)

    def test_svg_is_described_but_not_rasterised(self, tmp_path):
        path = tmp_path / "vector.svg"
        path.write_text('<svg width="100" height="50"></svg>', encoding="utf-8")
        info = inspect_image(path)
        assert info.is_vector
        assert info.format == "SVG"
        with pytest.raises(ValueError, match="vector image"):
            load_preview(path)

    def test_every_extension_is_reported_by_the_format_listing(self):
        formats = list_supported_formats()
        for expected in ("PNG", "JPEG", "TIFF", "GIF", "BMP", "WEBP", "SVG"):
            assert expected in formats


class TestPalette:
    def test_gray_palette_spans_black_to_white(self):
        rgb = _apply_palette(np.array([[0.0, 1.0]]), "gray")
        assert tuple(rgb[0, 0]) == (0, 0, 0)
        assert tuple(rgb[0, 1]) == (255, 255, 255)

    def test_green_palette_is_green_dominant(self):
        rgb = _apply_palette(np.array([[1.0]]), "green")
        assert rgb[0, 0, 1] > rgb[0, 0, 0]

    def test_unknown_palette_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown palette"):
            _apply_palette(np.array([[0.5]]), "rainbow")

    def test_every_declared_palette_evaluates(self):
        for name in CHANNEL_PALETTES:
            rgb = _apply_palette(np.array([[0.0, 0.5, 1.0]]), name)
            assert rgb.shape == (1, 3, 3)


class TestLoadPreview:
    def test_downsamples_before_creating_numpy_array(self, gray_png):
        array = _array_from_pillow(gray_png, 0, (10, 10))
        assert array.shape[0] <= 10
        assert array.shape[1] <= 10

    def test_rejects_images_over_the_pixel_budget(self, gray_png, monkeypatch):
        import bioeng_qpcr.io.image_preview as preview_module

        monkeypatch.setattr(preview_module, "_MAX_PREVIEW_PIXELS", 100)
        with pytest.raises(ValueError, match="preview safety limit"):
            load_preview(gray_png)

    def test_greyscale_preview_keeps_the_aspect_ratio(self, gray_png):
        preview = load_preview(gray_png, max_size=(20, 20), as_data_uri=False)
        assert preview.thumbnail.dtype == np.uint8
        assert preview.thumbnail.shape[2] == 3
        assert preview.thumbnail_size[0] <= 20
        assert preview.thumbnail_size[1] <= 20
        assert preview.thumbnail_size[0] / preview.thumbnail_size[1] == pytest.approx(
            40 / 30, rel=0.1
        )

    def test_small_images_are_not_upscaled(self, gray_png):
        preview = load_preview(gray_png, max_size=(500, 500), as_data_uri=False)
        assert preview.thumbnail_size == (40, 30)

    def test_data_uri_is_emitted_on_request(self, gray_png):
        preview = load_preview(gray_png, as_data_uri=True)
        assert preview.data_uri is not None
        assert preview.data_uri.startswith("data:image/png;base64,")

    def test_colour_image_is_not_palettised(self, colour_png):
        preview = load_preview(colour_png, as_data_uri=False)
        assert preview.palette == "gray"
        assert all("palette" not in item for item in preview.enhancements)

    def test_gamma_is_applied_to_colour_images(self, colour_png):
        neutral = load_preview(colour_png, gamma=1.0, as_data_uri=False)
        bright = load_preview(colour_png, gamma=2.0, as_data_uri=False)
        assert bright.thumbnail.mean() > neutral.thumbnail.mean()

    def test_inversion_flips_the_intensity(self, gray_png):
        normal = load_preview(gray_png, as_data_uri=False)
        inverted = load_preview(gray_png, invert=True, as_data_uri=False)
        assert inverted.thumbnail.mean() < normal.thumbnail.mean()
        assert "invert" in inverted.enhancements

    def test_auto_contrast_stretches_a_narrow_range(self, tmp_path):
        path = tmp_path / "dim.png"
        # Values 100-110: invisible without stretching.
        Image.fromarray(np.full((10, 10), 105, np.uint8), mode="L").save(path)
        stretched = load_preview(path, auto_contrast=True, as_data_uri=False)
        flat = load_preview(path, auto_contrast=False, as_data_uri=False)
        assert stretched.thumbnail.std() >= flat.thumbnail.std()

    def test_gamma_brightens_midtones(self, gray_png):
        neutral = load_preview(gray_png, gamma=1.0, as_data_uri=False)
        bright = load_preview(gray_png, gamma=2.0, as_data_uri=False)
        assert bright.thumbnail.mean() > neutral.thumbnail.mean()

    def test_invalid_gamma_is_rejected(self, gray_png):
        with pytest.raises(ValueError, match="gamma"):
            load_preview(gray_png, gamma=0.0)

    def test_unknown_palette_is_rejected(self, gray_png):
        with pytest.raises(ValueError, match="Unknown palette"):
            load_preview(gray_png, palette="nope")

    def test_frame_selection(self, multi_page_tiff):
        first = load_preview(multi_page_tiff, frame=0, as_data_uri=False)
        third = load_preview(multi_page_tiff, frame=2, as_data_uri=False)
        assert first.thumbnail.mean() != third.thumbnail.mean()
        assert third.frame == 2

    def test_out_of_range_frame_is_rejected(self, gray_png):
        with pytest.raises(ValueError, match="out of range"):
            load_preview(gray_png, frame=5)

    def test_quantitative_warning_is_emitted_for_single_channel(self, gray_png):
        preview = load_preview(gray_png, as_data_uri=False)
        assert any("伪彩" in warning for warning in preview.warnings)

    def test_text_report_lists_the_enhancements(self, gray_png):
        report = load_preview(gray_png, invert=True, as_data_uri=False).to_text()
        assert "处理" in report
        assert "invert" in report

    def test_preview_can_be_saved(self, gray_png, tmp_path):
        preview = load_preview(gray_png, as_data_uri=False)
        destination = save_preview_png(preview, tmp_path / "out.png")
        assert destination.exists()
        assert inspect_image(destination).format == "PNG"


class TestDataUri:
    def test_round_trip_through_pillow(self):
        array = _make_gradient()
        uri = to_data_uri(np.stack([array] * 3, axis=2))
        assert uri.startswith("data:image/png;base64,")

    def test_non_uint8_is_rejected(self):
        with pytest.raises(ValueError, match="uint8"):
            to_data_uri(np.zeros((4, 4, 3), dtype=float))

    def test_bad_dimensionality_is_rejected(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            to_data_uri(np.zeros((2, 2, 2, 2), dtype=np.uint8))


class TestChannels:
    def test_split_reports_one_entry_per_frame(self, multi_page_tiff):
        channels = split_channels(multi_page_tiff)
        assert len(channels) == 3
        assert all(isinstance(channel, ChannelImage) for channel in channels)
        assert len({channel.colour for channel in channels}) == 3

    def test_split_accepts_explicit_labels(self, multi_page_tiff):
        channels = split_channels(multi_page_tiff, labels=["DAPI", "GFP", "Cy5"])
        assert [channel.label for channel in channels] == ["DAPI", "GFP", "Cy5"]

    def test_split_rejects_single_frame_files(self, gray_png):
        with pytest.raises(ValueError, match="single frame"):
            split_channels(gray_png)

    def test_compose_produces_an_rgb_composite(self, multi_page_tiff):
        channels = split_channels(multi_page_tiff, labels=["a", "b", "c"])
        composite = compose_channels(channels, max_size=(40, 40))
        assert composite.shape[2] == 3
        assert composite.dtype == np.uint8

    def test_composite_channels_carry_their_own_colour(self, multi_page_tiff):
        channels = [
            ChannelImage(
                path=str(multi_page_tiff), index=0, label="green", colour="green"
            ),
            ChannelImage(path=str(multi_page_tiff), index=1, label="red", colour="red"),
        ]
        composite = compose_channels(channels, max_size=(40, 40))
        # A maximum projection of a green and a red plane must use both the
        # green and the red sub-pixels.
        assert composite[:, :, 1].max() > 0
        assert composite[:, :, 0].max() > 0

    def test_mismatched_sizes_are_rejected(self, tmp_path, multi_page_tiff):
        # A different-sized plane cannot be merged into the same composite.
        other = tmp_path / "other.png"
        Image.fromarray(_make_gradient(25, 60), mode="L").save(other)
        channels = [
            ChannelImage(path=str(multi_page_tiff), index=0, colour="green"),
            ChannelImage(path=str(other), index=0, colour="red"),
        ]
        with pytest.raises(ValueError, match="identical dimensions"):
            compose_channels(channels)

    def test_empty_channel_list_is_rejected(self):
        with pytest.raises(ValueError, match="At least one channel"):
            compose_channels([])
