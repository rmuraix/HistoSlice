"""Drawing tile coordinates/text on top of a (thumbnail) image."""

from __future__ import annotations

import html
from typing import Optional, Union

import numpy as np
import pyvips

from ._check import check_image
from ._imageio import _vips_to_array, has_svg_support

ERROR_TEXT_ITEM_LENGTH = (
    "Length of text items ({}) does not match length of coordinates ({})."
)

FONT_PROBE_SIZE = 64
DPI = 72

_COLOR_NAMES = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def _divide_xywh(
    xywh: tuple[int, int, int, int], divisor: Union[float, tuple[float, float]]
) -> tuple[int, int, int, int]:
    """Divide xywh-coordinates by a divisor."""
    if not isinstance(divisor, (tuple, list)):
        divisor = (divisor, divisor)
    w_div, h_div = divisor
    x, y, w, h = xywh
    return round(x / w_div), round(y / h_div), round(w / w_div), round(h / h_div)


def get_annotated_image(
    image: np.ndarray,
    coordinates: list[tuple[int, int, int, int]],
    downsample: Union[float, tuple[float, float]],
    *,
    rectangle_outline: str = "red",
    rectangle_fill: Optional[str] = None,
    rectangle_width: int = 1,
    highlight_first: bool = False,
    highlight_outline: str = "blue",
    text_items: Optional[list[str]] = None,
    text_color: str = "black",
    text_proportion: float = 0.75,
    text_font: str = "monospace",
    alpha: float = 0.0,
) -> np.ndarray:
    """Function to draw tiles to an image. Useful for visualising tiles/predictions.

    Args:
        image: Image to draw to.
        coordinates: Tile coordinates.
        downsample: Downsample for the image. If coordinates are from the same image,
            set this to 1.0.
        rectangle_outline: Outline color of each tile, as a CSS/SVG color name (e.g.
            "red") or hex code. Defaults to "red".
        rectangle_fill: Fill color of each tile. Defaults to None.
        rectangle_width: Width of each tile edges. Defaults to 1.
        highlight_first: Highlight first tile, useful when tiles overlap.
            Defaults to False.
        highlight_outline: Highlight color for the first tile. Defaults to "black".
        text_items: Text items for each tile. Length must match `coordinates`.
            Defaults to None.
        text_color: Text color. Defaults to "black".
        text_proportion: Proportion of space the text takes in each tile.
            Defaults to 0.75.
        text_font: Font family passed to `libvips`'s text renderer (resolved through
            `fontconfig`). Defaults to "monospace".
        alpha: Alpha value for blending the original image and drawn image.
            Defaults to 0.0.

    Raises:
        ValueError: Text item length does not match length of coordinates.

    Returns:
        Annotated image.
    """
    image = check_image(image)
    if image.ndim == 2:  # noqa
        image = np.repeat(image[:, :, None], 3, axis=2)
    if text_items is not None:
        if len(text_items) != len(coordinates):
            raise ValueError(
                ERROR_TEXT_ITEM_LENGTH.format(len(text_items), len(coordinates))
            )
    else:
        text_items = [None] * len(coordinates)

    boxes = [_divide_xywh(xywh, downsample) for xywh in coordinates]
    font_size = _resolve_font_size(
        boxes=boxes,
        text_items=text_items,
        text_proportion=text_proportion,
        font=text_font,
    )
    draw = _draw_with_svg if has_svg_support() else _draw_with_vips_ops
    annotated = draw(
        image=image,
        boxes=boxes,
        text_items=text_items,
        rectangle_outline=rectangle_outline,
        rectangle_fill=rectangle_fill,
        rectangle_width=rectangle_width,
        highlight_first=highlight_first,
        highlight_outline=highlight_outline,
        text_color=text_color,
        text_font=text_font,
        font_size=font_size,
    )
    # Blend.
    if alpha <= 0.0:
        return annotated
    if alpha >= 1.0:
        return image
    blended = (
        annotated.astype(np.float32) * (1.0 - alpha) + image.astype(np.float32) * alpha
    )
    return blended.round().astype(np.uint8)


def _resolve_font_size(
    *,
    boxes: list[tuple[int, int, int, int]],
    text_items: list[Optional[str]],
    text_proportion: float,
    font: str,
) -> Optional[int]:
    """Pick a single font size (in points) fitting `text_proportion` of the first
    annotated tile's width, reused for every text item."""
    first = next(
        ((box, text) for box, text in zip(boxes, text_items) if text is not None),
        None,
    )
    if first is None:
        return None
    max_length = max(3, max(len(str(x)) for x in text_items))
    (__, __, w, __), __ = first
    target_width = text_proportion * w / max_length
    probe = pyvips.Image.text("W", font=f"{font} {FONT_PROBE_SIZE}", dpi=DPI)
    ref_width = max(probe.width, 1)
    return max(1, round(FONT_PROBE_SIZE * target_width / ref_width))


def _svg_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    outline: Optional[str],
    fill: Optional[str],
    width: int,
) -> str:
    fill_attr = html.escape(fill) if fill else "none"
    outline_attr = html.escape(outline) if outline else "none"
    return (
        f'<rect x="{x}" y="{y}" width="{max(w, 0)}" height="{max(h, 0)}" '
        f'fill="{fill_attr}" stroke="{outline_attr}" stroke-width="{width}"/>'
    )


def _svg_text(x: int, y: int, *, text: str, color: str, font: str, size: int) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{html.escape(font)}" '
        f'font-size="{size}" fill="{html.escape(color)}" '
        f'dominant-baseline="hanging">{html.escape(str(text))}</text>'
    )


def _draw_with_svg(
    *,
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    text_items: list[Optional[str]],
    rectangle_outline: Optional[str],
    rectangle_fill: Optional[str],
    rectangle_width: int,
    highlight_first: bool,
    highlight_outline: str,
    text_color: str,
    text_font: str,
    font_size: Optional[int],
) -> np.ndarray:
    """Draw all rectangles/text as a single SVG overlay, rasterise it once, and
    composite it onto `image` in one shot (avoids repeated whole-image copies)."""
    height, width = image.shape[:2]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    ]
    for (x, y, w, h), text in zip(boxes, text_items):
        parts.append(
            _svg_rect(
                x,
                y,
                w,
                h,
                outline=rectangle_outline,
                fill=rectangle_fill,
                width=rectangle_width,
            )
        )
        if text is not None and font_size is not None:
            parts.append(
                _svg_text(
                    x + rectangle_width,
                    y + rectangle_width,
                    text=text,
                    color=text_color,
                    font=text_font,
                    size=font_size,
                )
            )
    if highlight_first and boxes:
        x, y, w, h = boxes[0]
        parts.append(
            _svg_rect(
                x,
                y,
                w,
                h,
                outline=highlight_outline,
                fill=rectangle_fill,
                width=rectangle_width,
            )
        )
    parts.append("</svg>")
    overlay = pyvips.Image.svgload_buffer("".join(parts).encode("utf-8"))
    base = pyvips.Image.new_from_array(image)
    composed = base.composite2(overlay, "over").flatten()
    return _vips_to_array(composed)


def _draw_with_vips_ops(
    *,
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    text_items: list[Optional[str]],
    rectangle_outline: Optional[str],
    rectangle_fill: Optional[str],
    rectangle_width: int,
    highlight_first: bool,
    highlight_outline: str,
    text_color: str,
    text_font: str,
    font_size: Optional[int],
) -> np.ndarray:
    """Fallback used when `libvips` was built without SVG (`librsvg`) support. Slower
    (each draw call copies the image) but has no extra dependencies."""
    canvas = pyvips.Image.new_from_array(image)
    for (x, y, w, h), text in zip(boxes, text_items):
        canvas = _draw_box(
            canvas,
            x,
            y,
            w,
            h,
            outline=rectangle_outline,
            fill=rectangle_fill,
            width=rectangle_width,
        )
        if text is not None and font_size is not None:
            canvas = _draw_text(
                canvas,
                x + rectangle_width,
                y + rectangle_width,
                text=text,
                color=text_color,
                font=text_font,
                size=font_size,
            )
    if highlight_first and boxes:
        x, y, w, h = boxes[0]
        canvas = _draw_box(
            canvas,
            x,
            y,
            w,
            h,
            outline=highlight_outline,
            fill=rectangle_fill,
            width=rectangle_width,
        )
    return _vips_to_array(canvas)


def _draw_box(
    canvas: "pyvips.Image",
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    outline: Optional[str],
    fill: Optional[str],
    width: int,
) -> "pyvips.Image":
    if fill:
        canvas = canvas.draw_rect(
            list(_color_to_rgb(fill)), x, y, max(w, 1), max(h, 1), fill=True
        )
    if outline:
        ink = list(_color_to_rgb(outline))
        for offset in range(max(width, 1)):
            canvas = canvas.draw_rect(
                ink,
                x + offset,
                y + offset,
                max(w - 2 * offset, 1),
                max(h - 2 * offset, 1),
                fill=False,
            )
    return canvas


def _draw_text(
    canvas: "pyvips.Image",
    x: int,
    y: int,
    *,
    text: str,
    color: str,
    font: str,
    size: int,
) -> "pyvips.Image":
    glyphs = pyvips.Image.text(str(text), font=f"{font} {size}", rgba=True, dpi=DPI)
    solid = (
        (
            pyvips.Image.black(glyphs.width, glyphs.height, bands=3)
            + list(_color_to_rgb(color))
        )
        .cast("uchar")
        .copy(interpretation="srgb")
    )
    colored = solid.bandjoin(glyphs[3])
    return canvas.composite2(colored, "over", x=x, y=y)


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    """Resolve a CSS colour name or `#rrggbb`/`#rgb` hex code to an RGB triplet."""
    color = color.strip().lower()
    if color in _COLOR_NAMES:
        return _COLOR_NAMES[color]
    if color.startswith("#"):
        hex_digits = color[1:]
        if len(hex_digits) == 3:  # noqa
            hex_digits = "".join(c * 2 for c in hex_digits)
        if len(hex_digits) == 6:  # noqa
            return tuple(int(hex_digits[i : i + 2], 16) for i in (0, 2, 4))
    return _COLOR_NAMES["black"]
