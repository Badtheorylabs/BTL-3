#!/usr/bin/env python3
"""Generate public aggregate figures for the BTL-3 compression report."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

INK = "#101827"
MUTED = "#596579"
GRID = "#D9E0EA"
PAPER = "#F7F8FA"
NAVY = "#172B4D"
BLUE = "#356AE6"
TEAL = "#19A28C"
GOLD = "#E0A22B"
RED = "#D65858"
VIOLET = "#7457D9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), PAPER)
    return image, ImageDraw.Draw(image)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int,
          fill: str = INK, bold: bool = False) -> None:
    draw.text(xy, text, font=font(size, bold), fill=fill)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, optimize=True)


def byte_allocation() -> None:
    image, draw = canvas(1800, 860)
    label(draw, (90, 62), "Where the 8.38B GGUF tensor bytes went", 54, NAVY, True)
    label(
        draw,
        (90, 130),
        "The packed decoder dominates; behavior repair is 0.39% of tensor payload bytes.",
        28,
        MUTED,
    )
    parts = [
        ("AVQ2", 5_498_130_320, BLUE),
        ("Affine INT4", 919_347_200, TEAL),
        ("BF16 islands", 1_205_862_400, GOLD),
        ("Vocabulary", 667_713_856, VIOLET),
        ("Small state", 57_767_936, NAVY),
        ("Behavior adapter", 32_440_320, RED),
    ]
    total = sum(value for _, value, _ in parts)
    x0, x1, y0, y1 = 90, 1710, 240, 370
    cursor = x0
    for index, (_, value, color) in enumerate(parts):
        width = round((x1 - x0) * value / total)
        if index == len(parts) - 1:
            width = x1 - cursor
        draw.rectangle((cursor, y0, cursor + width, y1), fill=color)
        cursor += width
    legend_y = 450
    for index, (name, value, color) in enumerate(parts):
        col, row = index % 2, index // 2
        x = 110 + col * 820
        y = legend_y + row * 105
        draw.rounded_rectangle((x, y, x + 42, y + 42), radius=8, fill=color)
        label(draw, (x + 66, y - 4), name, 31, INK, True)
        percent = value / total * 100
        label(
            draw,
            (x + 390, y),
            f"{value / 1e9:.3f} GB  |  {percent:.2f}%",
            27,
            MUTED,
        )
    label(draw, (90, 790), "Tensor payloads: 8,381,262,032 bytes  |  Final GGUF: 8,392,369,600 bytes", 28, NAVY, True)
    save(image, "byte-allocation.png")


def retention() -> None:
    image, draw = canvas(1800, 1050)
    label(draw, (90, 58), "Fresh sealed tool-behavior retention", 54, NAVY, True)
    label(
        draw,
        (90, 128),
        "Conditional on teacher-correct turns; the aggregate hides one weak category.",
        28,
        MUTED,
    )
    values = [
        ("Single", 100, 20),
        ("Parallel", 100, 20),
        ("Sequential", 100, 20),
        ("Parallel-multiple", 30, 10),
        ("Abstention", 100, 20),
        ("Overall", 92.2, 90),
    ]
    x0, x1 = 540, 1680
    top, gap, bar_h = 230, 123, 58
    for tick in range(0, 101, 20):
        x = x0 + round((x1 - x0) * tick / 100)
        draw.line((x, 205, x, 920), fill=GRID, width=2)
        label(draw, (x - 18, 935), str(tick), 23, MUTED)
    for index, (name, value, denominator) in enumerate(values):
        y = top + index * gap
        color = RED if name == "Parallel-multiple" else (TEAL if name == "Overall" else BLUE)
        label(draw, (95, y + 6), name, 30, INK, name == "Overall")
        draw.rounded_rectangle((x0, y, x1, y + bar_h), radius=15, fill="#E5EAF1")
        end = x0 + round((x1 - x0) * value / 100)
        draw.rounded_rectangle((x0, y, end, y + bar_h), radius=15, fill=color)
        value_text = f"{value:.1f}%" if value % 1 else f"{int(value)}%"
        if value >= 90:
            label(draw, (end - 108, y + 9), value_text, 28, "white", True)
        else:
            label(draw, (end + 18, y + 9), value_text, 28, color, True)
        label(draw, (95, y + 46), f"teacher-correct n={denominator}", 20, MUTED)
    label(
        draw,
        (90, 1000),
        "Overall: 83 of 90 teacher-correct turns retained; absolute score 83/100.",
        26,
        NAVY,
        True,
    )
    save(image, "sealed-retention.png")


def pipeline() -> None:
    image, draw = canvas(1800, 1040)
    label(draw, (90, 55), "The behavior-first compression loop", 54, NAVY, True)
    label(
        draw,
        (90, 125),
        "Local quantization proposes a candidate; emitted behavior decides whether it survives.",
        28,
        MUTED,
    )
    boxes = [
        ("1", "Pin teacher", "Revision, adapter hash,\nseparate data splits", BLUE),
        ("2", "Fit one layer", "FP64 curvature, AVQ2\nor group-128 INT4", VIOLET),
        ("3", "Replay prefix", "Feed compressed hidden\nstates into the next layer", TEAL),
        ("4", "Gate behavior", "Calls, arguments, order,\nabstention, stopping", GOLD),
        ("5", "Localize cliff", "Bisect prefix and test\nmodule overrides", RED),
        ("6", "Spend bytes", "Keep only repairs that\nrestore emitted behavior", NAVY),
    ]
    positions = [(90, 240), (650, 240), (1210, 240), (1210, 610), (650, 610), (90, 610)]
    for (number, title, body, color), (x, y) in zip(boxes, positions):
        draw.rounded_rectangle((x, y, x + 500, y + 250), radius=26, fill="white", outline=color, width=5)
        draw.ellipse((x + 28, y + 28, x + 92, y + 92), fill=color)
        label(draw, (x + 50, y + 40), number, 27, "white", True)
        label(draw, (x + 116, y + 28), title, 34, INK, True)
        for line_index, line in enumerate(body.splitlines()):
            label(draw, (x + 42, y + 125 + line_index * 40), line, 27, MUTED)
    arrows = [
        ((590, 365), (650, 365)),
        ((1150, 365), (1210, 365)),
        ((1460, 490), (1460, 610)),
        ((1210, 735), (1150, 735)),
        ((650, 735), (590, 735)),
    ]
    for start, end in arrows:
        draw.line((*start, *end), fill=INK, width=5)
        ex, ey = end
        if end[0] > start[0]:
            points = [(ex, ey), (ex - 18, ey - 12), (ex - 18, ey + 12)]
        elif end[0] < start[0]:
            points = [(ex, ey), (ex + 18, ey - 12), (ex + 18, ey + 12)]
        else:
            points = [(ex, ey), (ex - 12, ey - 18), (ex + 12, ey - 18)]
        draw.polygon(points, fill=INK)
    draw.arc((15, 390, 280, 900), start=95, end=270, fill=INK, width=5)
    draw.polygon([(92, 420), (78, 448), (112, 444)], fill=INK)
    label(draw, (90, 950), "Repeat until all 64 layers pass, then freeze and package.", 28, NAVY, True)
    save(image, "behavior-first-loop.png")


def main() -> None:
    byte_allocation()
    retention()
    pipeline()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
