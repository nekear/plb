from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, shift

from plb.utils import PROJECT_ROOT

DEFAULT_FONT_PATH = PROJECT_ROOT / "assets" / "NotoSansJP-Regular.ttf"


def _render_glyph(
    character: str,
    font_path: str,
    canvas_size: int = 256,
) -> np.ndarray:
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=round(canvas_size * 0.75))

    bounding_box = draw.textbbox((0, 0), character, font=font)
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]
    x = (canvas_size - width) / 2 - bounding_box[0]
    y = (canvas_size - height) / 2 - bounding_box[1]

    draw.text((x, y), character, font=font, fill=0)
    return np.asarray(image)


def _normalize_glyph(
    array: np.ndarray,
    target_size: int = 64,
    padding: int = 6,
) -> np.ndarray:
    ink_mask = array < 250
    if not ink_mask.any():
        return np.zeros((target_size, target_size), dtype=np.float32)

    ys, xs = np.where(ink_mask)
    crop = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    crop_height, crop_width = crop.shape
    usable_size = target_size - 2 * padding
    scale = min(usable_size / crop_width, usable_size / crop_height)
    new_width = max(1, round(crop_width * scale))
    new_height = max(1, round(crop_height * scale))

    resized = Image.fromarray(crop).resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (target_size, target_size), 255)
    canvas.paste(
        resized,
        (
            (target_size - new_width) // 2,
            (target_size - new_height) // 2,
        ),
    )

    return 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0


@lru_cache(maxsize=10_000)
def _normalized_glyph(character: str, font_path: str) -> np.ndarray:
    glyph = _normalize_glyph(_render_glyph(character, font_path))
    glyph.setflags(write=False)
    return glyph


def _soft_dice(
    array_1: np.ndarray,
    array_2: np.ndarray,
    blur_sigma: float = 1.0,
) -> float:
    if blur_sigma > 0:
        array_1 = gaussian_filter(array_1, sigma=blur_sigma)
        array_2 = gaussian_filter(array_2, sigma=blur_sigma)

    denominator = array_1.sum() + array_2.sum()
    if denominator == 0:
        return 0.0

    intersection = np.minimum(array_1, array_2).sum()
    return float(2.0 * intersection / denominator)


def _edge_map(array: np.ndarray) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(array)
    edges = np.hypot(gradient_x, gradient_y)
    maximum = edges.max()
    if maximum > 0:
        edges /= maximum
    return edges


def _chamfer_similarity(
    array_1: np.ndarray,
    array_2: np.ndarray,
    edge_threshold: float = 0.12,
    tolerance: float = 1.5,
) -> float:
    edges_1 = _edge_map(array_1) > edge_threshold
    edges_2 = _edge_map(array_2) > edge_threshold
    if not edges_1.any() or not edges_2.any():
        return 0.0

    distance_to_1 = distance_transform_edt(~edges_1)
    distance_to_2 = distance_transform_edt(~edges_2)
    similarity_1_to_2 = np.exp(
        -(distance_to_2[edges_1] ** 2) / (2.0 * tolerance**2)
    ).mean()
    similarity_2_to_1 = np.exp(
        -(distance_to_1[edges_2] ** 2) / (2.0 * tolerance**2)
    ).mean()
    return float(0.5 * (similarity_1_to_2 + similarity_2_to_1))


def _ink_balance(array_1: np.ndarray, array_2: np.ndarray) -> float:
    ink_1 = float(array_1.sum())
    ink_2 = float(array_2.sum())
    if ink_1 == 0 or ink_2 == 0:
        return 0.0
    return min(ink_1, ink_2) / max(ink_1, ink_2)


def _affine_variant(
    array: np.ndarray,
    angle: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    height, width = array.shape
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255))
    scaled_width = max(1, round(width * scale_x))
    scaled_height = max(1, round(height * scale_y))
    scaled = image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BICUBIC,
    )

    canvas = Image.new("L", (width, height), 0)
    canvas.paste(
        scaled,
        ((width - scaled_width) // 2, (height - scaled_height) // 2),
    )
    rotated = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=0,
    )
    return np.asarray(rotated, dtype=np.float32) / 255.0


def _score_glyph_candidate(
    glyph_1: np.ndarray,
    glyph_2: np.ndarray,
    complexity: float,
    blur_sigma: float = 1.0,
    edge_tolerance: float = 1.5,
) -> float:
    overlap = _soft_dice(glyph_1, glyph_2, blur_sigma=blur_sigma)
    edges = _chamfer_similarity(
        glyph_1,
        glyph_2,
        tolerance=edge_tolerance,
    )
    structural = 0.60 * overlap + 0.40 * edges
    return float(structural * complexity**0.9)


@lru_cache(maxsize=100_000)
def _directional_affine_similarity(
    character_1: str,
    character_2: str,
    font_path: str,
) -> float:
    """Transforms ``character_2`` towards ``character_1`` and returns the best score"""
    glyph_1 = _normalized_glyph(character_1, font_path)
    glyph_2 = _normalized_glyph(character_2, font_path)
    complexity = _ink_balance(glyph_1, glyph_2)
    best_score = 0.0
    max_shift = 2

    for angle in range(-20, 21, 5):
        for scale_x in (0.95, 1.0, 1.05):
            for scale_y in (0.95, 1.0, 1.05):
                transformed = _affine_variant(
                    glyph_2,
                    angle=angle,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )

                for dy in range(-max_shift, max_shift + 1):
                    for dx in range(-max_shift, max_shift + 1):
                        candidate = shift(
                            transformed,
                            shift=(dy, dx),
                            order=1,
                            mode="constant",
                            cval=0.0,
                            prefilter=False,
                        )
                        score = _score_glyph_candidate(
                            glyph_1,
                            candidate,
                            complexity=complexity,
                        )
                        transform_cost = (
                            (angle / 20.0) ** 2
                            + ((scale_x - 1.0) / 0.05) ** 2
                            + ((scale_y - 1.0) / 0.05) ** 2
                            + (dx / max_shift) ** 2
                            + (dy / max_shift) ** 2
                        )
                        score *= float(np.exp(-0.08 * transform_cost))
                        best_score = max(best_score, score)

    return float(best_score)


@lru_cache(maxsize=100_000)
def _aligned_similarity(
    character_1: str,
    character_2: str,
    font_path: str,
) -> float:
    glyph_1 = _normalized_glyph(character_1, font_path)
    glyph_2 = _normalized_glyph(character_2, font_path)
    complexity = _ink_balance(glyph_1, glyph_2)
    best_score = 0.0
    max_shift = 2

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            candidate = shift(
                glyph_2,
                shift=(dy, dx),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            score = _score_glyph_candidate(
                glyph_1,
                candidate,
                complexity=complexity,
            )
            transform_cost = (dx / max_shift) ** 2 + (dy / max_shift) ** 2
            score *= float(np.exp(-0.08 * transform_cost))
            best_score = max(best_score, score)

    return float(best_score)


@lru_cache(maxsize=100_000)
def _symmetric_pixel_similarity(
    character_1: str,
    character_2: str,
    font_path: str,
) -> float:
    aligned = _aligned_similarity(character_1, character_2, font_path)
    forward = _directional_affine_similarity(
        character_1,
        character_2,
        font_path,
    )
    reverse = _directional_affine_similarity(
        character_2,
        character_1,
        font_path,
    )
    directional_mean = 0.5 * (forward + reverse)
    return max(0.0, min(1.0, max(aligned, directional_mean)))


def compute_pixel_similarity(
    character_1: str,
    character_2: str,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> float:
    """Returns symmetric affine pixel similarity for two rendered characters"""
    if len(character_1) != 1 or len(character_2) != 1:
        raise ValueError("Both inputs must be single characters.")
    if character_1 == character_2:
        return 1.0

    first, second = sorted((character_1, character_2))
    resolved_font_path = str(Path(font_path).resolve())
    return _symmetric_pixel_similarity(first, second, resolved_font_path)
