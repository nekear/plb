import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    lexemes_df = pd.read_json("data/interim/lexemes.jsonl", lines=True)

    lexemes_df.head()
    return (lexemes_df,)


@app.cell
def _(lexemes_df):
    japanese_lexemes_df = lexemes_df[lexemes_df["language"] == "ja"]
    japanese_lexemes_df.head()
    return (japanese_lexemes_df,)


@app.cell
def _(japanese_lexemes_df):
    import re

    KANA_PATTERN = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]+$")

    def is_kana(text: str) -> bool:
        return bool(KANA_PATTERN.match(text))

    kana_count = japanese_lexemes_df["form"].apply(lambda x: is_kana(x)).sum()
    total_count = len(japanese_lexemes_df.index)

    print(
        f"{kana_count} kanas out of {total_count} ({(kana_count / total_count) * 100:.1f}%)"
    )
    return is_kana, total_count


@app.cell
def _():
    from functools import lru_cache

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from scipy.ndimage import (
        distance_transform_edt,
        gaussian_filter,
        shift,
    )

    def render_glyph(
        char: str,
        font_path: str,
        canvas_size: int = 256,
    ) -> np.ndarray:
        """Renders one glyph as an anti-aliased grayscale image."""
        image = Image.new("L", (canvas_size, canvas_size), 255)
        draw = ImageDraw.Draw(image)

        # Rendering large, then reducing later
        font = ImageFont.truetype(
            font_path,
            size=round(canvas_size * 0.75),
        )

        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x = (canvas_size - width) / 2 - bbox[0]
        y = (canvas_size - height) / 2 - bbox[1]

        draw.text((x, y), char, font=font, fill=0)
        return np.asarray(image)

    def normalize_glyph(
        arr: np.ndarray,
        target_size: int = 64,
        padding: int = 6,
    ) -> np.ndarray:
        """
        Crops the glyph, preserve its aspect ratio, and returns ink intensity
        in the range 0.0 to 1.0.
        """
        # Including anti-aliased edge pixels in the bounding box
        ink_mask = arr < 250

        if not ink_mask.any():
            return np.zeros((target_size, target_size), dtype=np.float32)

        ys, xs = np.where(ink_mask)
        crop = arr[
            ys.min() : ys.max() + 1,
            xs.min() : xs.max() + 1,
        ]

        crop_height, crop_width = crop.shape
        usable_size = target_size - 2 * padding

        # Preserve the original aspect ratio.
        scale = min(
            usable_size / crop_width,
            usable_size / crop_height,
        )

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

        # Converting white background/black ink into 0 background/1 ink
        return 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0

    @lru_cache(maxsize=10_000)
    def normalized_glyph(
        char: str,
        font_path: str,
        target_size: int = 64,
        padding: int = 6,
    ) -> np.ndarray:
        """Render and normalize a glyph once per character configuration."""
        glyph = normalize_glyph(
            render_glyph(char, font_path),
            target_size=target_size,
            padding=padding,
        )
        glyph.setflags(write=False)
        return glyph

    def soft_dice(
        arr1: np.ndarray,
        arr2: np.ndarray,
        blur_sigma: float = 1.0,
    ) -> float:
        if blur_sigma > 0:
            arr1 = gaussian_filter(arr1, sigma=blur_sigma)
            arr2 = gaussian_filter(arr2, sigma=blur_sigma)

        intersection = np.minimum(arr1, arr2).sum()
        denominator = arr1.sum() + arr2.sum()

        if denominator == 0:
            return 0.0

        return float(2.0 * intersection / denominator)

    def edge_map(arr: np.ndarray) -> np.ndarray:
        """Extracts a simple grayscale edge-strength map."""
        grad_y, grad_x = np.gradient(arr)
        edges = np.hypot(grad_x, grad_y)

        maximum = edges.max()
        if maximum > 0:
            edges /= maximum

        return edges

    def chamfer_similarity(
        arr1: np.ndarray,
        arr2: np.ndarray,
        edge_threshold: float = 0.12,
        tolerance: float = 1.5,
    ) -> float:
        """
        Compares glyph edges using symmetric Chamfer-style distance.

        An edge receives partial credit when the other glyph has an edge
        within a few pixels.
        """
        edges1 = edge_map(arr1) > edge_threshold
        edges2 = edge_map(arr2) > edge_threshold

        if not edges1.any() or not edges2.any():
            return 0.0

        # At each pixel, distance to the nearest edge in the other glyph
        distance_to_1 = distance_transform_edt(~edges1)
        distance_to_2 = distance_transform_edt(~edges2)

        similarity_1_to_2 = np.exp(
            -(distance_to_2[edges1] ** 2) / (2.0 * tolerance**2)
        ).mean()

        similarity_2_to_1 = np.exp(
            -(distance_to_1[edges2] ** 2) / (2.0 * tolerance**2)
        ).mean()

        return float(0.5 * (similarity_1_to_2 + similarity_2_to_1))

    def ink_balance(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        Penalizes comparisons between a simple glyph and a much more
        complicated glyph.
        """
        ink1 = float(arr1.sum())
        ink2 = float(arr2.sum())

        if ink1 == 0 or ink2 == 0:
            return 0.0

        return min(ink1, ink2) / max(ink1, ink2)

    def affine_variant(
        arr: np.ndarray,
        angle: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> np.ndarray:
        """
        Applies centered anisotropic scaling and rotation.

        Input and output values are floating-point ink intensities from 0 to 1
        """
        height, width = arr.shape

        image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))

        scaled_width = max(1, round(width * scale_x))
        scaled_height = max(1, round(height * scale_y))

        scaled = image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new("L", (width, height), 0)

        x = (width - scaled_width) // 2
        y = (height - scaled_height) // 2
        canvas.paste(scaled, (x, y))

        rotated = canvas.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=0,
        )

        return np.asarray(rotated, dtype=np.float32) / 255.0

    def compute_visual_similarity(
        char1: str,
        char2: str,
        font_path: str,
        target_size: int = 64,
        padding: int = 6,
        blur_sigma: float = 1.0,
        edge_tolerance: float = 1.5,
        max_shift: int = 2,
        return_details: bool = False,
    ):
        glyph1 = normalize_glyph(
            render_glyph(char1, font_path),
            target_size=target_size,
            padding=padding,
        )

        glyph2 = normalize_glyph(
            render_glyph(char2, font_path),
            target_size=target_size,
            padding=padding,
        )

        complexity = ink_balance(glyph1, glyph2)

        best = {
            "score": 0.0,
            "overlap": 0.0,
            "edge_similarity": 0.0,
            "ink_balance": complexity,
            "shift": (0, 0),
        }

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                shifted = shift(
                    glyph2,
                    shift=(dy, dx),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )

                overlap = soft_dice(
                    glyph1,
                    shifted,
                    blur_sigma=blur_sigma,
                )

                structure = chamfer_similarity(
                    glyph1,
                    shifted,
                    tolerance=edge_tolerance,
                )

                # Overlap remains the primary factor.
                # Structure and complexity act as penalties.
                score = overlap * np.sqrt(structure * complexity)

                if score > best["score"]:
                    best = {
                        "score": float(score),
                        "overlap": float(overlap),
                        "edge_similarity": float(structure),
                        "ink_balance": float(complexity),
                        "shift": (dy, dx),
                    }

        if return_details:
            return best

        return best["score"]

    def compute_affine_similarity(
        glyph1: np.ndarray,
        glyph2: np.ndarray,
        angles=range(-20, 21, 5),
        scales=(0.95, 1.0, 1.05),
        max_shift: int = 2,
        blur_sigma: float = 1.0,
        edge_tolerance: float = 1.5,
    ) -> dict:
        complexity = ink_balance(glyph1, glyph2)

        best = {
            "score": 0.0,
            "overlap": 0.0,
            "edges": 0.0,
            "ink": float(complexity),
            "angle": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shift": (0, 0),
            "transform_penalty": 1.0,
        }

        shift_denominator = max(max_shift, 1)

        for angle in angles:
            for scale_x in scales:
                for scale_y in scales:
                    transformed = affine_variant(
                        glyph2,
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

                            components = score_glyph_candidate(
                                glyph1,
                                candidate,
                                complexity=complexity,
                                blur_sigma=blur_sigma,
                                edge_tolerance=edge_tolerance,
                            )

                            transform_cost = (
                                (angle / 20.0) ** 2
                                + ((scale_x - 1.0) / 0.05) ** 2
                                + ((scale_y - 1.0) / 0.05) ** 2
                                + (dx / shift_denominator) ** 2
                                + (dy / shift_denominator) ** 2
                            )

                            penalty = float(np.exp(-0.08 * transform_cost))

                            score = components["score"] * penalty

                            if score > best["score"]:
                                best = {
                                    **components,
                                    "score": float(score),
                                    "angle": float(angle),
                                    "scale_x": float(scale_x),
                                    "scale_y": float(scale_y),
                                    "shift": (dy, dx),
                                    "transform_penalty": penalty,
                                }

        return best

    @lru_cache(maxsize=100_000)
    def compute_character_affine_similarity(
        char1: str,
        char2: str,
        font_path: str,
    ) -> dict:
        """Computes and caches the directional transform ``char2 -> char1``."""
        return compute_affine_similarity(
            normalized_glyph(char1, font_path),
            normalized_glyph(char2, font_path),
        )

    @lru_cache(maxsize=100_000)
    def compute_character_aligned_similarity(
        char1: str,
        char2: str,
        font_path: str,
    ) -> dict:
        """Computes and caches aligned similarity for a canonicalized pair."""
        return compute_aligned_similarity(
            normalized_glyph(char1, font_path),
            normalized_glyph(char2, font_path),
        )

    def compare_glyphs(
        char1: str,
        char2: str,
        font_path: str,
    ) -> dict:
        aligned_char1, aligned_char2 = sorted((char1, char2))
        aligned = compute_character_aligned_similarity(
            aligned_char1,
            aligned_char2,
            font_path,
        )

        # Affine matching is directional because only the second glyph is
        # transformed. Evaluate both directions before combining the scores.
        forward_affine = compute_character_affine_similarity(
            char1,
            char2,
            font_path,
        )
        reverse_affine = compute_character_affine_similarity(
            char2,
            char1,
            font_path,
        )

        directional_affine_mean = 0.5 * (
            forward_affine["score"] + reverse_affine["score"]
        )
        symmetric_affine = max(
            aligned["score"],
            directional_affine_mean,
        )

        return {
            "aligned_similarity": aligned["score"],
            "forward_affine_similarity": forward_affine["score"],
            "reverse_affine_similarity": reverse_affine["score"],
            "directional_affine_mean": directional_affine_mean,
            "affine_similarity": symmetric_affine,
            "affine_gain": symmetric_affine - aligned["score"],
            "aligned_details": aligned,
            "forward_affine_details": forward_affine,
            "reverse_affine_details": reverse_affine,
        }

    def score_glyph_candidate(
        glyph1: np.ndarray,
        glyph2: np.ndarray,
        complexity: float,
        blur_sigma: float = 1.0,
        edge_tolerance: float = 1.5,
    ) -> dict:
        overlap = soft_dice(
            glyph1,
            glyph2,
            blur_sigma=blur_sigma,
        )

        edges = chamfer_similarity(
            glyph1,
            glyph2,
            tolerance=edge_tolerance,
        )

        structural = 0.60 * overlap + 0.40 * edges

        score = structural * complexity**0.9

        return {
            "score": float(score),
            "overlap": float(overlap),
            "edges": float(edges),
            "ink": float(complexity),
        }

    def compute_aligned_similarity(
        glyph1: np.ndarray,
        glyph2: np.ndarray,
        max_shift: int = 2,
        blur_sigma: float = 1.0,
        edge_tolerance: float = 1.5,
    ) -> dict:
        complexity = ink_balance(glyph1, glyph2)

        best = {
            "score": 0.0,
            "overlap": 0.0,
            "edges": 0.0,
            "ink": float(complexity),
            "shift": (0, 0),
            "transform_penalty": 1.0,
        }

        shift_denominator = max(max_shift, 1)

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                candidate = shift(
                    glyph2,
                    shift=(dy, dx),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )

                components = score_glyph_candidate(
                    glyph1,
                    candidate,
                    complexity=complexity,
                    blur_sigma=blur_sigma,
                    edge_tolerance=edge_tolerance,
                )

                transform_cost = (dx / shift_denominator) ** 2 + (
                    dy / shift_denominator
                ) ** 2

                penalty = float(np.exp(-0.08 * transform_cost))

                score = components["score"] * penalty

                if score > best["score"]:
                    best = {
                        **components,
                        "score": float(score),
                        "shift": (dy, dx),
                        "transform_penalty": penalty,
                    }

        return best

    font_path = "assets/NotoSansJP-Regular.ttf"

    pairs = [
        ("シ", "ツ"),
        ("ソ", "ン"),
        ("ク", "ワ"),
        ("ア", "マ"),
        ("ス", "ヌ"),
        ("テ", "ラ"),
        ("ソ", "リ"),
        ("は", "ほ"),
        ("ね", "れ"),
        ("る", "ろ"),
        ("さ", "き"),
        ("い", "り"),
        ("つ", "う"),
        ("土", "士"),
        ("幸", "爪"),
        ("微", "入"),
        ("千", "干"),
        ("未", "末"),
    ]

    for char1, char2 in pairs:
        result = compare_glyphs(
            char1,
            char2,
            font_path,
        )

        forward = result["forward_affine_details"]
        reverse = result["reverse_affine_details"]

        print(
            f"{char1} / {char2}: "
            f"aligned={result['aligned_similarity']:.2f}, "
            f"forward={result['forward_affine_similarity']:.2f} "
            f"[{char2}->{char1}: angle={forward['angle']:+.0f}°, "
            f"scale=({forward['scale_x']:.2f}, {forward['scale_y']:.2f}), "
            f"shift={forward['shift']}, penalty={forward['transform_penalty']:.2f}], "
            f"reverse={result['reverse_affine_similarity']:.2f} "
            f"[{char1}->{char2}: angle={reverse['angle']:+.0f}°, "
            f"scale=({reverse['scale_x']:.2f}, {reverse['scale_y']:.2f}), "
            f"shift={reverse['shift']}, penalty={reverse['transform_penalty']:.2f}], "
            f"symmetric={result['affine_similarity']:.2f}, "
            f"gain={result['affine_gain']:+.2f}, "
        )

    print(f"Normalized glyph cache: {normalized_glyph.cache_info()}")
    print(
        f"Directional affine cache: {compute_character_affine_similarity.cache_info()}"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perceptual-distance studies usually cover the basic kana but often omit dakuten (が, ざ), handakuten (ぱ), small/yōon kana (ゃ, ッ), and the long-vowel mark (ー). Therefore, I'm checking the coverage on the dataset.
    """)
    return


@app.cell
def _(is_kana, japanese_lexemes_df, total_count):
    from plb.esm.kana_confusability import KanaConfusability

    confusability = KanaConfusability(
        xlsx_path="data/raw/esm/12144_2021_2664_MOESM1_ESM.xlsx"
    )

    def check_char_coverage(form: str) -> bool:
        try:
            for char in form:
                if is_kana(char):
                    confusability.compare(
                        char, "そ"
                    )  # Checking with a single symbol for simplicity, assuming that all permutations in ESM exist
                else:
                    continue

            return True
        except:
            return False

    kanas_not_in_esm = japanese_lexemes_df["form"].apply(check_char_coverage)

    print(
        f"Coverage of kanas in ESM based on our dataset: {kanas_not_in_esm.sum()} / {total_count} ({(kanas_not_in_esm.sum() / total_count) * 100:.1f}%)"
    )

    print("Some examples include:")
    print(list(japanese_lexemes_df[kanas_not_in_esm]["form"].iloc[0:40]))
    return


if __name__ == "__main__":
    app.run()
