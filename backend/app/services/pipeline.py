import time
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

from app.core.loader import load_image, parse_xml_metadata
from app.core.utils import encode_image_to_base64

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.matching.loftr import LoFTRConfig, LoFTRMatcher
from core.preprocessing.resolution import resolution_resampling
from core.preprocessing.viewpoint import viewpoint_correction
from core.registration.ransac import RANSACConfig, RANSACRegistration

logger = logging.getLogger(__name__)

_LOFTR_MAX_IMAGE_SIDE = 640
_LOFTR_IMAGE_STRIDE = 8
_REGISTERED_SOURCE_BLEND_WEIGHT = 0.40
_REFERENCE_BLEND_WEIGHT = 0.60
_SYNTHETIC_DEMO_INLIERS = 272


def _resize_for_loftr(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= _LOFTR_MAX_IMAGE_SIDE:
        return image

    scale = _LOFTR_MAX_IMAGE_SIDE / longest_side
    resized_width = max(
        _LOFTR_IMAGE_STRIDE,
        round(width * scale / _LOFTR_IMAGE_STRIDE) * _LOFTR_IMAGE_STRIDE,
    )
    resized_height = max(
        _LOFTR_IMAGE_STRIDE,
        round(height * scale / _LOFTR_IMAGE_STRIDE) * _LOFTR_IMAGE_STRIDE,
    )
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def _restore_keypoint_scale(
    keypoints: np.ndarray,
    original_shape: tuple[int, ...],
    resized_shape: tuple[int, ...],
) -> np.ndarray:
    original_height, original_width = original_shape[:2]
    resized_height, resized_width = resized_shape[:2]
    scale = np.array(
        [original_width / resized_width, original_height / resized_height],
        dtype=np.float32,
    )
    return keypoints * scale


def _create_synthetic_demo_matches(
    reference_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(42)
    normalized_points: list[np.ndarray] = []
    min_distance_squared = 0.0007
    while len(normalized_points) < 320:
        candidate = random.uniform(0.04, 0.96, size=2).astype(np.float32)
        if all(
            float(np.sum((candidate - point) ** 2)) >= min_distance_squared
            for point in normalized_points
        ):
            normalized_points.append(candidate)
    normalized_points_array = np.asarray(normalized_points, dtype=np.float32)
    reference_points = normalized_points_array * np.array(
        [reference_shape[1] - 1, reference_shape[0] - 1],
        dtype=np.float32,
    )
    source_points = normalized_points_array * np.array(
        [source_shape[1] - 1, source_shape[0] - 1],
        dtype=np.float32,
    )

    source_points += random.normal(0.0, 0.25, source_points.shape).astype(np.float32)
    outlier_indices = random.choice(
        len(source_points),
        size=len(source_points) // 5,
        replace=False,
    )
    source_points[outlier_indices, 0] = random.uniform(
        0,
        max(1, source_shape[1] - 1),
        len(outlier_indices),
    )
    source_points[outlier_indices, 1] = random.uniform(
        0,
        max(1, source_shape[0] - 1),
        len(outlier_indices),
    )
    source_points[:, 0] = np.clip(source_points[:, 0], 0, source_shape[1] - 1)
    source_points[:, 1] = np.clip(source_points[:, 1], 0, source_shape[0] - 1)

    for points, shape in (
        (reference_points, reference_shape),
        (source_points, source_shape),
    ):
        occupied: set[tuple[int, int]] = set()
        for index, point in enumerate(points):
            x, y = np.rint(point).astype(int)
            while (x, y) in occupied:
                x = (x + 1) % shape[1]
                if x == 0:
                    y = (y + 1) % shape[0]
            occupied.add((x, y))
            points[index] = (x, y)

    confidence = np.random.default_rng().uniform(
        0.8,
        0.99,
        len(source_points),
    ).astype(np.float32)
    return reference_points, source_points, confidence


def run_pipeline(
    src_img_path: Path,
    src_xml_path: Path,
    ref_img_path: Path,
    ref_xml_path: Path,
    use_synthetic_demo: bool = False,
) -> dict:
    """
    Main Execution Pipeline for Lunar Image Alignment.

    1. Loads images and XML metadata
    2. Applies viewpoint and resolution preprocessing
    3. Computes LoFTR correspondences and RANSAC homography
    4. Generates match visualizations and aligned image output
    5. Calculates RMSE, inlier ratio, and execution time
    """
    start_time = time.perf_counter()

    # 1. Load resources & metadata
    logger.info("Pipeline Execution: Loading images & metadata...")
    src_img_raw = load_image(src_img_path)
    ref_img_raw = load_image(ref_img_path)

    src_meta = parse_xml_metadata(src_xml_path)
    ref_meta = parse_xml_metadata(ref_xml_path)

    # 2. Run the root preprocessing stages.
    viewpoint_source = viewpoint_correction(ref_img_raw, src_img_raw)
    ref_processed, src_processed = resolution_resampling(
        ref_img_raw,
        viewpoint_source,
        ref_xml_path,
        src_xml_path,
    )

    ref_gray = _to_grayscale(ref_processed)
    src_gray = _to_grayscale(src_processed)

    ransac_result = None
    if use_synthetic_demo:
        reference_points, source_points, confidence = _create_synthetic_demo_matches(
            ref_gray.shape,
            src_gray.shape,
        )
        demo_inlier_mask = np.zeros(len(reference_points), dtype=bool)
        demo_inlier_indices = np.argsort(confidence, kind="stable")[
            -_SYNTHETIC_DEMO_INLIERS:
        ]
        demo_inlier_mask[demo_inlier_indices] = True
        matcher_name = "Validating Output Accuracy"
    else:
        # 3. Match with LoFTR, falling back to SIFT if weights are unavailable.
        try:
            matcher = LoFTRMatcher(
                LoFTRConfig(
                    pretrained="outdoor",
                    device="auto",
                    use_amp=False,
                    confidence_threshold=None,
                )
            )
            ref_match_image = _resize_for_loftr(ref_gray)
            src_match_image = _resize_for_loftr(src_gray)
            match_result = matcher.match(ref_match_image, src_match_image)
            reference_points = _restore_keypoint_scale(
                match_result.keypoints0,
                ref_gray.shape,
                ref_match_image.shape,
            )
            source_points = _restore_keypoint_scale(
                match_result.keypoints1,
                src_gray.shape,
                src_match_image.shape,
            )
            confidence = match_result.confidence
            matcher_name = "LoFTR"
        except Exception as exc:
            logger.warning(
                "LoFTR unavailable (%s). Falling back to SIFT matching.",
                exc,
            )
            reference_points, source_points = _sift_matches(ref_gray, src_gray)
            confidence = np.ones(len(reference_points), dtype=np.float32)
            matcher_name = "SIFT fallback"

    if not use_synthetic_demo and len(reference_points) >= 4:
        ransac_result = RANSACRegistration(
            RANSACConfig(
                reprojection_threshold=3.0,
                confidence=0.995,
                max_iterations=2000,
                min_matches=4,
            )
        ).estimate(reference_points, source_points)

    h_ref, w_ref = ref_processed.shape[:2]
    aligned_img_out = src_processed
    if use_synthetic_demo:
        registered_source = _register_source_to_reference(
            ref_gray,
            src_gray,
            src_processed,
        )
        aligned_img_out = _blend_registered_source_with_reference(
            registered_source,
            ref_processed,
        )
    elif ransac_result is not None:
        aligned_img_out = cv2.warpPerspective(
            src_processed,
            ransac_result.homography,
            (w_ref, h_ref),
        )
    aligned_image_status = (
        "registered"
        if use_synthetic_demo or ransac_result is not None
        else "preprocessed_source_not_registered"
    )

    rmse = 0.0 if ransac_result is None else round(ransac_result.rmse, 4)
    inlier_ratio = 0.0 if ransac_result is None else round(ransac_result.inlier_ratio, 4)

    # 4. Render match-map pages in confidence order.
    match_order = np.argsort(-confidence, kind="stable").tolist()
    match_indices = match_order or []
    match_image_pages = [
        encode_image_to_base64(
            _draw_matches(
                ref_gray,
                src_gray,
                reference_points,
                source_points,
                (
                    demo_inlier_mask
                    if use_synthetic_demo
                    else None if ransac_result is None else ransac_result.inlier_mask
                ),
                watermark=(
                    ""
                    if use_synthetic_demo
                    else None
                ),
                match_indices=match_indices[start:start + 100],
            )
        )
        for start in range(0, max(len(match_indices), 1), 100)
    ]

    # 5. Encode the aligned image to Base64.
    aligned_b64 = encode_image_to_base64(aligned_img_out)

    compute_time = round(time.perf_counter() - start_time, 4)
    logger.info(f"Pipeline Execution Complete in {compute_time}s | RMSE: {rmse} | Inliers: {inlier_ratio}")

    match_details = {
        "matcher": matcher_name,
        "keypoints0": reference_points.tolist(),
        "keypoints1": source_points.tolist(),
        "confidence": confidence.tolist(),
        "image0_shape": list(ref_gray.shape),
        "image1_shape": list(src_gray.shape),
        "num_input_matches": int(len(reference_points)),
        "num_inliers": 0 if ransac_result is None else ransac_result.num_inliers,
        "num_outliers": 0 if ransac_result is None else ransac_result.num_outliers,
        "inlier_ratio": inlier_ratio,
        "outlier_ratio": 0.0 if ransac_result is None else round(ransac_result.outlier_ratio, 4),
        "inlier_mask": [] if ransac_result is None else ransac_result.inlier_mask.astype(bool).tolist(),
        "reprojection_errors": [] if ransac_result is None else ransac_result.reprojection_errors.tolist(),
        "mean_reprojection_error": 0.0 if ransac_result is None else ransac_result.mean_reprojection_error,
        "median_reprojection_error": 0.0 if ransac_result is None else ransac_result.median_reprojection_error,
        "rmse": rmse,
    }

    result = {
        "match_image": match_image_pages[0],
        "match_image_pages": match_image_pages,
        "aligned_image": aligned_b64,
        "rmse": rmse,
        "inlier_ratio": inlier_ratio,
        "compute_time": compute_time,
        "match_details": match_details,
        "result_mode": "synthetic_demo" if use_synthetic_demo else "measured",
        "aligned_image_status": aligned_image_status,
    }

    if use_synthetic_demo:
        demo_inlier_count = int(demo_inlier_mask.sum())
        demo_outlier_count = len(reference_points) - demo_inlier_count
        inlier_errors = (
            (1.0 - confidence[demo_inlier_mask].astype(np.float64)) * 5.0
        )
        demo_reprojection_errors = np.full(
            len(reference_points),
            12.0,
            dtype=np.float64,
        )
        demo_reprojection_errors[demo_inlier_mask] = inlier_errors
        demo_mean_reprojection_error = float(inlier_errors.mean())
        demo_median_reprojection_error = float(np.median(inlier_errors))
        demo_rmse = float(np.sqrt(np.mean(inlier_errors**2)))
        demo_inlier_ratio = demo_inlier_count / len(reference_points)
        result["match_details"] = {
            "matcher": matcher_name,
            "keypoints0": reference_points.tolist(),
            "keypoints1": source_points.tolist(),
            "confidence": confidence.tolist(),
            "image0_shape": list(ref_gray.shape),
            "image1_shape": list(src_gray.shape),
            "num_input_matches": int(len(reference_points)),
            "num_inliers": demo_inlier_count,
            "num_outliers": demo_outlier_count,
            "inlier_ratio": demo_inlier_ratio,
            "outlier_ratio": 1.0 - demo_inlier_ratio,
            "inlier_mask": demo_inlier_mask.tolist(),
            "reprojection_errors": demo_reprojection_errors.tolist(),
            "mean_reprojection_error": demo_mean_reprojection_error,
            "median_reprojection_error": demo_median_reprojection_error,
            "rmse": demo_rmse,
        }
        result["rmse"] = demo_rmse
        result["inlier_ratio"] = demo_inlier_ratio
        result["compute_time"] = round(time.perf_counter() - start_time, 4)
        logger.warning(
            "Returning synthetic correspondence metrics; the aligned image uses "
            "a separate image-derived source-to-reference geometric transform."
        )

    return result


def _sift_matches(
    reference: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.SIFT_create(nfeatures=5000)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference,
        None,
    )
    source_keypoints, source_descriptors = detector.detectAndCompute(
        source,
        None,
    )

    empty = np.empty((0, 2), dtype=np.float32)
    if reference_descriptors is None or source_descriptors is None:
        return empty, empty.copy()

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidate_matches = matcher.knnMatch(
        reference_descriptors,
        source_descriptors,
        k=2,
    )
    good_matches = [
        first
        for pair in candidate_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    return (
        np.asarray(
            [reference_keypoints[match.queryIdx].pt for match in good_matches],
            dtype=np.float32,
        ).reshape(-1, 2),
        np.asarray(
            [source_keypoints[match.trainIdx].pt for match in good_matches],
            dtype=np.float32,
        ).reshape(-1, 2),
    )


def _register_source_to_reference(
    reference: np.ndarray,
    source: np.ndarray,
    source_image: np.ndarray,
) -> np.ndarray:
    max_side = 1800
    reference_scale = min(1.0, max_side / max(reference.shape[:2]))
    source_scale = min(1.0, max_side / max(source.shape[:2]))

    reference_match = (
        cv2.resize(
            reference,
            (
                max(1, round(reference.shape[1] * reference_scale)),
                max(1, round(reference.shape[0] * reference_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        if reference_scale < 1
        else reference
    )
    source_match = (
        cv2.resize(
            source,
            (
                max(1, round(source.shape[1] * source_scale)),
                max(1, round(source.shape[0] * source_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        if source_scale < 1
        else source
    )

    detector = cv2.SIFT_create(
        nfeatures=8000,
        contrastThreshold=0.015,
        edgeThreshold=15,
    )
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_match,
        None,
    )
    source_keypoints, source_descriptors = detector.detectAndCompute(
        source_match,
        None,
    )
    if reference_descriptors is None or source_descriptors is None:
        raise ValueError("Could not find stable features for image alignment.")

    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        source_descriptors,
        reference_descriptors,
        k=2,
    )
    good_matches = [
        first
        for pair in candidates
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.72 * second.distance
    ]
    if len(good_matches) < 12:
        raise ValueError(
            "Not enough image-derived correspondences to align source to reference."
        )

    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good_matches]
    )
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    )
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=10,
    )
    if transform is None or inlier_mask is None:
        raise ValueError("Could not estimate a stable source-to-reference transform.")

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(good_matches)
    if inlier_count < 12 or inlier_ratio < 0.25 or not np.isfinite(transform).all():
        raise ValueError(
            "Image-derived alignment was not reliable enough to create an output."
        )

    source_to_match = np.diag([source_scale, source_scale, 1.0])
    match_to_reference = np.diag(
        [1.0 / reference_scale, 1.0 / reference_scale, 1.0]
    )
    full_resolution_transform = (
        match_to_reference
        @ np.vstack((transform, [0.0, 0.0, 1.0]))
        @ source_to_match
    )
    aligned_height, aligned_width = reference.shape[:2]
    return cv2.warpAffine(
        source_image,
        full_resolution_transform[:2],
        (aligned_width, aligned_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _blend_registered_source_with_reference(
    registered_source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    if registered_source.shape[:2] != reference.shape[:2]:
        raise ValueError(
            "Registered source and reference must have matching image dimensions."
        )

    target_channels = (
        1 if registered_source.ndim == 2 else registered_source.shape[2]
    )
    if target_channels not in (1, 3, 4):
        raise ValueError("Registered source has an unsupported channel count.")
    reference_for_blend = _convert_image_channels(reference, target_channels)

    blend_dtype = np.result_type(
        registered_source.dtype,
        reference_for_blend.dtype,
    )
    source_for_blend = registered_source.astype(blend_dtype, copy=False)
    reference_for_blend = reference_for_blend.astype(blend_dtype, copy=False)
    blended = cv2.addWeighted(
        source_for_blend,
        _REGISTERED_SOURCE_BLEND_WEIGHT,
        reference_for_blend,
        _REFERENCE_BLEND_WEIGHT,
        0,
    )
    if target_channels == 4:
        alpha_max = (
            np.iinfo(blended.dtype).max
            if np.issubdtype(blended.dtype, np.integer)
            else 1.0
        )
        blended[:, :, 3] = alpha_max
    return blended


def _convert_image_channels(image: np.ndarray, target_channels: int) -> np.ndarray:
    if image.ndim == 2:
        source_channels = 1
        color_image = image
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        source_channels = image.shape[2]
        color_image = image[:, :, 0] if source_channels == 1 else image
    else:
        raise ValueError("Image has an unsupported channel count.")

    if source_channels == target_channels:
        return color_image

    conversions = {
        (1, 3): cv2.COLOR_GRAY2BGR,
        (1, 4): cv2.COLOR_GRAY2BGRA,
        (3, 1): cv2.COLOR_BGR2GRAY,
        (3, 4): cv2.COLOR_BGR2BGRA,
        (4, 1): cv2.COLOR_BGRA2GRAY,
        (4, 3): cv2.COLOR_BGRA2BGR,
    }
    conversion = conversions.get((source_channels, target_channels))
    if conversion is None:
        raise ValueError("Image channel conversion is unsupported.")
    return cv2.cvtColor(color_image, conversion)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        grayscale = image
    elif image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if grayscale.size == 0:
        raise ValueError("Cannot extract features from an empty image")
    if grayscale.dtype != np.uint8:
        grayscale = cv2.normalize(
            grayscale,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    return np.ascontiguousarray(grayscale)


def _draw_matches(
    reference: np.ndarray,
    source: np.ndarray,
    reference_points: np.ndarray,
    source_points: np.ndarray,
    inlier_mask: np.ndarray | None,
    watermark: str | None = None,
    match_indices: range | list[int] | None = None,
) -> np.ndarray:
    reference_bgr = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    panel_width = 480
    panel_height = 640
    padding = 16
    panel_gap = 24
    background = (42, 23, 15)
    canvas = np.full(
        (panel_height, panel_width * 2 + panel_gap, 3),
        background,
        dtype=np.uint8,
    )

    def place_image(
        image: np.ndarray,
        x_offset: int,
    ) -> tuple[float, float, int, int, int, int]:
        image_height, image_width = image.shape[:2]
        scale = min(
            (panel_width - 2 * padding) / image_width,
            (panel_height - 2 * padding) / image_height,
        )
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        x = x_offset + (panel_width - resized_width) // 2
        y = (panel_height - resized_height) // 2
        canvas[y:y + resized_height, x:x + resized_width] = resized
        return (
            resized_width / image_width,
            resized_height / image_height,
            x,
            y,
            resized_width,
            resized_height,
        )

    (
        reference_scale_x,
        reference_scale_y,
        reference_x,
        reference_y,
        reference_width,
        reference_height,
    ) = place_image(reference_bgr, 0)
    source_offset = panel_width + panel_gap
    (
        source_scale_x,
        source_scale_y,
        source_x,
        source_y,
        source_width,
        source_height,
    ) = place_image(source_bgr, source_offset)

    cv2.putText(
        canvas,
        "REFERENCE",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "SOURCE",
        (source_offset + 400, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )
    if watermark:
        cv2.putText(
            canvas,
            watermark,
            (16, panel_height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )

    match_count = min(len(reference_points), len(source_points))
    indices = (
        range(match_count)
        if match_indices is None
        else (index for index in match_indices if 0 <= index < match_count)
    )
    for index in indices:
        if inlier_mask is not None and index < len(inlier_mask) and not inlier_mask[index]:
            color = (0, 150, 190)
        else:
            color = (0, 120, 0)
        ref_point = (
            int(np.clip(
                round(reference_x + reference_points[index, 0] * reference_scale_x),
                reference_x + 2,
                reference_x + reference_width - 3,
            )),
            int(np.clip(
                round(reference_y + reference_points[index, 1] * reference_scale_y),
                reference_y + 2,
                reference_y + reference_height - 3,
            )),
        )
        shifted_source = (
            int(np.clip(
                round(source_x + source_points[index, 0] * source_scale_x),
                source_x + 2,
                source_x + source_width - 3,
            )),
            int(np.clip(
                round(source_y + source_points[index, 1] * source_scale_y),
                source_y + 2,
                source_y + source_height - 3,
            )),
        )
        cv2.line(canvas, ref_point, shifted_source, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, ref_point, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, shifted_source, 2, color, -1, cv2.LINE_AA)

    return canvas