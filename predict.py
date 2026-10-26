"""Offline segmentation and contamination quantification, Eqs. (19)--(22)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

WARP_WIDTH = 1200
WARP_HEIGHT = 1800
GRID_ROWS = 8
GRID_COLS = 10
B_SMOOTH_KERNEL = 15
MIN_COMPONENT_AREA = 500
EPSILON = 1e-6


def order_quad_points(points):

    """Unique cyclic vertices; preserve consistent portrait orientation."""

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    if not np.isfinite(pts).all() or len(np.unique(pts, axis=0)) != 4:

        raise ValueError("INVALID_QUAD: four distinct finite vertices required")

    center = pts.mean(axis=0)

    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]

    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)

    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)) or abs(cv2.contourArea(pts)) < 4:

        raise ValueError("INVALID_QUAD: degenerate/non-convex polygon")

    # Same convention for current/reference panels; no duplicate extrema at 45 degrees.

    widths = np.linalg.norm(pts[1]-pts[0]) + np.linalg.norm(pts[2]-pts[3])

    heights = np.linalg.norm(pts[3]-pts[0]) + np.linalg.norm(pts[2]-pts[1])

    if widths > heights:

        pts = np.roll(pts, -1, axis=0)

    return pts.astype(np.float32)

def quad_from_extended_panel_edges(component_mask: np.ndarray) -> Optional[np.ndarray]:

    """Fit four supported straight panel edges and intersect their extensions.

    Only geometry is reconstructed. The original class mask remains unchanged,

    so missing/occluded pixels are not filled into the DSI valid region.

    Return None if four sufficiently long straight edges cannot be supported.

    """

    from itertools import combinations

    binary = component_mask.astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:

        return None

    contour = max(contours, key=cv2.contourArea)

    points = contour.reshape(-1, 2).astype(np.float64)

    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, True)

    polygon = cv2.approxPolyDP(hull, 0.003 * perimeter, True).reshape(-1, 2)

    h, w = binary.shape

    _, _, bw, bh = cv2.boundingRect(contour)

    diagonal = float(np.hypot(bw, bh))

    tolerance = max(1.5, 0.003 * diagonal)

    minimum_span = max(15.0, 0.10 * diagonal)

    candidates = []

    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):

        direction = (end - start).astype(np.float64)

        length = float(np.linalg.norm(direction))

        if length < minimum_span:

            continue

        direction /= length

        normal = np.array([-direction[1], direction[0]])

        offset = points - start

        projection = offset @ direction

        keep = ((np.abs(offset @ normal) <= tolerance)

                & (projection >= -tolerance) & (projection <= length + tolerance))

        support = points[keep]

        if len(support) < 12:

            continue

        for _ in range(2):

            vx, vy, px, py = cv2.fitLine(

                support.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01

            ).reshape(-1)

            direction = np.array([vx, vy], dtype=np.float64)

            normal = np.array([-vy, vx], dtype=np.float64)

            origin = np.array([px, py], dtype=np.float64)

            distance = np.abs((support - origin) @ normal)

            support = support[distance <= tolerance]

            if len(support) < 12:

                break

        if len(support) < 12:

            continue

        projection = (support - origin) @ direction

        lo, hi = np.percentile(projection, [1, 99])

        span = float(hi - lo)

        residual = np.abs((support - origin) @ normal)

        # Reject curved/short fragments and convex-hull bridges with no real edge.

        if (span < minimum_span or len(support) / max(span, 1.0) < 0.55

                or np.sqrt(np.mean(residual ** 2)) > 0.55 * tolerance):

            continue

        candidate = dict(normal=normal, c=float(normal @ origin), origin=origin,

                         direction=direction, lo=float(lo), hi=float(hi), span=span)

        duplicate = None

        for idx, other in enumerate(candidates):

            alignment = float(normal @ other['normal'])

            if (abs(alignment) > np.cos(np.deg2rad(5))

                    and abs(float(normal @ other['origin']) - candidate['c']) < 2 * tolerance):

                duplicate = idx

                break

        if duplicate is None:

            candidates.append(candidate)

        elif span > candidates[duplicate]['span']:

            candidates[duplicate] = candidate

    if len(candidates) < 4:

        return None

    candidates = sorted(candidates, key=lambda item: item['span'], reverse=True)[:10]

    # Check coverage against the real contour, not a filled convex hull.

    contour_sample = points[::max(1, len(points) // 600)]

    visible_area = float(cv2.contourArea(contour))

    best_quad, best_score = None, -np.inf

    for four in combinations(candidates, 4):

        intersections = []

        for left, right in combinations(four, 2):

            matrix = np.stack([left['normal'], right['normal']])

            if abs(np.linalg.det(matrix)) < np.sin(np.deg2rad(12)):

                continue

            vertex = np.linalg.solve(matrix, [left['c'], right['c']])

            if (-tolerance <= vertex[0] <= w - 1 + tolerance

                    and -tolerance <= vertex[1] <= h - 1 + tolerance):

                intersections.append(vertex)

        # A quadrilateral may have additional opposite-edge intersections in frame.

        for vertices in combinations(intersections, 4):

            try:

                quad = order_quad_points(np.array(vertices, dtype=np.float32))

            except ValueError:

                continue

            area = abs(float(cv2.contourArea(quad)))

            if not (0.98 * visible_area <= area <= 1.65 * visible_area):

                continue

            ordered_lines = []

            for a, b in zip(quad, np.roll(quad, -1, axis=0)):

                distances = [max(abs(float(line['normal'] @ a) - line['c']),

                                 abs(float(line['normal'] @ b) - line['c'])) for line in four]

                idx = int(np.argmin(distances))

                if distances[idx] > 2 * tolerance or idx in ordered_lines:

                    break

                line = four[idx]

                side_length = float(np.linalg.norm(b - a))

                if line['span'] < 0.30 * side_length:

                    break

                endpoints = (np.stack([a, b]) - line['origin']) @ line['direction']

                extension = max(line['lo'] - min(endpoints), max(endpoints) - line['hi'], 0.0)

                if extension > 0.50 * side_length + tolerance:

                    break

                ordered_lines.append(idx)

            if len(ordered_lines) != 4:

                continue

            distances = np.array([cv2.pointPolygonTest(quad.reshape(-1, 1, 2),

                                  (float(p[0]), float(p[1])), True) for p in contour_sample])

            if np.mean(distances >= -2 * tolerance) < 0.995:

                continue

            score = sum(line['span'] for line in four) - 0.02 * area / max(diagonal, 1)

            if score > best_score:

                best_quad, best_score = quad, score

    return best_quad

def contour_to_quad(component_mask: np.ndarray) -> Optional[np.ndarray]:

    mask_uint8 = (component_mask.astype(np.uint8) * 255)

    kernel = np.ones((5, 5), np.uint8)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:

        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_COMPONENT_AREA:

        return None

    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, True)

    # Missing corners add a diagonal/notch to the visible outline. Recover the

    # four real straight edges before the original coarse polygon approximation.

    detailed_hull = cv2.approxPolyDP(hull, 0.006 * perimeter, True)

    if len(detailed_hull) > 4:

        extended_quad = quad_from_extended_panel_edges(component_mask)

        if extended_quad is not None:

            print("[corners] EDGE_EXTENSION: fitted visible edges and intersected extensions")

            return extended_quad

        # Do not silently treat an unsupported missing-corner outline as a rectangle.

        print("[failure] EDGE_EXTENSION_UNSUPPORTED: four reliable straight edges not found")

        return None

    for ratio in (0.015, 0.02, 0.03, 0.04, 0.05):

        approx = cv2.approxPolyDP(hull, ratio * perimeter, True)

        if len(approx) == 4:

            return order_quad_points(approx.reshape(4, 2))

    return quad_from_extended_panel_edges(component_mask)


def validate_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != tuple(shape):
        raise ValueError('A class-ID mask must match the original image height and width.')
    if not np.isin(mask, [0, 1, 2]).all():
        raise ValueError('Expected mask labels 0=background, 1=panel, 2=bird droppings.')
    return mask.astype(np.uint8)


def select_panel(mask, mode='largest', index=0):
    """Select a single connected panel including its bird-dropping pixels."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        ((mask == 1) | (mask == 2)).astype(np.uint8), connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
    if not candidates:
        return None
    if mode == 'largest':
        selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    elif mode == 'auto_center':
        center = np.array([mask.shape[1]/2, mask.shape[0]/2])
        selected = min(candidates, key=lambda i: np.linalg.norm(centers[i]-center))
    elif mode == 'index':
        candidates.sort(key=lambda i: (centers[i, 0], centers[i, 1]))
        if not 0 <= index < len(candidates):
            raise ValueError('Panel index out of range.')
        selected = candidates[index]
    else:
        raise ValueError('Unknown panel selection mode.')
    return np.where(labels == selected, mask, 0).astype(np.uint8)


def compute_bird_statistics(mask):
    """Eq. (19), in percent. Supply the selected panel mask."""
    solar = int((mask == 1).sum())
    bird = int((mask == 2).sum())
    panel = solar + bird
    return {'solar_pixels': solar, 'bird_pixels': bird, 'panel_pixels': panel,
            'bird_ratio_on_panel': None if panel == 0 else 100.0 * bird/panel}


def warp_panel_to_rect(image, mask, mode='largest', index=0, quad=None):
    mask = validate_mask(mask, image.shape[:2])
    selected = select_panel(mask, mode, index)
    if selected is None:
        return None
    if quad is None:
        quad = contour_to_quad((selected > 0).astype(np.uint8))
    else:
        quad = order_quad_points(quad)
    if quad is None:
        return None
    destination = np.array([[0, 0], [WARP_WIDTH-1, 0],
                            [WARP_WIDTH-1, WARP_HEIGHT-1], [0, WARP_HEIGHT-1]], np.float32)
    homography = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination)
    if not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) != 3:
        return None
    warped_image = cv2.warpPerspective(image, homography, (WARP_WIDTH, WARP_HEIGHT), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(selected, homography, (WARP_WIDTH, WARP_HEIGHT), flags=cv2.INTER_NEAREST)
    return {'warped_image': warped_image, 'warped_mask': warped_mask,
            'selected_mask': selected, 'panel_quad': quad, 'homography': homography}


def rgb_to_lab_b_channel(image):
    """OpenCV float RGB [0,1] -> signed CIELAB b*, no uint8 offset."""
    rgb = np.asarray(image)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)/255.0
    else:
        rgb = rgb.astype(np.float32)
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('Floating-point RGB input must be finite and in [0,1].')
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 2]


def smooth_b_channel(b_channel, valid):
    """15x15 Gaussian smoothing using valid panel pixels only.

    Normalized convolution prevents bird/occluder colors entering neighboring pixels.
    """
    weights = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(b_channel*weights, (B_SMOOTH_KERNEL, B_SMOOTH_KERNEL), 0)
    denominator = cv2.GaussianBlur(weights, (B_SMOOTH_KERNEL, B_SMOOTH_KERNEL), 0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPSILON)


def compute_grid_dust_metrics(current, current_mask, reference, reference_mask):
    """Eqs. (20)--(22): positive clean-reference residual and valid-pixel mean."""
    if current.shape != reference.shape or current_mask.shape != reference_mask.shape:
        raise ValueError('Current and reference must have the same rectified size.')
    current_valid = current_mask == 1
    reference_valid = reference_mask == 1
    valid = current_valid & reference_valid
    if not valid.any():
        return {'dsi_b': None, 'failure_reason': 'NO_COMMON_VALID_PIXELS',
                'grid_score_map': None, 'grid_weights': None, 'valid_mask': valid, 'residual_map': None}
    current_b = smooth_b_channel(rgb_to_lab_b_channel(current), current_valid)
    reference_b = smooth_b_channel(rgb_to_lab_b_channel(reference), reference_valid)
    residual = np.maximum(current_b-reference_b, 0)
    residual[~valid] = 0
    scores = np.full((GRID_ROWS, GRID_COLS), np.nan, np.float64)
    weights = np.zeros((GRID_ROWS, GRID_COLS), np.int64)
    yy = np.linspace(0, current.shape[0], GRID_ROWS+1, dtype=int)
    xx = np.linspace(0, current.shape[1], GRID_COLS+1, dtype=int)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            region = np.s_[yy[row]:yy[row+1], xx[col]:xx[col+1]]
            local = valid[region]
            count = int(local.sum())
            if count:
                weights[row, col] = count
                scores[row, col] = float(residual[region][local].mean(dtype=np.float64))
    nonempty = weights > 0
    dsi = float(np.average(scores[nonempty], weights=weights[nonempty]))
    return {'dsi_b': dsi, 'failure_reason': None, 'grid_score_map': scores,
            'grid_weights': weights, 'valid_mask': valid, 'residual_map': residual}


def quantify(image, mask, reference, reference_mask, mode='largest', index=0):
    mask = validate_mask(mask, image.shape[:2])
    reference_mask = validate_mask(reference_mask, reference.shape[:2])
    selected = select_panel(mask, mode, index)
    stats = compute_bird_statistics(np.zeros_like(mask) if selected is None else selected)
    current = warp_panel_to_rect(image, mask, mode, index)
    clean = warp_panel_to_rect(reference, reference_mask)
    metrics = {'R_percent': stats['bird_ratio_on_panel'], 'DSI_b_star': None,
               'coverage_valid': stats['bird_ratio_on_panel'] is not None,
               'dsi_valid': False, 'rectified_width': WARP_WIDTH, 'rectified_height': WARP_HEIGHT,
               'grid_rows': GRID_ROWS, 'grid_cols': GRID_COLS, 'b_smoothing_kernel': B_SMOOTH_KERNEL,
               'class_pixels': stats}
    if current is None or clean is None:
        metrics['failure_reason'] = 'CURRENT_PANEL_GEOMETRY_FAILED' if current is None else 'REFERENCE_PANEL_GEOMETRY_FAILED'
        return metrics, None
    dust = compute_grid_dust_metrics(current['warped_image'], current['warped_mask'],
                                     clean['warped_image'], clean['warped_mask'])
    metrics.update(DSI_b_star=dust['dsi_b'], dsi_valid=dust['dsi_b'] is not None,
                   failure_reason=dust['failure_reason'],
                   current_quad=current['panel_quad'].tolist(), reference_quad=clean['panel_quad'].tolist(),
                   valid_dust_pixels=int(dust['valid_mask'].sum()))
    return metrics, {'current': current, 'reference': clean, 'dust': dust}


def load_model(model_dir, device):
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
    from model_structure import CustomSegformerDecodeHead, DECODER_ARCHITECTURE
    model_dir = Path(model_dir)
    if (model_dir/'model.safetensors').exists():
        from safetensors.torch import load_file
        state = load_file(str(model_dir/'model.safetensors'), device='cpu')
    elif (model_dir/'pytorch_model.bin').exists():
        state = torch.load(str(model_dir/'pytorch_model.bin'), map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError('Provide model.safetensors or pytorch_model.bin in --model-dir.')
    if any(k.startswith(('decode_head.deep_decoder.', 'decode_head.fuse_head.')) for k in state):
        raise ValueError(
            'This checkpoint uses the previous multistage convolutional decoder. '
            'The Fig. 1 MLP-concat-SCPP decoder requires newly trained weights.')
    config = SegformerConfig.from_pretrained(str(model_dir), local_files_only=True)
    model = SegformerForSemanticSegmentation(config)
    if any(k.startswith('decode_head.proj_layers.') for k in state):
        architecture = getattr(config, 'custom_decoder_architecture', None)
        if architecture != DECODER_ARCHITECTURE:
            raise ValueError(
                'The custom checkpoint does not identify the Fig. 1 decoder architecture. '
                'Use weights saved by the updated training program.')
        model.decode_head = CustomSegformerDecodeHead(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(str(model_dir), local_files_only=True, do_reduce_labels=False)
    return model, processor


def run_segmentation(model, processor, image, device):
    import torch
    import torch.nn.functional as F
    encoded = processor(images=image, return_tensors='pt')
    with torch.no_grad():
        logits = model(pixel_values=encoded['pixel_values'].to(device)).logits
    return F.interpolate(logits, size=(image.height, image.width), mode='bilinear',
                         align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)


def read_mask(path, shape):
    return validate_mask(np.asarray(Image.open(path)), shape)


def main():
    parser = argparse.ArgumentParser(description='Single-panel coverage and clean-reference DSI')
    parser.add_argument('image', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--reference-mask', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--select', choices=['largest', 'auto_center', 'index'], default='largest')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.model_dir is None and (args.mask is None or args.reference_mask is None):
        parser.error('Provide --model-dir, or both --mask and --reference-mask.')
    current_pil = Image.open(args.image).convert('RGB')
    reference_pil = Image.open(args.reference).convert('RGB')
    image, reference = np.asarray(current_pil), np.asarray(reference_pil)
    model = processor = None
    if args.mask is None or args.reference_mask is None:
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model, processor = load_model(args.model_dir, device)
    mask = read_mask(args.mask, image.shape[:2]) if args.mask else run_segmentation(model, processor, current_pil, device)
    reference_mask = read_mask(args.reference_mask, reference.shape[:2]) if args.reference_mask else run_segmentation(model, processor, reference_pil, device)
    metrics, detail = quantify(image, mask, reference, reference_mask, args.select, args.index)
    metrics.update(image=str(args.image), reference=str(args.reference), model_dir=str(args.model_dir) if args.model_dir else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    (args.output_dir/f'{stem}_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    Image.fromarray(mask).save(args.output_dir/f'{stem}_mask.png')
    if detail:
        Image.fromarray(detail['current']['warped_image']).save(args.output_dir/f'{stem}_rectified.png')
        Image.fromarray(detail['current']['warped_mask']).save(args.output_dir/f'{stem}_rectified_mask.png')
        Image.fromarray(detail['dust']['valid_mask'].astype(np.uint8)*255).save(args.output_dir/f'{stem}_valid.png')
        if detail['dust']['residual_map'] is not None:
            np.savez_compressed(args.output_dir/f'{stem}_dust.npz',
                                grid=detail['dust']['grid_score_map'], weights=detail['dust']['grid_weights'],
                                residual=detail['dust']['residual_map'])
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
