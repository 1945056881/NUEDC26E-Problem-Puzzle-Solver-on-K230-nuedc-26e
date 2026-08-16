"""2026 TI Cup E-problem: K230 upper-controller program.

Pipeline:
    camera -> A4 corner calibration -> high-resolution birdview
    -> upper-half piece detection -> polygon/texture puzzle solver
    -> pick/place commands for the lower motion controller

Copy this file, puzzle_solver.py and birdview_k230.py to the K230.
"""

import gc
import math
import os
import time

import cv2
import image
import ulab.numpy as np

from machine import FPIOA, UART
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

from birdview_k230 import make_homography
from puzzle_solver import (
    average_polygon_frames,
    merge_collinear_vertices,
    normalize_polygon,
    placement_packets,
    polygon_area,
    polygon_centroid,
    solve_puzzle,
)


# ================================ User settings ================================
# --------------------------- Physical size inputs -----------------------------
# Enter the paper and the reconstructed metal-plate dimensions used right now.
# The current temporary setup is B5 paper and an approximately 10 x 10 cm plate.
PAPER_WIDTH_CM = 19.0
PAPER_HEIGHT_CM = 26.5
EXPECTED_TARGET_WIDTH_CM = 10.0
EXPECTED_TARGET_HEIGHT_CM = 10.0
TARGET_SIZE_TOLERANCE_CM = 1.2

# True: constrain the solver around EXPECTED_TARGET_* above.
# False: accept any rectangle that fits the lower half of the paper.
USE_EXPECTED_TARGET_SIZE = True

# Enable only after changing back to the official E-problem paper/pieces.
STRICT_CONTEST_TARGET_SIZE = False
USE_EDGE_TEXTURE_SCORE = False
ENABLE_PARTIAL_EDGE_FALLBACK = True

# --------------------------- Resolution preset -------------------------------
# True uses the OV5647 at 1280 x 960 and a 30 px/cm birdview.  This gives
# about 2.25 times as many birdview pixels as the original 20 px/cm mode.
# Set False if the firmware reports insufficient memory.
HIGH_RESOLUTION_MODE = True

# -------------------------- Piece detection tuning ---------------------------
# Main parameters to tune on site.  Metal is white in the binary mask when its
# grayscale value is inside this inclusive range.
DETECTION_MODE = "gray_range"
PIECE_GRAY_MIN = 15
PIECE_GRAY_MAX = 120

# Increase OPEN_SIZE if neighbouring pieces stick together; reduce it if sharp
# corners are being eroded.  CLOSE_SIZE only repairs tiny edge gaps.
MORPH_OPEN_SIZE = 5
MORPH_CLOSE_SIZE = 3

# Collect 30 consecutive stable frames, average them, then solve and draw.
DETECTION_STABLE_FRAMES = 30
DETECTION_MAX_CENTRE_DRIFT_CM = 0.30
# Average the latest stable contours before puzzle solving.
DETECTION_AVERAGE_FRAMES = 30
# A vertex is removed when the included angle is within this value of 180°.
COLLINEAR_MERGE_DEG = 15.0

SOLVER_MAX_GAP_RATIO = 0.25
# Allow at most one piece whose detected outside edge misses the fitted
# rectangle.  Set to 0 for ideal contours; keep 1 for real camera images.
SOLVER_MAX_MISSING_OUTER_PIECES = 1
# Averaged outlines may overlap slightly at a seam.  This is an area limit,
# not a distance threshold.  The current capture needs about 0.44 cm^2.
SOLVER_MAX_CAMERA_PAIR_OVERLAP_CM2 = 0.65
SOLVER_MAX_CAMERA_TOTAL_OVERLAP_RATIO = 0.015
# Never enter the expensive generic partial-edge search on the K230.
SOLVER_PARTIAL_FAST_PATH_ONLY = True

# Camera and birdview processing resolution.  The LCD output remains a fixed
# 800 x 480 canvas, so changing these values does not alter VO layer attributes.
CAMERA_W = 1280 if HIGH_RESOLUTION_MODE else 800
CAMERA_H = 960 if HIGH_RESOLUTION_MODE else 600
CAMERA_FPS = 20
CAMERA_HMIRROR = False
CAMERA_VFLIP = False

# 30 px/cm gives 3 pixels/mm in the rectified paper image.
PAPER_PIXELS_PER_CM = 30 if HIGH_RESOLUTION_MODE else 20

# Automatically lock the largest A4 quadrilateral.  The manual points are used
# after AUTO_PAPER_TIMEOUT_FRAMES if automatic detection does not settle.
AUTO_PAPER_DETECT = True
AUTO_PAPER_STABLE_FRAMES = 3
AUTO_PAPER_MAX_DRIFT_PX = 14.0
AUTO_PAPER_TIMEOUT_FRAMES = 150
# Scale the original 800 x 600 fallback calibration with camera resolution.
MANUAL_PAPER_POINTS = (
    (int(185 * CAMERA_W / 800), int(8 * CAMERA_H / 600)),
    (int(615 * CAMERA_W / 800), int(8 * CAMERA_H / 600)),
    (int(615 * CAMERA_W / 800), int(592 * CAMERA_H / 600)),
    (int(185 * CAMERA_W / 800), int(592 * CAMERA_H / 600)),
)
PAPER_MIN_CAMERA_AREA_RATIO = 0.20

# The E-problem pieces start in the upper half.  Keep away from the black split.
DIVIDER_Y_CM = PAPER_HEIGHT_CM * 0.5
DIVIDER_EXCLUSION_CM = 0.55
MASK_BORDER_CM = 0.35
# Optional literal-white mode threshold.
WHITE_MIN_CHANNEL = 210
# Maximum per-channel deviation still considered A4 paper.
BACKGROUND_CHANNEL_TOLERANCE = 32
EDGE_ASSIST_ENABLE = False
EDGE_CANNY_LOW = 28
EDGE_CANNY_HIGH = 95
MIN_PIECE_AREA_CM2 = 1.3
MAX_PIECE_AREA_CM2 = 55.0
# 题目规定真实边长不小于 2 cm；较小值只用于删除反光产生的假短边。
MIN_DETECTED_EDGE_CM = 1.20

# Place the reconstructed rectangle in the centre of the lower half.
TARGET_CENTER_CM = (
    PAPER_WIDTH_CM * 0.5,
    PAPER_HEIGHT_CM * 0.75,
)
TARGET_DIRECTION_ARROW_CM = 1.15
TARGET_COLORS = (
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
)

# Lower-controller link is deliberately off for first bench testing.
LOWER_UART_ENABLE = False
LOWER_UART_TX_PIN = 5
LOWER_UART_RX_PIN = 6
LOWER_UART_BAUDRATE = 115200

DISPLAY_TYPE = Display.ST7701
DISPLAY_W = 800
DISPLAY_H = 480
DISPLAY_TO_IDE = True
LOG_INTERVAL_FRAMES = 15
# ==============================================================================


BIRD_W = int(PAPER_WIDTH_CM * PAPER_PIXELS_PER_CM + 0.5)
BIRD_H = int(PAPER_HEIGHT_CM * PAPER_PIXELS_PER_CM + 0.5)
UPPER_LIMIT_PX = int(
    (DIVIDER_Y_CM - DIVIDER_EXCLUSION_CM) * PAPER_PIXELS_PER_CM)
# The K230 VO layer cannot change width/height/position after its first frame.
# Both calibration and birdview are therefore letterboxed into this fixed canvas.
DISPLAY_CANVAS = np.zeros(
    (DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)


def _point_xy(raw):
    try:
        return int(raw[0][0]), int(raw[0][1])
    except (TypeError, IndexError):
        return int(raw[0]), int(raw[1])


def order_quad(points):
    """Return four points as top-left, top-right, bottom-right, bottom-left."""
    points = [(int(point[0]), int(point[1])) for point in points]
    by_y = sorted(points, key=lambda point: point[1])
    top = sorted(by_y[:2], key=lambda point: point[0])
    bottom = sorted(by_y[2:], key=lambda point: point[0])
    return (top[0], top[1], bottom[1], bottom[0])


def detect_paper_quad(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = (
        CAMERA_W * CAMERA_H * PAPER_MIN_CAMERA_AREA_RATIO)
    best_area = 0.0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area or area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4:
            continue
        best_area = area
        best = order_quad([_point_xy(point) for point in approx])
    return best


def quad_max_drift(first, second):
    if first is None or second is None:
        return 1000000.0
    maximum = 0.0
    for a, b in zip(first, second):
        distance = math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        maximum = max(maximum, distance)
    return maximum


def make_paper_homography(source_points):
    destination = (
        (0.0, 0.0),
        (BIRD_W - 1.0, 0.0),
        (BIRD_W - 1.0, BIRD_H - 1.0),
        (0.0, BIRD_H - 1.0),
    )
    return make_homography(source_points, destination)


def gray_range_mask(frame):
    """Return 255 for the measured metal grayscale range, otherwise 0."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([PIECE_GRAY_MIN], dtype=np.uint8)
    upper = np.array([PIECE_GRAY_MAX], dtype=np.uint8)
    return cv2.inRange(gray, lower, upper)


def clear_mask_border(mask):
    """Remove bright warp/A4 boundary lines before contour extraction."""
    border = max(2, int(MASK_BORDER_CM * PAPER_PIXELS_PER_CM))
    cv2.rectangle(
        mask, (0, 0),
        (mask.shape[1] - 1, mask.shape[0] - 1),
        (0,), border)
    return mask


def _piece_score(camera_frame, source_points):
    """Count metal pixels that land in the candidate birdview upper half."""
    matrix = make_paper_homography(source_points)
    preview = cv2.warpPerspective(
        camera_frame, matrix, (BIRD_W, BIRD_H))
    upper_roi = preview[0:UPPER_LIMIT_PX, :, :]
    piece_mask = clear_mask_border(gray_range_mask(upper_roi))
    score = cv2.countNonZero(piece_mask)
    del piece_mask
    del preview
    return score


def orient_paper_quad(camera_frame, ordered_quad):
    """Rotate the A4 ROI so the half containing pieces becomes the upper half."""
    top_left, top_right, bottom_right, bottom_left = ordered_quad
    horizontal = (
        math.sqrt((top_right[0] - top_left[0]) ** 2 +
                  (top_right[1] - top_left[1]) ** 2) +
        math.sqrt((bottom_right[0] - bottom_left[0]) ** 2 +
                  (bottom_right[1] - bottom_left[1]) ** 2)
    ) * 0.5
    vertical = (
        math.sqrt((bottom_left[0] - top_left[0]) ** 2 +
                  (bottom_left[1] - top_left[1]) ** 2) +
        math.sqrt((bottom_right[0] - top_right[0]) ** 2 +
                  (bottom_right[1] - top_right[1]) ** 2)
    ) * 0.5

    if horizontal > vertical:
        # Candidate A: camera-left -> A4-upper.  Candidate B: right -> upper.
        candidates = (
            (bottom_left, top_left, top_right, bottom_right),
            (top_right, bottom_right, bottom_left, top_left),
        )
    else:
        # Candidate A: camera-top -> A4-upper.  Candidate B: bottom -> upper.
        candidates = (
            (top_left, top_right, bottom_right, bottom_left),
            (bottom_right, bottom_left, top_left, top_right),
        )

    first_score = _piece_score(camera_frame, candidates[0])
    second_score = _piece_score(camera_frame, candidates[1])
    selected = candidates[0] if first_score >= second_score else candidates[1]
    print(
        "A4 orientation scores: %d / %d, selected=%d" % (
            first_score, second_score,
            0 if first_score >= second_score else 1))
    return selected


def estimate_paper_color(bird):
    """Measure paper colour only in the initially empty lower A4 half."""
    channels = ([], [], [])
    # Stay at least 1 cm away from the split line and A4 borders.  The E-problem
    # guarantees that pieces are initially placed in the upper half, so these
    # samples cannot be contaminated even when pieces cover most of that half.
    y_start = int((DIVIDER_Y_CM + 1.0) * PAPER_PIXELS_PER_CM)
    y_end = int((PAPER_HEIGHT_CM - 1.0) * PAPER_PIXELS_PER_CM)
    for row in range(1, 6):
        y = y_start + int((y_end - y_start) * row / 6)
        for column in range(1, 8):
            x = int(BIRD_W * column / 8)
            pixel = bird[y, x]
            channels[0].append(int(pixel[0]))
            channels[1].append(int(pixel[1]))
            channels[2].append(int(pixel[2]))
    result = []
    for values in channels:
        values.sort()
        result.append(values[len(values) // 2])
    return tuple(result)


def _approx_piece(contour):
    perimeter = cv2.arcLength(contour, True)
    for ratio in (0.012, 0.018, 0.025, 0.035, 0.050):
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if 3 <= len(approx) <= 5:
            return approx
    return None


def _remove_short_false_edges(polygon):
    """Merge tiny bevels/reflection notches while preserving 3-5 real edges."""
    result = polygon[:]
    while len(result) > 3:
        shortest_length = 1000000.0
        shortest_index = -1
        for index in range(len(result)):
            a = result[index]
            b = result[(index + 1) % len(result)]
            length = math.sqrt(
                (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            if length < shortest_length:
                shortest_length = length
                shortest_index = index
        if shortest_length >= MIN_DETECTED_EDGE_CM:
            break
        # The short segment is normally a clipped/noisy corner.  Removing its
        # second vertex restores the intended intersection of neighbouring edges.
        del result[(shortest_index + 1) % len(result)]
    return normalize_polygon(result)


def _find_candidate_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _contour_center_px(contour):
    x, y, width, height = cv2.boundingRect(contour)
    return x + width * 0.5, y + height * 0.5


def _merge_contour_candidates(primary, secondary):
    """Keep edge-only candidates unless colour detection found the same piece."""
    merged = list(primary)
    for candidate in secondary:
        center = _contour_center_px(candidate)
        area = cv2.contourArea(candidate)
        duplicate = False
        for existing in merged:
            other_center = _contour_center_px(existing)
            other_area = cv2.contourArea(existing)
            center_distance = math.sqrt(
                (center[0] - other_center[0]) ** 2 +
                (center[1] - other_center[1]) ** 2)
            area_ratio = (
                min(area, other_area) / max(area, other_area)
                if max(area, other_area) > 0 else 0.0)
            if (center_distance <= 0.8 * PAPER_PIXELS_PER_CM and
                    area_ratio >= 0.45):
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    return merged


def detect_pieces(bird):
    upper = bird[0:UPPER_LIMIT_PX, :, :]
    background = estimate_paper_color(bird)
    if DETECTION_MODE == "gray_range":
        piece_mask = gray_range_mask(upper)
    elif DETECTION_MODE == "non_white":
        white_lower = np.array(
            [WHITE_MIN_CHANNEL, WHITE_MIN_CHANNEL, WHITE_MIN_CHANNEL],
            dtype=np.uint8)
        white_upper = np.array([255, 255, 255], dtype=np.uint8)
        paper_mask = cv2.inRange(upper, white_lower, white_upper)
        _, piece_mask = cv2.threshold(
            paper_mask, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        tolerance = BACKGROUND_CHANNEL_TOLERANCE
        lower = np.array(
            [max(0, value - tolerance) for value in background],
            dtype=np.uint8)
        upper_bound = np.array(
            [min(255, value + tolerance) for value in background],
            dtype=np.uint8)
        paper_mask = cv2.inRange(upper, lower, upper_bound)
        _, piece_mask = cv2.threshold(
            paper_mask, 127, 255, cv2.THRESH_BINARY_INV)
    # Separate pieces that are only one or two pixels apart/touching after warp.
    # RETR_EXTERNAL ignores holes inside a metal piece, so a large close kernel
    # is unnecessary and would incorrectly bridge neighbouring pieces.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
    piece_mask = cv2.morphologyEx(
        piece_mask, cv2.MORPH_OPEN, open_kernel)
    piece_mask = cv2.morphologyEx(
        piece_mask, cv2.MORPH_CLOSE, close_kernel)
    clear_mask_border(piece_mask)

    color_contours = _find_candidate_contours(piece_mask)

    edge_mask = None
    edge_contours = []
    if EDGE_ASSIST_ENABLE:
        # Zinc sheet and bare metal can be nearly the same colour as the A4
        # background, but their physical outline still gives a strong gradient.
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_mask = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edge_mask = cv2.morphologyEx(
            edge_mask, cv2.MORPH_CLOSE, edge_kernel)
        edge_mask = cv2.dilate(edge_mask, open_kernel)
        edge_contours = _find_candidate_contours(edge_mask)

    contours = _merge_contour_candidates(color_contours, edge_contours)
    minimum_area = MIN_PIECE_AREA_CM2 * PAPER_PIXELS_PER_CM ** 2
    maximum_area = MAX_PIECE_AREA_CM2 * PAPER_PIXELS_PER_CM ** 2
    detected = []
    detected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (minimum_area <= area <= maximum_area):
            continue
        approx = _approx_piece(contour)
        if approx is None:
            continue
        pixel_points = [_point_xy(point) for point in approx]
        polygon = merge_collinear_vertices(
            _remove_short_false_edges(normalize_polygon([
                (point[0] / PAPER_PIXELS_PER_CM,
                 point[1] / PAPER_PIXELS_PER_CM)
                for point in pixel_points
            ])),
            COLLINEAR_MERGE_DEG)
        edge_is_too_short = False
        for index in range(len(polygon)):
            a = polygon[index]
            b = polygon[(index + 1) % len(polygon)]
            length = math.sqrt(
                (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            if length < MIN_DETECTED_EDGE_CM:
                edge_is_too_short = True
                break
        if edge_is_too_short:
            continue
        detected.append(polygon)
        detected_contours.append(approx)

    order = sorted(
        range(len(detected)),
        key=lambda index: (
            polygon_centroid(detected[index])[1],
            polygon_centroid(detected[index])[0],
        ))
    detected = [detected[index] for index in order]
    detected_contours = [detected_contours[index] for index in order]
    return (
        detected, detected_contours, piece_mask, background,
        len(color_contours), len(edge_contours))


def sample_edge_descriptors(bird, polygons):
    """Sample image content just inside each polygon edge."""
    all_descriptors = []
    offset_px = max(2, int(0.14 * PAPER_PIXELS_PER_CM))
    for polygon in polygons:
        center = polygon_centroid(polygon)
        piece_descriptors = []
        for edge_index in range(len(polygon)):
            a = polygon[edge_index]
            b = polygon[(edge_index + 1) % len(polygon)]
            descriptor = []
            for sample_index in range(1, 8):
                fraction = sample_index / 8.0
                x_cm = a[0] + (b[0] - a[0]) * fraction
                y_cm = a[1] + (b[1] - a[1]) * fraction
                toward_x = center[0] - x_cm
                toward_y = center[1] - y_cm
                magnitude = math.sqrt(toward_x ** 2 + toward_y ** 2)
                if magnitude > 0.0001:
                    toward_x /= magnitude
                    toward_y /= magnitude
                x = int(x_cm * PAPER_PIXELS_PER_CM +
                        toward_x * offset_px + 0.5)
                y = int(y_cm * PAPER_PIXELS_PER_CM +
                        toward_y * offset_px + 0.5)
                x = min(max(x, 1), BIRD_W - 2)
                y = min(max(y, 1), BIRD_H - 2)
                sums = [0, 0, 0]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        pixel = bird[y + dy, x + dx]
                        for channel in range(3):
                            sums[channel] += int(pixel[channel])
                descriptor.append((
                    sums[0] // 9, sums[1] // 9, sums[2] // 9))
            piece_descriptors.append(descriptor)
        all_descriptors.append(piece_descriptors)
    return all_descriptors


def centres_are_stable(previous, current):
    if previous is None or len(previous) != len(current):
        return False
    for first, second in zip(previous, current):
        if len(first) != len(second):
            return False
        a = polygon_centroid(first)
        b = polygon_centroid(second)
        drift = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        if drift > DETECTION_MAX_CENTRE_DRIFT_CM:
            return False
    return True


def print_polygon_geometry(polygons):
    for piece_id, polygon in enumerate(polygons):
        lengths = []
        for index in range(len(polygon)):
            a = polygon[index]
            b = polygon[(index + 1) % len(polygon)]
            lengths.append(math.sqrt(
                (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
        print(
            "AVG P%d v=%d area=%.2f edges=%s" % (
                piece_id, len(polygon), polygon_area(polygon),
                "/".join("%.2f" % value for value in lengths)))
        print(
            "AVG P%d points=%s" % (
                piece_id,
                ";".join(
                    [
                        "%.2f,%.2f" % (point[0], point[1])
                        for point in polygon
                    ])))


def solver_size_options():
    if STRICT_CONTEST_TARGET_SIZE:
        return {
            "target_long_min_cm": 8.5,
            "target_long_max_cm": 12.5,
            "target_short_min_cm": 4.5,
            "target_short_max_cm": 9.5,
        }
    if USE_EXPECTED_TARGET_SIZE:
        expected_long = max(
            EXPECTED_TARGET_WIDTH_CM, EXPECTED_TARGET_HEIGHT_CM)
        expected_short = min(
            EXPECTED_TARGET_WIDTH_CM, EXPECTED_TARGET_HEIGHT_CM)
        return {
            "target_long_min_cm": max(
                1.0, expected_long - TARGET_SIZE_TOLERANCE_CM),
            "target_long_max_cm":
                expected_long + TARGET_SIZE_TOLERANCE_CM,
            "target_short_min_cm": max(
                1.0, expected_short - TARGET_SIZE_TOLERANCE_CM),
            "target_short_max_cm":
                expected_short + TARGET_SIZE_TOLERANCE_CM,
        }
    return {
        "target_long_min_cm": 4.0,
        "target_long_max_cm": PAPER_WIDTH_CM - 0.5,
        "target_short_min_cm": 3.0,
        "target_short_max_cm": PAPER_HEIGHT_CM * 0.5 - 0.5,
    }


def init_lower_uart():
    if not LOWER_UART_ENABLE:
        return None
    fpioa = FPIOA()
    fpioa.set_function(LOWER_UART_TX_PIN, fpioa.UART2_TXD)
    fpioa.set_function(LOWER_UART_RX_PIN, fpioa.UART2_RXD)
    return UART(
        UART.UART2,
        baudrate=LOWER_UART_BAUDRATE,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )


def transmit_solution(uart, solution):
    for packet in placement_packets(solution):
        print("TX", packet.strip())
        if uart is not None:
            uart.write(packet.encode("ascii"))
            time.sleep_ms(20)


def draw_polygon_cm(frame, polygon, color, thickness=2):
    for index in range(len(polygon)):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        cv2.line(
            frame,
            (int(a[0] * PAPER_PIXELS_PER_CM + 0.5),
             int(a[1] * PAPER_PIXELS_PER_CM + 0.5)),
            (int(b[0] * PAPER_PIXELS_PER_CM + 0.5),
             int(b[1] * PAPER_PIXELS_PER_CM + 0.5)),
            color, thickness)


def draw_detected_pieces(frame, polygons):
    """Outline every detected source piece and mark its pick centre."""
    for piece_id, polygon in enumerate(polygons):
        color = TARGET_COLORS[piece_id % len(TARGET_COLORS)]
        draw_polygon_cm(frame, polygon, color, 3)
        center = polygon_centroid(polygon)
        point = (
            int(center[0] * PAPER_PIXELS_PER_CM),
            int(center[1] * PAPER_PIXELS_PER_CM))
        cv2.circle(frame, point, 5, color, -1)
        cv2.putText(
            frame, "P%d" % piece_id,
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_direction_arrow(frame, center_cm, rotation_deg, color):
    """Draw where the source image's upward direction goes after placement."""
    angle = math.radians(rotation_deg)
    # In the image coordinate system +y points down, so source UP is (0, -1).
    direction_x = math.sin(angle)
    direction_y = -math.cos(angle)
    length_px = TARGET_DIRECTION_ARROW_CM * PAPER_PIXELS_PER_CM
    start = (
        int(center_cm[0] * PAPER_PIXELS_PER_CM + 0.5),
        int(center_cm[1] * PAPER_PIXELS_PER_CM + 0.5),
    )
    end = (
        int(start[0] + direction_x * length_px + 0.5),
        int(start[1] + direction_y * length_px + 0.5),
    )
    cv2.line(frame, start, end, color, 2)

    head_length = max(5, int(length_px * 0.32))
    perpendicular_x = -direction_y
    perpendicular_y = direction_x
    left = (
        int(end[0] - direction_x * head_length +
            perpendicular_x * head_length * 0.48),
        int(end[1] - direction_y * head_length +
            perpendicular_y * head_length * 0.48),
    )
    right = (
        int(end[0] - direction_x * head_length -
            perpendicular_x * head_length * 0.48),
        int(end[1] - direction_y * head_length -
            perpendicular_y * head_length * 0.48),
    )
    cv2.line(frame, end, left, color, 2)
    cv2.line(frame, end, right, color, 2)
    cv2.putText(
        frame, "UP", (end[0] + 3, end[1] - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)


def draw_solution_preview(frame, solution):
    """Draw the planned puzzle in the lower A4 area with orientation marks."""
    all_points = []
    for placement in solution["placements"]:
        piece_id = placement["piece_id"]
        color = TARGET_COLORS[piece_id % len(TARGET_COLORS)]
        polygon = placement["target_polygon_cm"]
        center = placement["target_center_cm"]
        rotation = placement["rotation_deg"]
        all_points.extend(polygon)

        draw_polygon_cm(frame, polygon, color, 3)
        center_px = (
            int(center[0] * PAPER_PIXELS_PER_CM + 0.5),
            int(center[1] * PAPER_PIXELS_PER_CM + 0.5),
        )
        cv2.circle(frame, center_px, 4, color, -1)
        draw_direction_arrow(frame, center, rotation, color)
        cv2.putText(
            frame, "P%d %+.0fdeg" % (piece_id, rotation),
            (center_px[0] + 5, center_px[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)

    # Draw the final rectangle and a global A4-up reference arrow.
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    cv2.rectangle(
        frame,
        (int(min_x * PAPER_PIXELS_PER_CM),
         int(min_y * PAPER_PIXELS_PER_CM)),
        (int(max_x * PAPER_PIXELS_PER_CM),
         int(max_y * PAPER_PIXELS_PER_CM)),
        (255, 255, 255), 1)
    reference_x = max_x + 0.65
    if reference_x > PAPER_WIDTH_CM - 0.35:
        reference_x = min_x - 0.65
    draw_direction_arrow(
        frame, (reference_x, (min_y + max_y) * 0.5),
        0.0, (255, 255, 255))


def show_frame(frame, is_birdview=True):
    if is_birdview:
        show_h = DISPLAY_H
        show_w = int(BIRD_W * show_h / BIRD_H)
    else:
        show_w = int(CAMERA_W * DISPLAY_H / CAMERA_H)
        show_h = DISPLAY_H
    resized = cv2.resize(frame, (show_w, show_h))
    DISPLAY_CANVAS[:, :, :] = 0
    display_x = (DISPLAY_W - show_w) // 2
    DISPLAY_CANVAS[
        0:show_h, display_x:display_x + show_w, :] = resized
    show_image = image.Image(
        DISPLAY_W, DISPLAY_H, image.RGB888,
        alloc=image.ALLOC_REF, data=DISPLAY_CANVAS)
    # Always use the same VO layer attributes.
    Display.show_image(show_image, x=0, y=0)
    del show_image
    del resized


def main():
    sensor = Sensor(width=1280, height=960, fps=CAMERA_FPS)
    lower_uart = None
    homography = None
    locked_quad = None
    previous_quad = None
    quad_stable_count = 0
    calibration_frames = 0
    previous_polygons = None
    stable_count = 0
    stable_polygon_frames = []
    averaged_source_polygons = None
    solution = None
    frame_count = 0
    clock = time.clock()

    try:
        os.exitpoint(os.EXITPOINT_ENABLE)
        sensor.reset()
        sensor.set_framesize(width=CAMERA_W, height=CAMERA_H)
        sensor.set_pixformat(Sensor.RGB888)
        sensor.set_hmirror(CAMERA_HMIRROR)
        sensor.set_vflip(CAMERA_VFLIP)
        Display.init(
            DISPLAY_TYPE, width=DISPLAY_W, height=DISPLAY_H,
            fps=CAMERA_FPS, to_ide=DISPLAY_TO_IDE)
        MediaManager.init()
        sensor.run()
        time.sleep(0.5)
        lower_uart = init_lower_uart()

        while True:
            os.exitpoint()
            clock.tick()
            camera_image = sensor.snapshot()
            camera_frame = camera_image.to_numpy_ref()
            frame_count += 1

            if homography is None:
                calibration_frames += 1
                quad = (
                    detect_paper_quad(camera_frame)
                    if AUTO_PAPER_DETECT else MANUAL_PAPER_POINTS)
                if quad is not None:
                    drift = quad_max_drift(previous_quad, quad)
                    if drift <= AUTO_PAPER_MAX_DRIFT_PX:
                        quad_stable_count += 1
                    else:
                        quad_stable_count = 1
                    previous_quad = quad
                    for index in range(4):
                        cv2.line(
                            camera_frame, quad[index],
                            quad[(index + 1) % 4], (0, 255, 0), 3)
                if quad_stable_count >= AUTO_PAPER_STABLE_FRAMES:
                    locked_quad = orient_paper_quad(camera_frame, quad)
                elif calibration_frames >= AUTO_PAPER_TIMEOUT_FRAMES:
                    locked_quad = orient_paper_quad(
                        camera_frame, MANUAL_PAPER_POINTS)
                    print("paper auto calibration timeout; use manual points")
                if locked_quad is not None:
                    homography = make_paper_homography(locked_quad)
                    print("paper locked:", locked_quad)
                    print(
                        "paper=%.1fx%.1fcm expected target=%.1fx%.1fcm" % (
                            PAPER_WIDTH_CM, PAPER_HEIGHT_CM,
                            EXPECTED_TARGET_WIDTH_CM,
                            EXPECTED_TARGET_HEIGHT_CM))
                    print(
                        "resolution camera=%dx%d bird=%dx%d %.1fpx/cm" % (
                            CAMERA_W, CAMERA_H, BIRD_W, BIRD_H,
                            PAPER_PIXELS_PER_CM))
                cv2.putText(
                    camera_frame, "CALIBRATING A4 %d/%d" % (
                        quad_stable_count, AUTO_PAPER_STABLE_FRAMES),
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
                show_frame(camera_frame, False)
                del camera_image
                gc.collect()
                continue

            bird = cv2.warpPerspective(
                camera_frame, homography, (BIRD_W, BIRD_H))
            (polygons, contours, piece_mask, background,
             color_candidate_count,
             edge_candidate_count) = detect_pieces(bird)
            cv2.line(
                bird, (0, int(DIVIDER_Y_CM * PAPER_PIXELS_PER_CM)),
                (BIRD_W - 1, int(DIVIDER_Y_CM * PAPER_PIXELS_PER_CM)),
                (0, 255, 255), 2)

            if solution is None and 2 <= len(polygons) <= 4:
                if centres_are_stable(previous_polygons, polygons):
                    stable_count += 1
                else:
                    stable_count = 1
                    stable_polygon_frames = []
                previous_polygons = polygons
                stable_polygon_frames.append(polygons)
                if len(stable_polygon_frames) > DETECTION_AVERAGE_FRAMES:
                    stable_polygon_frames.pop(0)
                if stable_count >= DETECTION_STABLE_FRAMES:
                    averaged_polygons = average_polygon_frames(
                        stable_polygon_frames)
                    averaged_source_polygons = averaged_polygons
                    print_polygon_geometry(averaged_polygons)
                    descriptors = (
                        sample_edge_descriptors(bird, averaged_polygons)
                        if USE_EDGE_TEXTURE_SCORE else None)
                    solve_options = solver_size_options()
                    solve_options.update({
                        "target_center_cm": TARGET_CENTER_CM,
                        "edge_descriptors": descriptors,
                        "collinear_merge_deg": COLLINEAR_MERGE_DEG,
                        "max_gap_ratio": SOLVER_MAX_GAP_RATIO,
                        "max_missing_outer_pieces":
                            SOLVER_MAX_MISSING_OUTER_PIECES,
                        "max_camera_pair_overlap_cm2":
                            SOLVER_MAX_CAMERA_PAIR_OVERLAP_CM2,
                        "max_camera_total_overlap_ratio":
                            SOLVER_MAX_CAMERA_TOTAL_OVERLAP_RATIO,
                        "partial_fast_path_only":
                            SOLVER_PARTIAL_FAST_PATH_ONLY,
                        "allow_partial_edge_matches": False,
                    })
                    draw_detected_pieces(bird, averaged_polygons)
                    cv2.putText(
                        bird, "SOLVING 30-FRAME AVERAGE...",
                        (18, BIRD_H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 0, 255), 2)
                    show_frame(bird, True)
                    print(
                        "SOLVING target long %.1f..%.1f short %.1f..%.1f" % (
                            solve_options["target_long_min_cm"],
                            solve_options["target_long_max_cm"],
                            solve_options["target_short_min_cm"],
                            solve_options["target_short_max_cm"]))
                    print("SOLVING stage 1: equal full edges")
                    solution = solve_puzzle(averaged_polygons, solve_options)
                    if (solution is None and
                            ENABLE_PARTIAL_EDGE_FALLBACK):
                        print(
                            "full-edge solve failed; "
                            "stage 2: partial/T-junction edges")
                        solve_options["allow_partial_edge_matches"] = True
                        solution = solve_puzzle(
                            averaged_polygons, solve_options)
                    if solution is None:
                        print("pieces stable, but no valid rectangle solution")
                        stable_count = 0
                        stable_polygon_frames = []
                        averaged_source_polygons = None
                    else:
                        print(
                            "SOLVED avg=%d %.2fx%.2fcm gap=%.3f "
                            "overlap=%.3fcm2 snaps=%d "
                            "outer=%d/%d texture=%.3f" % (
                                len(stable_polygon_frames),
                                solution["target_width_cm"],
                                solution["target_height_cm"],
                                solution["gap_ratio"],
                                solution["overlap_area_cm2"],
                                solution["snapped_seams"],
                                solution["outer_piece_count"],
                                len(solution["placements"]),
                                solution["texture_cost"]))
                        transmit_solution(lower_uart, solution)
            else:
                previous_polygons = polygons if polygons else None
                if solution is None:
                    stable_count = 0
                    stable_polygon_frames = []
                    averaged_source_polygons = None

            source_polygons_to_draw = (
                averaged_source_polygons
                if averaged_source_polygons is not None else polygons)
            draw_detected_pieces(bird, source_polygons_to_draw)

            if solution is not None:
                draw_solution_preview(bird, solution)
                cv2.putText(
                    bird, "SOLVED %.1fx%.1fcm" % (
                        solution["target_width_cm"],
                        solution["target_height_cm"]),
                    (8, BIRD_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
            else:
                cv2.putText(
                    bird, "P=%d C=%d E=%d stable=%d/%d bg=%d/%d/%d" % (
                        len(polygons),
                        color_candidate_count, edge_candidate_count,
                        stable_count, DETECTION_STABLE_FRAMES,
                        background[0], background[1], background[2]),
                    (8, BIRD_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 0, 0), 1)

            show_frame(bird, True)
            if LOG_INTERVAL_FRAMES and frame_count % LOG_INTERVAL_FRAMES == 0:
                print(
                    "fps=%.1f pieces=%d stable=%d/%d solved=%s" % (
                        clock.fps(), len(polygons), stable_count,
                        DETECTION_STABLE_FRAMES,
                        str(solution is not None)))

            del piece_mask
            del contours
            del polygons
            del bird
            del camera_image
            gc.collect()

    except KeyboardInterrupt:
        print("user stop")
    except BaseException as error:
        print("puzzle exception:", error)
        raise
    finally:
        if lower_uart is not None:
            lower_uart.deinit()
        sensor.stop()
        Display.deinit()
        time.sleep_ms(100)
        MediaManager.deinit()


if __name__ == "__main__":
    main()
