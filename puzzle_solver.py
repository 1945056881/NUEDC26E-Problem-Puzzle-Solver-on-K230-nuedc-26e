"""Small, dependency-free polygon puzzle solver for the K230 puzzle device.

The solver only uses straight polygon edges.  It is intentionally independent
from OpenCV so the same code can be unit-tested on a PC and imported by
CanMV/MicroPython on the K230.
"""

import math


DEFAULT_OPTIONS = {
    "edge_abs_tolerance_cm": 0.48,
    "edge_rel_tolerance": 0.11,
    "overlap_epsilon_cm": 0.12,
    # Small overlap between averaged camera contours is measurement noise.
    # These limits are used only by the fast partial/T-junction layout.
    "max_camera_pair_overlap_cm2": 0.65,
    "max_camera_total_overlap_ratio": 0.015,
    "adjacent_snap_abs_tolerance_cm": 0.55,
    "adjacent_snap_rel_tolerance": 0.07,
    "max_gap_ratio": 0.18,
    "outer_edge_tolerance_cm": 0.45,
    # Camera contours can move one reconstructed outside edge away from the
    # fitted rectangle.  Keep it as a scored defect instead of rejecting an
    # otherwise gap-free assembly outright.
    "max_missing_outer_pieces": 1,
    "missing_outer_piece_penalty": 0.02,
    "target_center_cm": (10.5, 22.3),
    "target_long_min_cm": 8.5,
    "target_long_max_cm": 12.5,
    "target_short_min_cm": 4.5,
    "target_short_max_cm": 9.5,
    "edge_descriptors": None,
    "texture_weight": 0.08,
    "max_texture_distance": 0.65,
    "collinear_merge_deg": 15.0,
    "allow_partial_edge_matches": True,
    "partial_fast_path_only": False,
    "partial_edge_min_ratio": 0.22,
    "compound_edge_abs_tolerance_cm": 0.55,
    "compound_edge_rel_tolerance": 0.05,
    "early_stop_score": 0.035,
    "search_beam_width": 64,
}


def _cross(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0]))


def _distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def polygon_area_signed(poly):
    value = 0.0
    for index in range(len(poly)):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % len(poly)]
        value += x1 * y2 - x2 * y1
    return value * 0.5


def polygon_area(poly):
    return abs(polygon_area_signed(poly))


def polygon_centroid(poly):
    signed_area = polygon_area_signed(poly)
    if abs(signed_area) < 0.000001:
        return (
            sum(point[0] for point in poly) / len(poly),
            sum(point[1] for point in poly) / len(poly),
        )
    factor_sum_x = 0.0
    factor_sum_y = 0.0
    for index in range(len(poly)):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % len(poly)]
        factor = x1 * y2 - x2 * y1
        factor_sum_x += (x1 + x2) * factor
        factor_sum_y += (y1 + y2) * factor
    divisor = 6.0 * signed_area
    return factor_sum_x / divisor, factor_sum_y / divisor


def normalize_polygon(poly):
    """Remove repeated points and force every polygon to the same winding."""
    cleaned = []
    for raw_point in poly:
        point = (float(raw_point[0]), float(raw_point[1]))
        if not cleaned or _distance(cleaned[-1], point) > 0.001:
            cleaned.append(point)
    if len(cleaned) > 1 and _distance(cleaned[0], cleaned[-1]) <= 0.001:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ValueError("a puzzle piece needs at least three vertices")
    if polygon_area_signed(cleaned) < 0:
        cleaned.reverse()
    return cleaned


def merge_collinear_vertices(poly, angle_tolerance_deg=15.0):
    """Merge adjacent edges when their included angle is almost 180 degrees."""
    result = normalize_polygon(poly)
    changed = True
    while changed and len(result) > 3:
        changed = False
        best_index = -1
        best_deviation = 1000000.0
        for index in range(len(result)):
            previous = result[index - 1]
            current = result[index]
            following = result[(index + 1) % len(result)]
            first = (
                previous[0] - current[0],
                previous[1] - current[1])
            second = (
                following[0] - current[0],
                following[1] - current[1])
            first_length = math.sqrt(first[0] ** 2 + first[1] ** 2)
            second_length = math.sqrt(second[0] ** 2 + second[1] ** 2)
            if first_length < 0.0001 or second_length < 0.0001:
                best_index = index
                best_deviation = 0.0
                break
            cosine = (
                first[0] * second[0] + first[1] * second[1]) / (
                    first_length * second_length)
            cosine = max(-1.0, min(1.0, cosine))
            included_angle = math.degrees(math.acos(cosine))
            deviation = abs(180.0 - included_angle)
            if deviation < best_deviation:
                best_deviation = deviation
                best_index = index
        if best_index >= 0 and best_deviation <= angle_tolerance_deg:
            del result[best_index]
            changed = True
    return normalize_polygon(result)


def align_polygon_vertices(reference, candidate):
    """Cyclically align equal-topology polygons for temporal averaging."""
    reference = normalize_polygon(reference)
    candidate = normalize_polygon(candidate)
    if len(reference) != len(candidate):
        raise ValueError("polygon vertex counts differ")
    best = None
    for shift in range(len(candidate)):
        shifted = (
            candidate[shift:] + candidate[:shift])
        error = 0.0
        for first, second in zip(reference, shifted):
            error += (
                (first[0] - second[0]) ** 2 +
                (first[1] - second[1]) ** 2)
        if best is None or error < best[0]:
            best = (error, shifted)
    return best[1]


def average_polygon_frames(frames):
    """Average matching piece vertices over multiple stable detections."""
    if not frames:
        raise ValueError("no polygon frames to average")
    piece_count = len(frames[0])
    if piece_count == 0:
        raise ValueError("no pieces to average")
    reference = [
        normalize_polygon(poly) for poly in frames[0]]
    sums = [
        [[point[0], point[1]] for point in poly]
        for poly in reference
    ]
    for frame in frames[1:]:
        if len(frame) != piece_count:
            raise ValueError("piece counts differ")
        for piece_index in range(piece_count):
            aligned = align_polygon_vertices(
                reference[piece_index], frame[piece_index])
            for vertex_index, point in enumerate(aligned):
                sums[piece_index][vertex_index][0] += point[0]
                sums[piece_index][vertex_index][1] += point[1]
    divisor = float(len(frames))
    return [
        [
            (point[0] / divisor, point[1] / divisor)
            for point in piece_sums
        ]
        for piece_sums in sums
    ]


def transform_polygon(poly, pose):
    angle, tx, ty = pose
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        (cosine * x - sine * y + tx,
         sine * x + cosine * y + ty)
        for x, y in poly
    ]


def _point_in_polygon(point, poly, epsilon):
    """Return 1 inside, 0 on the boundary, and -1 outside."""
    px, py = point
    inside = False
    for index in range(len(poly)):
        a = poly[index]
        b = poly[(index + 1) % len(poly)]
        cross_value = abs(_cross(a, b, point))
        if cross_value <= epsilon * max(1.0, _distance(a, b)):
            if (min(a[0], b[0]) - epsilon <= px <=
                    max(a[0], b[0]) + epsilon and
                    min(a[1], b[1]) - epsilon <= py <=
                    max(a[1], b[1]) + epsilon):
                return 0
        if ((a[1] > py) != (b[1] > py)):
            hit_x = (
                (b[0] - a[0]) * (py - a[1]) /
                (b[1] - a[1]) + a[0])
            if px < hit_x:
                inside = not inside
    return 1 if inside else -1


def _proper_intersection(a, b, c, d, epsilon):
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    return (
        ab_c * ab_d < -(epsilon ** 2) and
        cd_a * cd_b < -(epsilon ** 2)
    )


def polygons_overlap(poly_a, poly_b, epsilon=0.025):
    """True only for positive-area overlap; touching edges are allowed."""
    for index_a in range(len(poly_a)):
        a = poly_a[index_a]
        b = poly_a[(index_a + 1) % len(poly_a)]
        for index_b in range(len(poly_b)):
            c = poly_b[index_b]
            d = poly_b[(index_b + 1) % len(poly_b)]
            if _proper_intersection(a, b, c, d, epsilon):
                return True
    for point in poly_a:
        if _point_in_polygon(point, poly_b, epsilon) == 1:
            return True
    for point in poly_b:
        if _point_in_polygon(point, poly_a, epsilon) == 1:
            return True
    if _point_in_polygon(polygon_centroid(poly_a), poly_b, epsilon) == 1:
        return True
    if _point_in_polygon(polygon_centroid(poly_b), poly_a, epsilon) == 1:
        return True
    return False


def convex_overlap_area(poly_a, poly_b):
    """Return the intersection area of two convex CCW polygons."""
    output = [point for point in poly_a]
    for clip_index in range(len(poly_b)):
        clip_a = poly_b[clip_index]
        clip_b = poly_b[(clip_index + 1) % len(poly_b)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _cross(clip_a, clip_b, previous) >= -0.000001
        for current in input_points:
            current_inside = (
                _cross(clip_a, clip_b, current) >= -0.000001)
            if current_inside != previous_inside:
                segment_x = current[0] - previous[0]
                segment_y = current[1] - previous[1]
                clip_x = clip_b[0] - clip_a[0]
                clip_y = clip_b[1] - clip_a[1]
                denominator = segment_x * clip_y - segment_y * clip_x
                if abs(denominator) > 0.000001:
                    ratio = (
                        (clip_a[0] - previous[0]) * clip_y -
                        (clip_a[1] - previous[1]) * clip_x
                    ) / denominator
                    output.append((
                        previous[0] + ratio * segment_x,
                        previous[1] + ratio * segment_y))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    return polygon_area(output) if len(output) >= 3 else 0.0


def _pose_matching_edges(fixed_a, fixed_b, moving_a, moving_b):
    """Align reversed edge directions and their midpoints without scaling."""
    source_angle = math.atan2(
        moving_b[1] - moving_a[1], moving_b[0] - moving_a[0])
    target_angle = math.atan2(
        fixed_a[1] - fixed_b[1], fixed_a[0] - fixed_b[0])
    angle = target_angle - source_angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    moving_mid_x = (moving_a[0] + moving_b[0]) * 0.5
    moving_mid_y = (moving_a[1] + moving_b[1]) * 0.5
    fixed_mid_x = (fixed_a[0] + fixed_b[0]) * 0.5
    fixed_mid_y = (fixed_a[1] + fixed_b[1]) * 0.5
    rotated_x = cosine * moving_mid_x - sine * moving_mid_y
    rotated_y = sine * moving_mid_x + cosine * moving_mid_y
    return (
        angle,
        fixed_mid_x - rotated_x,
        fixed_mid_y - rotated_y,
    )


def _edge_alignment_candidates(
        fixed_a, fixed_b, moving_a, moving_b, tolerance, options,
        partial_offsets=None):
    """Return full-edge and endpoint-aligned partial-edge attachments."""
    fixed_length = _distance(fixed_a, fixed_b)
    moving_length = _distance(moving_a, moving_b)
    difference = abs(fixed_length - moving_length)
    if difference <= tolerance:
        return [(
            _pose_matching_edges(
                fixed_a, fixed_b, moving_a, moving_b),
            True, True, False)]
    if not options["allow_partial_edge_matches"]:
        return []
    shorter = min(fixed_length, moving_length)
    longer = max(fixed_length, moving_length)
    if longer < 0.0001 or (
            shorter / longer < options["partial_edge_min_ratio"]):
        return []

    candidates = []
    if fixed_length > moving_length:
        unit_x = (fixed_b[0] - fixed_a[0]) / fixed_length
        unit_y = (fixed_b[1] - fixed_a[1]) / fixed_length
        offsets = (
            partial_offsets if partial_offsets is not None
            else (0.0, fixed_length - moving_length))
        fixed_segments = []
        seen_offsets = set()
        for offset in offsets:
            offset_key = int(round(offset * 1000.0))
            if offset_key in seen_offsets:
                continue
            seen_offsets.add(offset_key)
            fixed_segments.append((
                (fixed_a[0] + unit_x * offset,
                 fixed_a[1] + unit_y * offset),
                (fixed_a[0] + unit_x * (offset + moving_length),
                 fixed_a[1] + unit_y * (offset + moving_length)),
            ))
        for segment_a, segment_b in fixed_segments:
            candidates.append((
                _pose_matching_edges(
                    segment_a, segment_b, moving_a, moving_b),
                False, True, True))
    else:
        unit_x = (moving_b[0] - moving_a[0]) / moving_length
        unit_y = (moving_b[1] - moving_a[1]) / moving_length
        offsets = (
            partial_offsets if partial_offsets is not None
            else (0.0, moving_length - fixed_length))
        moving_segments = []
        seen_offsets = set()
        for offset in offsets:
            offset_key = int(round(offset * 1000.0))
            if offset_key in seen_offsets:
                continue
            seen_offsets.add(offset_key)
            moving_segments.append((
                (moving_a[0] + unit_x * offset,
                 moving_a[1] + unit_y * offset),
                (moving_a[0] + unit_x * (offset + fixed_length),
                 moving_a[1] + unit_y * (offset + fixed_length)),
            ))
        for segment_a, segment_b in moving_segments:
            candidates.append((
                _pose_matching_edges(
                    fixed_a, fixed_b, segment_a, segment_b),
                True, False, True))
    return candidates


def _canonical_edge_pair(first, second):
    return (first, second) if first < second else (second, first)


def _small_permutations(values):
    if len(values) <= 1:
        return [values[:]]
    result = []
    for index in range(len(values)):
        head = values[index]
        remaining = values[:index] + values[index + 1:]
        for tail in _small_permutations(remaining):
            result.append([head] + tail)
    return result


def find_compound_edge_layouts(polygons, options):
    """Find long edges composed of two or three shorter piece edges."""
    references = []
    for piece_index, polygon in enumerate(polygons):
        for edge_index in range(len(polygon)):
            a = polygon[edge_index]
            b = polygon[(edge_index + 1) % len(polygon)]
            references.append((
                (piece_index, edge_index), _distance(a, b)))

    allowed_pairs = set()
    relations = []
    offsets_by_pair = {}

    def record_relation(long_ref, long_length, short_items):
        short_sum = sum(item[1] for item in short_items)
        tolerance = max(
            options["compound_edge_abs_tolerance_cm"],
            options["compound_edge_rel_tolerance"] * long_length)
        error = abs(long_length - short_sum)
        if error > tolerance:
            return
        relations.append((
            long_ref, long_length,
            tuple(short_items), short_sum, error))
        for short_ref, _ in short_items:
            allowed_pairs.add(
                _canonical_edge_pair(long_ref, short_ref))

        # Distribute the measured total-length error over both ends.  Every
        # ordering is possible until polygon overlap/rectangle scoring prunes it.
        base_offset = (long_length - short_sum) * 0.5
        for ordering in _small_permutations(list(short_items)):
            offset = base_offset
            for short_ref, short_length in ordering:
                key = (long_ref, short_ref)
                values = offsets_by_pair.setdefault(key, [])
                if not any(abs(value - offset) < 0.001 for value in values):
                    values.append(offset)
                offset += short_length

    for long_ref, long_length in references:
        long_piece = long_ref[0]
        short_candidates = [
            item for item in references
            if item[0][0] != long_piece and item[1] < long_length
        ]
        for first_index in range(len(references)):
            first_ref, first_length = references[first_index]
            if (first_ref[0] == long_piece or
                    first_length >= long_length):
                continue
            for second_index in range(first_index + 1, len(references)):
                second_ref, second_length = references[second_index]
                if (second_ref[0] == long_piece or
                        second_ref[0] == first_ref[0] or
                        second_length >= long_length):
                    continue
                record_relation(
                    long_ref, long_length,
                    [(first_ref, first_length),
                     (second_ref, second_length)])

        # A main diagonal can be shared by one edge from each of the other
        # three pieces, as in the four-piece E-problem sample.
        for first_index in range(len(short_candidates)):
            first = short_candidates[first_index]
            for second_index in range(
                    first_index + 1, len(short_candidates)):
                second = short_candidates[second_index]
                if second[0][0] == first[0][0]:
                    continue
                for third_index in range(
                        second_index + 1, len(short_candidates)):
                    third = short_candidates[third_index]
                    if (third[0][0] == first[0][0] or
                            third[0][0] == second[0][0]):
                        continue
                    record_relation(
                        long_ref, long_length,
                        [first, second, third])
    return allowed_pairs, relations, offsets_by_pair


def find_compound_edge_pairs(polygons, options):
    pairs, relations, _ = find_compound_edge_layouts(
        polygons, options)
    return pairs, relations


def _bbox_at_angle(polygons, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    min_x = 1e30
    min_y = 1e30
    max_x = -1e30
    max_y = -1e30
    for poly in polygons:
        for x, y in poly:
            rx = cosine * x - sine * y
            ry = sine * x + cosine * y
            min_x = min(min_x, rx)
            min_y = min(min_y, ry)
            max_x = max(max_x, rx)
            max_y = max(max_y, ry)
    return min_x, min_y, max_x, max_y


def _minimum_edge_rectangle(polygons):
    best = None
    for poly in polygons:
        for index in range(len(poly)):
            a = poly[index]
            b = poly[(index + 1) % len(poly)]
            angle = -math.atan2(b[1] - a[1], b[0] - a[0])
            bounds = _bbox_at_angle(polygons, angle)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            area = width * height
            if best is None or area < best[0]:
                best = (area, angle, bounds)
    if best is None:
        return None
    area, angle, bounds = best
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width < height:
        angle -= math.pi * 0.5
        bounds = _bbox_at_angle(polygons, angle)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area = width * height
    return area, angle, bounds, width, height


def _piece_has_outer_edge(poly, bounds, tolerance):
    min_x, min_y, max_x, max_y = bounds
    for index in range(len(poly)):
        a = poly[index]
        b = poly[(index + 1) % len(poly)]
        if (abs(a[0] - min_x) <= tolerance and
                abs(b[0] - min_x) <= tolerance):
            return True
        if (abs(a[0] - max_x) <= tolerance and
                abs(b[0] - max_x) <= tolerance):
            return True
        if (abs(a[1] - min_y) <= tolerance and
                abs(b[1] - min_y) <= tolerance):
            return True
        if (abs(a[1] - max_y) <= tolerance and
                abs(b[1] - max_y) <= tolerance):
            return True
    return False


def _options_with_defaults(options):
    result = {}
    for key, value in DEFAULT_OPTIONS.items():
        result[key] = value
    if options:
        for key, value in options.items():
            result[key] = value
    return result


def _angle_degrees(angle):
    value = math.degrees(angle)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _descriptor_distance(first, second):
    if first is None or second is None:
        return 0.0
    sample_count = min(len(first), len(second))
    if sample_count == 0:
        return 0.0
    total = 0.0
    values = 0
    for index in range(sample_count):
        left = first[index]
        right = second[sample_count - index - 1]
        channel_count = min(len(left), len(right))
        for channel in range(channel_count):
            total += abs(float(left[channel]) - float(right[channel])) / 255.0
            values += 1
    return total / values if values else 0.0


def solve_puzzle(raw_polygons, options=None):
    """Solve 2-4 straight-edged pieces and return pick/place transforms.

    Input and output coordinates are centimetres in the rectified A4 plane.
    The returned rotation is the physical rotation around each piece centroid.
    """
    opts = _options_with_defaults(options)
    polygons = [
        merge_collinear_vertices(poly, opts["collinear_merge_deg"])
        for poly in raw_polygons
    ]
    piece_count = len(polygons)
    if piece_count < 2 or piece_count > 4:
        raise ValueError("the E-problem requires two to four pieces")
    for poly in polygons:
        if len(poly) < 3 or len(poly) > 5:
            raise ValueError("each detected piece must have 3 to 5 edges")

    areas = [polygon_area(poly) for poly in polygons]
    total_piece_area = sum(areas)
    if total_piece_area <= 0.01:
        raise ValueError("detected puzzle area is invalid")

    poses = [None] * piece_count
    poses[0] = (0.0, 0.0, 0.0)
    placed_polygons = [None] * piece_count
    placed_polygons[0] = polygons[0]
    best = [None]
    seen_by_depth = [set() for _ in range(piece_count + 1)]
    used_edges = set()
    texture_costs = []
    (compound_edge_pairs, compound_relations,
     compound_edge_offsets) = find_compound_edge_layouts(
        polygons, opts)

    def evaluate(overlap_area=0.0, snapped_seams=0):
        assembly = [placed_polygons[index] for index in range(piece_count)]
        rectangle = _minimum_edge_rectangle(assembly)
        if rectangle is None:
            return
        rectangle_area, global_angle, bounds, width, height = rectangle
        if rectangle_area <= 0.01:
            return
        # Subtract duplicated overlap from the covered area.  This prevents a
        # noisy overlap from making the assembly look artificially complete.
        gap_ratio = max(
            0.0,
            (rectangle_area - total_piece_area + overlap_area) /
            rectangle_area)
        if gap_ratio > opts["max_gap_ratio"]:
            return
        if not (opts["target_long_min_cm"] <= width <=
                opts["target_long_max_cm"]):
            return
        if not (opts["target_short_min_cm"] <= height <=
                opts["target_short_max_cm"]):
            return

        aligned = [
            transform_polygon(poly, (global_angle, 0.0, 0.0))
            for poly in assembly
        ]
        aligned_bounds = _bbox_at_angle(assembly, global_angle)
        outer_piece_count = 0
        for poly in aligned:
            if _piece_has_outer_edge(
                    poly, aligned_bounds,
                    opts["outer_edge_tolerance_cm"]):
                outer_piece_count += 1
        missing_outer_pieces = piece_count - outer_piece_count
        if missing_outer_pieces > int(
                opts["max_missing_outer_pieces"]):
            return

        texture_cost = (
            sum(texture_costs) / len(texture_costs)
            if texture_costs else 0.0)
        # Prefer a tight rectangle, then continuous image content across seams.
        score = (
            gap_ratio +
            opts["texture_weight"] * texture_cost +
            missing_outer_pieces *
            opts["missing_outer_piece_penalty"])
        if best[0] is None or score < best[0]["score"]:
            best[0] = {
                "score": score,
                "gap_ratio": gap_ratio,
                "overlap_area_cm2": overlap_area,
                "snapped_seams": snapped_seams,
                "texture_cost": texture_cost,
                "outer_piece_count": outer_piece_count,
                "width_cm": width,
                "height_cm": height,
                "global_angle": global_angle,
                "bounds": aligned_bounds,
                "poses": poses[:],
                "assembly": [poly[:] for poly in assembly],
            }

    def try_main_edge_split():
        """Fast path: one main edge is split across all remaining pieces."""
        def candidate_overlap_area(candidate_polygons):
            total_overlap = 0.0
            for first in range(piece_count):
                for second in range(first + 1, piece_count):
                    overlap = convex_overlap_area(
                        candidate_polygons[first],
                        candidate_polygons[second])
                    if overlap > opts[
                            "max_camera_pair_overlap_cm2"]:
                        return None
                    total_overlap += overlap
            if total_overlap > (
                    total_piece_area *
                    opts["max_camera_total_overlap_ratio"]):
                return None
            return total_overlap

        def evaluate_edge_snaps(
                ordering, chain_index, candidate_poses,
                candidate_polygons, snapped_seams):
            if (best[0] is not None and
                    best[0]["score"] <= opts["early_stop_score"]):
                return
            if chain_index >= len(ordering) - 1:
                overlap_area = candidate_overlap_area(
                    candidate_polygons)
                if overlap_area is None:
                    return
                for index in range(piece_count):
                    poses[index] = candidate_poses[index]
                    placed_polygons[index] = (
                        candidate_polygons[index])
                evaluate(overlap_area, snapped_seams)
                return

            # Consecutive pieces touch at one endpoint of their main-edge
            # segment.  Their two incident edges should form another seam.
            left_piece, left_main_edge, _ = ordering[chain_index]
            right_piece, right_main_edge, _ = ordering[chain_index + 1]
            left_seam = (
                left_main_edge - 1) % len(polygons[left_piece])
            right_seam = (
                right_main_edge + 1) % len(polygons[right_piece])
            left_length = _distance(
                polygons[left_piece][left_seam],
                polygons[left_piece][
                    (left_seam + 1) % len(polygons[left_piece])])
            right_length = _distance(
                polygons[right_piece][right_seam],
                polygons[right_piece][
                    (right_seam + 1) % len(polygons[right_piece])])
            tolerance = max(
                opts["adjacent_snap_abs_tolerance_cm"],
                opts["adjacent_snap_rel_tolerance"] *
                max(left_length, right_length))

            # Keep the original main-edge placement as one candidate.
            evaluate_edge_snaps(
                ordering, chain_index + 1,
                candidate_poses, candidate_polygons, snapped_seams)
            if abs(left_length - right_length) > tolerance:
                return

            # Snap the left piece to the already transformed right seam.
            right_poly = candidate_polygons[right_piece]
            left_pose = _pose_matching_edges(
                right_poly[right_seam],
                right_poly[(right_seam + 1) % len(right_poly)],
                polygons[left_piece][left_seam],
                polygons[left_piece][
                    (left_seam + 1) % len(polygons[left_piece])])
            left_poses = candidate_poses[:]
            left_polygons = [poly[:] for poly in candidate_polygons]
            left_poses[left_piece] = left_pose
            left_polygons[left_piece] = transform_polygon(
                polygons[left_piece], left_pose)
            evaluate_edge_snaps(
                ordering, chain_index + 1,
                left_poses, left_polygons, snapped_seams + 1)

            # Snap the right piece to the already transformed left seam.
            left_poly = candidate_polygons[left_piece]
            right_pose = _pose_matching_edges(
                left_poly[left_seam],
                left_poly[(left_seam + 1) % len(left_poly)],
                polygons[right_piece][right_seam],
                polygons[right_piece][
                    (right_seam + 1) % len(polygons[right_piece])])
            right_poses = candidate_poses[:]
            right_polygons = [poly[:] for poly in candidate_polygons]
            right_poses[right_piece] = right_pose
            right_polygons[right_piece] = transform_polygon(
                polygons[right_piece], right_pose)
            evaluate_edge_snaps(
                ordering, chain_index + 1,
                right_poses, right_polygons, snapped_seams + 1)

        for long_piece in range(piece_count):
            long_polygon = polygons[long_piece]
            other_pieces = [
                index for index in range(piece_count)
                if index != long_piece]
            for long_edge in range(len(long_polygon)):
                long_a = long_polygon[long_edge]
                long_b = long_polygon[
                    (long_edge + 1) % len(long_polygon)]
                long_length = _distance(long_a, long_b)
                tolerance = max(
                    opts["compound_edge_abs_tolerance_cm"],
                    opts["compound_edge_rel_tolerance"] * long_length)

                selections = []

                def choose_edge(position, selected):
                    if position >= len(other_pieces):
                        selections.append(selected[:])
                        return
                    piece_index = other_pieces[position]
                    polygon = polygons[piece_index]
                    for edge_index in range(len(polygon)):
                        a = polygon[edge_index]
                        b = polygon[(edge_index + 1) % len(polygon)]
                        length = _distance(a, b)
                        if length < long_length:
                            selected.append(
                                (piece_index, edge_index, length))
                            choose_edge(position + 1, selected)
                            selected.pop()

                choose_edge(0, [])
                for selected in selections:
                    selected_sum = sum(item[2] for item in selected)
                    if abs(long_length - selected_sum) > tolerance:
                        continue
                    for ordering in _small_permutations(selected):
                        candidate_poses = [None] * piece_count
                        candidate_polygons = [None] * piece_count
                        candidate_poses[long_piece] = (0.0, 0.0, 0.0)
                        candidate_polygons[long_piece] = long_polygon
                        offset = (long_length - selected_sum) * 0.5
                        unit_x = (long_b[0] - long_a[0]) / long_length
                        unit_y = (long_b[1] - long_a[1]) / long_length
                        valid = True
                        for (moving_piece, moving_edge,
                             moving_length) in ordering:
                            segment_a = (
                                long_a[0] + unit_x * offset,
                                long_a[1] + unit_y * offset)
                            segment_b = (
                                long_a[0] +
                                unit_x * (offset + moving_length),
                                long_a[1] +
                                unit_y * (offset + moving_length))
                            moving_polygon = polygons[moving_piece]
                            moving_a = moving_polygon[moving_edge]
                            moving_b = moving_polygon[
                                (moving_edge + 1) % len(moving_polygon)]
                            pose = _pose_matching_edges(
                                segment_a, segment_b,
                                moving_a, moving_b)
                            transformed = transform_polygon(
                                moving_polygon, pose)
                            for existing in candidate_polygons:
                                if existing is None:
                                    continue
                                overlap_area = convex_overlap_area(
                                    transformed, existing)
                                if overlap_area > opts[
                                        "max_camera_pair_overlap_cm2"]:
                                    valid = False
                                    break
                            if not valid:
                                break
                            candidate_poses[moving_piece] = pose
                            candidate_polygons[moving_piece] = transformed
                            offset += moving_length
                        if not valid:
                            continue
                        evaluate_edge_snaps(
                            ordering, 0, candidate_poses,
                            candidate_polygons, 0)
                        if (best[0] is not None and
                                best[0]["score"] <=
                                opts["early_stop_score"]):
                            return

    def search(placed_count):
        if (best[0] is not None and
                best[0]["score"] <= opts["early_stop_score"]):
            return
        if placed_count == piece_count:
            evaluate()
            return

        state_values = []
        for index in range(piece_count):
            pose = poses[index]
            if pose is not None:
                state_values.extend((
                    index,
                    int(round(_angle_degrees(pose[0]) / 2.0)),
                    int(round(pose[1] * 5.0)),
                    int(round(pose[2] * 5.0)),
                ))
        for piece_index, edge_index in sorted(used_edges):
            state_values.extend((piece_index, edge_index))
        state_values.append(int(round(sum(texture_costs) * 100.0)))
        state_key = tuple(state_values)
        if state_key in seen_by_depth[placed_count]:
            return
        seen_by_depth[placed_count].add(state_key)

        branches = []
        for fixed_index in range(piece_count):
            fixed_poly = placed_polygons[fixed_index]
            if fixed_poly is None:
                continue
            for moving_index in range(piece_count):
                if poses[moving_index] is not None:
                    continue
                moving_source = polygons[moving_index]
                for fixed_edge in range(len(fixed_poly)):
                    if (fixed_index, fixed_edge) in used_edges:
                        continue
                    fixed_a = fixed_poly[fixed_edge]
                    fixed_b = fixed_poly[(fixed_edge + 1) % len(fixed_poly)]
                    fixed_length = _distance(fixed_a, fixed_b)
                    for moving_edge in range(len(moving_source)):
                        if (moving_index, moving_edge) in used_edges:
                            continue
                        moving_a = moving_source[moving_edge]
                        moving_b = moving_source[
                            (moving_edge + 1) % len(moving_source)]
                        moving_length = _distance(moving_a, moving_b)
                        tolerance = max(
                            opts["edge_abs_tolerance_cm"],
                            opts["edge_rel_tolerance"] *
                            max(fixed_length, moving_length))
                        edge_pair = _canonical_edge_pair(
                            (fixed_index, fixed_edge),
                            (moving_index, moving_edge))
                        if (abs(fixed_length - moving_length) > tolerance and
                                edge_pair not in compound_edge_pairs):
                            continue
                        partial_offsets = None
                        if fixed_length > moving_length:
                            partial_offsets = compound_edge_offsets.get((
                                (fixed_index, fixed_edge),
                                (moving_index, moving_edge)))
                        elif moving_length > fixed_length:
                            partial_offsets = compound_edge_offsets.get((
                                (moving_index, moving_edge),
                                (fixed_index, fixed_edge)))
                        alignments = _edge_alignment_candidates(
                            fixed_a, fixed_b, moving_a, moving_b,
                            tolerance, opts, partial_offsets)
                        for (pose, fixed_fully_used,
                             moving_fully_used, is_partial) in alignments:
                            descriptor_cost = 0.0
                            descriptors = opts["edge_descriptors"]
                            if descriptors is not None and not is_partial:
                                descriptor_cost = _descriptor_distance(
                                    descriptors[fixed_index][fixed_edge],
                                    descriptors[moving_index][moving_edge])
                                if descriptor_cost > opts[
                                        "max_texture_distance"]:
                                    continue
                            candidate = transform_polygon(
                                moving_source, pose)

                            overlaps = False
                            for other_index in range(piece_count):
                                other = placed_polygons[other_index]
                                if other is not None and polygons_overlap(
                                        candidate, other,
                                        opts["overlap_epsilon_cm"]):
                                    overlaps = True
                                    break
                            if overlaps:
                                continue

                            all_points = []
                            for other in placed_polygons:
                                if other is not None:
                                    all_points.extend(other)
                            all_points.extend(candidate)
                            x_values = [
                                point[0] for point in all_points]
                            y_values = [
                                point[1] for point in all_points]
                            # Any valid 12 x 9 cm rectangle has a 15 cm diagonal.
                            if (max(x_values) - min(x_values) > 16.0 or
                                    max(y_values) - min(y_values) > 16.0):
                                continue

                            partial_assembly = [
                                poly for poly in placed_polygons
                                if poly is not None]
                            partial_assembly.append(candidate)
                            partial_rectangle = _minimum_edge_rectangle(
                                partial_assembly)
                            partial_area = sum(
                                polygon_area(poly)
                                for poly in partial_assembly)
                            compactness = 1.0
                            if (partial_rectangle is not None and
                                    partial_rectangle[0] > 0.001):
                                compactness = max(
                                    0.0,
                                    (partial_rectangle[0] - partial_area) /
                                    partial_rectangle[0])
                            # Full edge contacts and compact partial assemblies
                            # are much more likely than long, sparse chains.
                            priority = (
                                compactness +
                                (0.025 if is_partial else 0.0) +
                                descriptor_cost * 0.03)
                            branches.append((
                                priority,
                                moving_index, pose, candidate,
                                fixed_index, fixed_edge, moving_edge,
                                fixed_fully_used, moving_fully_used,
                                descriptor_cost,
                            ))

        branches.sort(key=lambda item: item[0])
        beam_width = int(opts["search_beam_width"])
        if beam_width > 0 and len(branches) > beam_width:
            branches = branches[:beam_width]

        for branch in branches:
            (_, moving_index, pose, candidate,
             fixed_index, fixed_edge, moving_edge,
             fixed_fully_used, moving_fully_used,
             descriptor_cost) = branch
            added_edges = []
            if fixed_fully_used:
                key = (fixed_index, fixed_edge)
                used_edges.add(key)
                added_edges.append(key)
            if moving_fully_used:
                key = (moving_index, moving_edge)
                used_edges.add(key)
                added_edges.append(key)
            poses[moving_index] = pose
            placed_polygons[moving_index] = candidate
            texture_costs.append(descriptor_cost)
            search(placed_count + 1)
            texture_costs.pop()
            for key in added_edges:
                used_edges.remove(key)
            poses[moving_index] = None
            placed_polygons[moving_index] = None
            if (best[0] is not None and
                    best[0]["score"] <= opts["early_stop_score"]):
                return

    # This fast path deliberately joins several short edges to one long edge,
    # so it belongs to the partial/T-junction fallback only.
    if opts["allow_partial_edge_matches"]:
        try_main_edge_split()
        if best[0] is None and opts["partial_fast_path_only"]:
            return None
    if best[0] is None:
        for index in range(piece_count):
            poses[index] = None
            placed_polygons[index] = None
        poses[0] = (0.0, 0.0, 0.0)
        placed_polygons[0] = polygons[0]
        search(1)
    if best[0] is None:
        return None

    solution = best[0]
    global_angle = solution["global_angle"]
    aligned = [
        transform_polygon(poly, (global_angle, 0.0, 0.0))
        for poly in solution["assembly"]
    ]
    bounds = solution["bounds"]
    current_center = (
        (bounds[0] + bounds[2]) * 0.5,
        (bounds[1] + bounds[3]) * 0.5,
    )
    target_center = opts["target_center_cm"]
    shift_x = target_center[0] - current_center[0]
    shift_y = target_center[1] - current_center[1]

    placements = []
    target_polygons = []
    for index in range(piece_count):
        target_poly = [
            (point[0] + shift_x, point[1] + shift_y)
            for point in aligned[index]
        ]
        target_polygons.append(target_poly)
        source_center = polygon_centroid(polygons[index])
        target_piece_center = polygon_centroid(target_poly)
        total_angle = solution["poses"][index][0] + global_angle
        placements.append({
            "piece_id": index,
            "source_center_cm": source_center,
            "target_center_cm": target_piece_center,
            "rotation_deg": _angle_degrees(total_angle),
            "source_polygon_cm": polygons[index],
            "target_polygon_cm": target_poly,
        })

    return {
        "score": solution["score"],
        "gap_ratio": solution["gap_ratio"],
        "overlap_area_cm2": solution["overlap_area_cm2"],
        "snapped_seams": solution["snapped_seams"],
        "texture_cost": solution["texture_cost"],
        "outer_piece_count": solution["outer_piece_count"],
        "target_width_cm": solution["width_cm"],
        "target_height_cm": solution["height_cm"],
        "target_center_cm": target_center,
        "placements": placements,
        "target_polygons_cm": target_polygons,
    }


def build_ascii_packet(command, *values):
    """Build an NMEA-like UART line with an XOR checksum."""
    fields = ["PZ", str(command)]
    fields.extend(str(value) for value in values)
    body = ",".join(fields)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return "$%s*%02X\r\n" % (body, checksum)


def placement_packets(solution):
    packets = [build_ascii_packet(
        "BEGIN", len(solution["placements"]))]
    for item in solution["placements"]:
        source = item["source_center_cm"]
        target = item["target_center_cm"]
        packets.append(build_ascii_packet(
            "MOVE",
            item["piece_id"],
            int(round(source[0] * 10.0)),
            int(round(source[1] * 10.0)),
            int(round(target[0] * 10.0)),
            int(round(target[1] * 10.0)),
            int(round(item["rotation_deg"] * 10.0)),
        ))
    packets.append(build_ascii_packet("END"))
    return packets
