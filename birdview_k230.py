"""K230 俯视图（IPM）映射示例。

将摄像头画面中的地面区域映射到按厘米标尺排列的俯视图。
所有会因车体、摄像头位置或视野变化而改变的参数均在文件顶部。
"""

import gc
import os
import time
import math

import cv2
import image
import ulab.numpy as np

from machine import UART, FPIOA

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager


# ================================ 用户参数 =================================
# ----------------------------- 车体与地面尺寸 ------------------------------
CAR_WIDTH_CM = 15
CAR_LENGTH_CM = 25

# 摄像头可见地面区域：以车头中心（也是摄像头中心）为距离参考。
GROUND_NEAR_DISTANCE_CM = 11.5        # 可见区域近端距车头。
GROUND_FAR_DISTANCE_CM = 60.5         # 可见区域远端距车头。
GROUND_NEAR_WIDTH_CM = 42             # 近端横向宽度。
GROUND_FAR_WIDTH_CM = 70              # 远端横向宽度。

# ------------------------------- 摄像头采集 -------------------------------
IMAGE_W = 160                          # 宽度建议为 16 的倍数。
IMAGE_H = 120
SENSOR_FPS = 30
CAMERA_HMIRROR = True
CAMERA_VFLIP = True

# 源图像中的四个地面角点，格式为 (x, y)，坐标原点在左上角。
# 顺序必须为：远端左、远端右、近端右、近端左。
# 默认使用整张 160x120 图像。若只映射其中一块地面 ROI，请按 LCD 画面调节。
SRC_FAR_LEFT = (0, 0)
SRC_FAR_RIGHT = (IMAGE_W - 1, 0)
SRC_NEAR_RIGHT = (IMAGE_W - 1, IMAGE_H - 1)
SRC_NEAR_LEFT = (0, IMAGE_H - 1)

# ------------------------------ 俯视图输出尺度 -----------------------------
# 输出平面覆盖的真实地面范围。前方范围必须包含远端距离；
# 向后范围用于在俯视图中完整显示小车模型。
BIRD_WORLD_WIDTH_CM = 80
BIRD_FORWARD_DISTANCE_CM = 60.5
BIRD_REAR_DISTANCE_CM = 25
BIRD_PIXELS_PER_CM = 2

# 在俯视图上绘制小车矩形。小车前沿（摄像头所在位置）距车头为 0 cm。
SHOW_CAR_MODEL = True
CAR_MODEL_GRAY = 255                  # 0 为黑，255 为白。
CAR_MODEL_FRONT_GRAY = 255
CAR_MODEL_THICKNESS = 1

# 实际轨迹圆弧：以“后轴中心”为运动参考点。
# 摄像头位于车头基准；车长 25 cm，后轴中心距车尾 6 cm，故为 19 cm。
REAR_AXLE_BEHIND_CAMERA_CM = CAR_LENGTH_CM - 6.0
WHEELBASE_CM = 14.3
# 绘制左 27° 到右 27°的候选圆弧，每 2° 一条。注意：右 26°、右 27°
# 超出当前实测范围，是按阿克曼几何模型外推的参考轨迹。
SHOW_ALL_TRAJECTORIES = True
TRAJECTORY_MIN_ANGLE_DEG = -27.0
TRAJECTORY_MAX_ANGLE_DEG = 27.0
TRAJECTORY_ANGLE_STEP_DEG = 2.0
# 候选轨迹只画短线，避免遮住道路；与当前道路中心线最接近的那条画成长线。
TRAJECTORY_SHORT_FORWARD_CM = 18
TRAJECTORY_SELECTED_FORWARD_CM = 60
TRAJECTORY_MAX_TURN_DEG = 85         # 小半径时避免画过半圆。
TRAJECTORY_GRAY = 120
TRAJECTORY_THICKNESS = 1
TRAJECTORY_SELECTED_GRAY = 255
TRAJECTORY_SELECTED_THICKNESS = 2

# 实测“真实 PWM 脉宽 -> 实际曲率”数据。
# 曲率单位 1/cm，右转为正、左转为负；按 PWM 从小到大排序。
STEERING_PWM_CURVATURE = (
    (1222,  0.033333), (1300,  0.026110), (1400,  0.017857),
    (1477,  0.010190), (1544,  0.003958), (1577,  0.002627),
    (1644, -0.003958), (1655, -0.004387), (1755, -0.007234),
    (1833, -0.016807), (1933, -0.023981), (2011, -0.029762),
    (2111, -0.036496),
)

# True：仅显示原始画面和四边形，方便标定 SRC_*；False：显示俯视图。
CALIBRATION_MODE = False
SOURCE_GUIDE_COLOR = (255, 0, 0)
SOURCE_GUIDE_THICKNESS = 1

# -------------------------------- LCD 显示 --------------------------------
DISPLAY_TYPE = Display.ST7701
DISPLAY_W = 800
DISPLAY_H = 480
DISPLAY_FPS = 30
DISPLAY_TO_IDE = True
# True 时按统一比例缩放并在 LCD 左右留黑边，保证车体和道路几何比例正确。
DISPLAY_KEEP_ASPECT = True

LOG_INTERVAL_FRAMES = 30              # 0 表示关闭日志。
GC_INTERVAL_FRAMES = 30               # 0 表示关闭主动 GC。

# ----------------------------- 俯视图巡线处理 -----------------------------
# 下面参数与 eight_neighbor_k230.py 的巡线处理含义一致。
LANE_WHITE_TRACK = True               # True：亮赛道、暗边界；False：黑线白底。
LANE_BIN_THRESHOLD = 60        # 固定二值化阈值，范围 0～254。
LANE_FILTER_ENABLE = False            # 逐像素滤波会降低帧率。
LANE_FILTER_FILL_BLACK_AT_WHITE_NEIGHBORS = 5
LANE_FILTER_REMOVE_WHITE_AT_WHITE_NEIGHBORS = 2
LANE_FRAME_BORDER_WIDTH = 2
LANE_SIDE_SEARCH_MARGIN = 2
# 俯视图中从车头前方多少厘米处开始寻找左右边界；
# 默认就是摄像头可见地面的近端位置。
LANE_START_DISTANCE_CM = GROUND_NEAR_DISTANCE_CM
LANE_EDGE_MEET_DISTANCE = 2
LANE_MAX_STEPS_MULTIPLIER = 3
LANE_SHOW_CENTER_LINE = True
LANE_CENTER_LINE_GRAY = 127

# --------------------------- MSPM0 舵机串口控制 ---------------------------
# UART2/IO5/IO6 已在本工程内检查，无其他设备占用。
# 若后续将这三个资源分配给别的设备，必须先改为 True，禁止重复初始化。
UART2_IO5_IO6_RESERVED_BY_OTHER_DEVICE = False
SERVO_UART_TX_PIN = 5                 # K230 IO5 -> MSPM0 PB16 (RX)
SERVO_UART_RX_PIN = 6                 # K230 IO6 <- MSPM0 PB15 (TX，可选)
SERVO_UART_BAUDRATE = 115200
SERVO_SEND_INTERVAL_FRAMES = 1         # 每处理 1 帧发送一次最新舵机角度。
SERVO_UART_TX_LOG = True              # 每次发送后打印十进制度数和原始 HEX。
SERVO_ANGLE_MIN = 63
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_MAX = 117

# 使用俯视图中心线进行前瞻控制；距离以车头/摄像头为参考。
CONTROL_LOOKAHEAD_DISTANCE_CM = 35
# ============================================================================


# 输出像素尺寸由真实尺寸与比例自动计算；通常无需改动。
BIRD_WORLD_DEPTH_CM = BIRD_FORWARD_DISTANCE_CM + BIRD_REAR_DISTANCE_CM
BIRD_W = int(BIRD_WORLD_WIDTH_CM * BIRD_PIXELS_PER_CM)
BIRD_H = int(BIRD_WORLD_DEPTH_CM * BIRD_PIXELS_PER_CM)
LANE_BORDER_MIN = LANE_SIDE_SEARCH_MARGIN
LANE_BORDER_MAX = BIRD_W - LANE_SIDE_SEARCH_MARGIN - 1
LANE_MAX_STEPS = BIRD_H * LANE_MAX_STEPS_MULTIPLIER


def clamp_servo_angle(angle):
    """将舵机角度安全转换为整数，并限制在 MSPM0 协议范围内。"""
    try:
        value = float(angle)
    except (TypeError, ValueError):
        return SERVO_ANGLE_CENTER
    # NaN 不等于自身；无穷大或异常大数也不允许直接发送。
    if value != value or value > 1000000 or value < -1000000:
        return SERVO_ANGLE_CENTER
    value = int(round(value))
    if value < SERVO_ANGLE_MIN:
        value = SERVO_ANGLE_MIN
    elif value > SERVO_ANGLE_MAX:
        value = SERVO_ANGLE_MAX
    return value


def build_servo_command(angle):
    """构造一个原始字节的 MSPM0 舵机角度命令。"""
    value = clamp_servo_angle(angle)
    return bytes((value,))


def send_servo_angle(uart, angle):
    """发送一个原始角度字节；返回实际发送的角度数值。"""
    value = clamp_servo_angle(angle)
    command = build_servo_command(value)
    written = uart.write(command)
    if SERVO_UART_TX_LOG:
        print("UART2 TX angle=%d raw=[%02X] written=%s" % (
            value, value, str(written)))
    return value


def init_servo_uart():
    """映射 IO5/IO6 并初始化 K230 UART2；冲突时拒绝重复使用。"""
    if UART2_IO5_IO6_RESERVED_BY_OTHER_DEVICE:
        raise RuntimeError("UART2 / IO5 / IO6 已被其他设备保留，禁止重复初始化")
    fpioa = FPIOA()
    fpioa.set_function(SERVO_UART_TX_PIN, fpioa.UART2_TXD)
    fpioa.set_function(SERVO_UART_RX_PIN, fpioa.UART2_RXD)
    return UART(UART.UART2, baudrate=SERVO_UART_BAUDRATE,
                bits=UART.EIGHTBITS, parity=UART.PARITY_NONE,
                stop=UART.STOPBITS_ONE)


def solve_linear_system(matrix, vector):
    """用高斯消元求解小型线性方程组，避免 getPerspectiveTransform 的栈开销。"""
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot = column
        for row in range(column + 1, size):
            if abs(augmented[row][column]) > abs(augmented[pivot][column]):
                pivot = row
        if abs(augmented[pivot][column]) < 0.000001:
            raise ValueError("透视变换的四个点无效或共线")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]

    return [augmented[row][size] for row in range(size)]


def make_destination_points():
    """根据厘米参数计算四个点在俯视图中的目标像素坐标。"""
    def point(distance_cm, width_cm, is_right):
        x_cm = (BIRD_WORLD_WIDTH_CM + width_cm) / 2 if is_right else \
               (BIRD_WORLD_WIDTH_CM - width_cm) / 2
        # y=0 是最远处，y=BIRD_H 是小车后方；因此车头在下方而道路在上方。
        y_cm = BIRD_FORWARD_DISTANCE_CM - distance_cm
        return (x_cm * BIRD_PIXELS_PER_CM, y_cm * BIRD_PIXELS_PER_CM)

    return (
        point(GROUND_FAR_DISTANCE_CM, GROUND_FAR_WIDTH_CM, False),
        point(GROUND_FAR_DISTANCE_CM, GROUND_FAR_WIDTH_CM, True),
        point(GROUND_NEAR_DISTANCE_CM, GROUND_NEAR_WIDTH_CM, True),
        point(GROUND_NEAR_DISTANCE_CM, GROUND_NEAR_WIDTH_CM, False),
    )


def make_homography(source_points, destination_points):
    """计算 3×3 单应矩阵：源图像像素坐标 -> 俯视图像素坐标。"""
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(source_points, destination_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.append(v)

    values = solve_linear_system(matrix, vector)
    return np.array([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], 1.0],
    ], dtype=np.float)


def lane_make_binary(img):
    """使用固定阈值将俯视灰度图二值化。"""
    if LANE_WHITE_TRACK:
        img.binary([(LANE_BIN_THRESHOLD + 1, 255)])
    else:
        img.binary([(0, LANE_BIN_THRESHOLD)])


def lane_filter(img):
    """八邻域去噪；与原巡线程序相同，开启时会降低帧率。"""
    for y in range(1, BIRD_H - 1):
        for x in range(1, BIRD_W - 1):
            total = (
                img.get_pixel(x - 1, y - 1) + img.get_pixel(x, y - 1) +
                img.get_pixel(x + 1, y - 1) + img.get_pixel(x - 1, y) +
                img.get_pixel(x + 1, y) + img.get_pixel(x - 1, y + 1) +
                img.get_pixel(x, y + 1) + img.get_pixel(x + 1, y + 1)
            )
            pixel = img.get_pixel(x, y)
            if (total >= 255 * LANE_FILTER_FILL_BLACK_AT_WHITE_NEIGHBORS and
                    pixel == 0):
                img.set_pixel(x, y, 255)
            elif (total <= 255 * LANE_FILTER_REMOVE_WHITE_AT_WHITE_NEIGHBORS and
                    pixel == 255):
                img.set_pixel(x, y, 0)


def lane_draw_black_frame(img):
    """添加黑色边框，防止八邻域搜索越过图像边界。"""
    for y in range(BIRD_H):
        for offset in range(LANE_FRAME_BORDER_WIDTH):
            img.set_pixel(offset, y, 0)
            img.set_pixel(BIRD_W - 1 - offset, y, 0)
    for x in range(BIRD_W):
        for offset in range(LANE_FRAME_BORDER_WIDTH):
            img.set_pixel(x, offset, 0)


def lane_start_points(img, y):
    """在底部搜索离画面中心最近的左右边界起点。"""
    middle = BIRD_W // 2
    left = None
    right = None
    for x in range(middle, LANE_BORDER_MIN, -1):
        if img.get_pixel(x, y) == 255 and img.get_pixel(x - 1, y) == 0:
            left = (x, y)
            break
    for x in range(middle, LANE_BORDER_MAX):
        if img.get_pixel(x, y) == 255 and img.get_pixel(x + 1, y) == 0:
            right = (x, y)
            break
    return left, right


LANE_LEFT_SEEDS = ((0, 1), (-1, 1), (-1, 0), (-1, -1),
                   (0, -1), (1, -1), (1, 0), (1, 1))
LANE_RIGHT_SEEDS = ((0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1), (-1, -1), (-1, 0), (-1, 1))


def lane_next_edge_point(img, point, seeds):
    """在当前点周围按八邻域寻找下一处黑白边缘点。"""
    x, y = point
    candidates = []
    for index in range(8):
        x1 = x + seeds[index][0]
        y1 = y + seeds[index][1]
        x2 = x + seeds[(index + 1) & 7][0]
        y2 = y + seeds[(index + 1) & 7][1]
        if (x1 < 0 or x1 >= BIRD_W or y1 < 0 or y1 >= BIRD_H or
                x2 < 0 or x2 >= BIRD_W or y2 < 0 or y2 >= BIRD_H):
            continue
        if img.get_pixel(x1, y1) == 0 and img.get_pixel(x2, y2) == 255:
            candidates.append((x1, y1, index))
    if not candidates:
        return point, 0
    candidate = min(candidates, key=lambda item: item[1])
    return (candidate[0], candidate[1]), candidate[2]


def lane_trace_edges(img, left_start, right_start):
    """从两个底部起点向上追踪左右赛道边界。"""
    left_points = []
    right_points = []
    left = left_start
    right = right_start
    highest = BIRD_H - 1

    for _ in range(LANE_MAX_STEPS):
        left_points.append(left)
        new_left, left_direction = lane_next_edge_point(img, left, LANE_LEFT_SEEDS)
        if len(right_points) >= 2 and len(left_points) >= 3:
            if (right_points[-1] == right_points[-2] == right or
                    left_points[-1] == left_points[-2] == left_points[-3]):
                break
        if (abs(right[0] - left[0]) < LANE_EDGE_MEET_DISTANCE and
                abs(right[1] - left[1]) < LANE_EDGE_MEET_DISTANCE):
            highest = (right[1] + left[1]) // 2
            break
        if right[1] < left[1]:
            left = new_left
            continue
        if left_direction == 7 and right[1] > left[1]:
            left_points.pop()
        else:
            left = new_left
        right_points.append(right)
        right, _ = lane_next_edge_point(img, right, LANE_RIGHT_SEEDS)

    return left_points, right_points, highest


def lane_process(img):
    """执行与原程序相同的二值化、边界追踪和中心线计算。"""
    lane_make_binary(img)
    if LANE_FILTER_ENABLE:
        lane_filter(img)
    lane_draw_black_frame(img)

    start_y = lane_start_y()
    left_start, right_start = lane_start_points(img, start_y)
    if left_start is None or right_start is None:
        return None

    left_points, right_points, top_y = lane_trace_edges(img, left_start, right_start)
    left_border = [LANE_BORDER_MIN] * BIRD_H
    right_border = [LANE_BORDER_MAX] * BIRD_H
    for x, y in left_points:
        if 0 <= y < BIRD_H:
            left_border[y] = x + 1
    for x, y in right_points:
        if 0 <= y < BIRD_H:
            right_border[y] = x - 1

    centre_line = [0] * BIRD_H
    for y in range(top_y, BIRD_H - 1):
        centre_line[y] = (left_border[y] + right_border[y]) // 2
    return top_y, centre_line


def lane_start_y():
    """将车头前方距离换算成俯视图的起始搜索行。"""
    y = int((BIRD_FORWARD_DISTANCE_CM - LANE_START_DISTANCE_CM) *
            BIRD_PIXELS_PER_CM)
    return max(1, min(BIRD_H - 2, y))


def steering_angle_from_centre_line(centre_line):
    """用 Pure Pursuit 根据俯视图中心线计算等效前轮转向角。

    正值为右转、负值为左转；车道中心线无效时返回 None。
    """
    target_y = int((BIRD_FORWARD_DISTANCE_CM - CONTROL_LOOKAHEAD_DISTANCE_CM) *
                   BIRD_PIXELS_PER_CM)
    if target_y < 0 or target_y >= BIRD_H:
        return None
    target_x = centre_line[target_y]
    if target_x == 0:
        return None

    lateral_cm = (target_x - BIRD_W // 2) / BIRD_PIXELS_PER_CM
    forward_cm = CONTROL_LOOKAHEAD_DISTANCE_CM + REAR_AXLE_BEHIND_CAMERA_CM
    distance_squared = lateral_cm * lateral_cm + forward_cm * forward_cm
    if distance_squared <= 0:
        return None
    curvature = 2.0 * lateral_cm / distance_squared
    return math.degrees(math.atan(WHEELBASE_CM * curvature))


def draw_car_model(img):
    """按真实车宽、车长在俯视灰度图下方绘制小车矩形和前沿标记。"""
    car_width_px = int(CAR_WIDTH_CM * BIRD_PIXELS_PER_CM)
    car_length_px = int(CAR_LENGTH_CM * BIRD_PIXELS_PER_CM)
    car_x = (BIRD_W - car_width_px) // 2
    car_y = int(BIRD_FORWARD_DISTANCE_CM * BIRD_PIXELS_PER_CM)
    img.draw_rectangle(car_x, car_y, car_width_px, car_length_px,
                       color=CAR_MODEL_GRAY, thickness=CAR_MODEL_THICKNESS)
    img.draw_line(car_x, car_y, car_x + car_width_px, car_y,
                  color=CAR_MODEL_FRONT_GRAY, thickness=CAR_MODEL_THICKNESS)


def curvature_from_pwm(pwm_us):
    """由实测表反查当前 PWM 对应的实际曲率；中间值做线性插值。"""
    if pwm_us <= STEERING_PWM_CURVATURE[0][0]:
        return STEERING_PWM_CURVATURE[0][1]
    if pwm_us >= STEERING_PWM_CURVATURE[-1][0]:
        return STEERING_PWM_CURVATURE[-1][1]
    for index in range(len(STEERING_PWM_CURVATURE) - 1):
        pwm0, curvature0 = STEERING_PWM_CURVATURE[index]
        pwm1, curvature1 = STEERING_PWM_CURVATURE[index + 1]
        if pwm0 <= pwm_us <= pwm1:
            ratio = (pwm_us - pwm0) / (pwm1 - pwm0)
            return curvature0 + ratio * (curvature1 - curvature0)
    return 0.0


def draw_predicted_trajectory(img, curvature, forward_cm, color, thickness):
    """根据实测 PWM 对应曲率，绘制后轴中心实际会走出的预测圆弧。"""
    rear_axle_y = int((BIRD_FORWARD_DISTANCE_CM + REAR_AXLE_BEHIND_CAMERA_CM) *
                      BIRD_PIXELS_PER_CM)
    rear_axle_x = BIRD_W // 2

    if abs(curvature) < 0.000001:
        end_y = rear_axle_y - int(forward_cm * BIRD_PIXELS_PER_CM)
        img.draw_line(rear_axle_x, rear_axle_y, rear_axle_x, end_y,
                      color=color, thickness=thickness)
        return

    radius_cm = 1.0 / curvature       # 右转为正，左转为负。
    max_arc_cm = min(forward_cm,
                     abs(radius_cm) * math.radians(TRAJECTORY_MAX_TURN_DEG))
    previous_x = rear_axle_x
    previous_y = rear_axle_y
    for distance_cm in range(1, int(max_arc_cm) + 1):
        theta = distance_cm / radius_cm
        lateral_cm = radius_cm * (1.0 - math.cos(theta))
        forward_cm = radius_cm * math.sin(theta)
        x = rear_axle_x + int(lateral_cm * BIRD_PIXELS_PER_CM)
        y = rear_axle_y - int(forward_cm * BIRD_PIXELS_PER_CM)
        if (0 <= previous_x < BIRD_W and 0 <= previous_y < BIRD_H and
                0 <= x < BIRD_W and 0 <= y < BIRD_H):
            img.draw_line(previous_x, previous_y, x, y,
                          color=color, thickness=thickness)
        previous_x = x
        previous_y = y


def draw_all_trajectories(img, target_angle_deg):
    """候选圆弧均为短线；离当前控制角度最近的候选线为长白线。"""
    count = int((TRAJECTORY_MAX_ANGLE_DEG - TRAJECTORY_MIN_ANGLE_DEG) /
                TRAJECTORY_ANGLE_STEP_DEG)
    selected_index = None
    if target_angle_deg is not None:
        selected_index = min(
            range(count + 1),
            key=lambda i: abs((TRAJECTORY_MIN_ANGLE_DEG +
                               i * TRAJECTORY_ANGLE_STEP_DEG) - target_angle_deg))
    for index in range(count + 1):
        angle_deg = TRAJECTORY_MIN_ANGLE_DEG + index * TRAJECTORY_ANGLE_STEP_DEG
        curvature = math.tan(math.radians(angle_deg)) / WHEELBASE_CM
        if index == selected_index:
            draw_predicted_trajectory(
                img, curvature, TRAJECTORY_SELECTED_FORWARD_CM,
                TRAJECTORY_SELECTED_GRAY, TRAJECTORY_SELECTED_THICKNESS)
        else:
            draw_predicted_trajectory(
                img, curvature, TRAJECTORY_SHORT_FORWARD_CM,
                TRAJECTORY_GRAY, TRAJECTORY_THICKNESS)


def draw_source_guides(img, source_points):
    """在原图上画出用于标定的四边形。"""
    for index in range(4):
        x1, y1 = source_points[index]
        x2, y2 = source_points[(index + 1) % 4]
        img.draw_line(x1, y1, x2, y2,
                      color=SOURCE_GUIDE_COLOR, thickness=SOURCE_GUIDE_THICKNESS)


def main():
    source_points = (SRC_FAR_LEFT, SRC_FAR_RIGHT, SRC_NEAR_RIGHT, SRC_NEAR_LEFT)
    destination_points = make_destination_points()
    homography = make_homography(source_points, destination_points)
    sensor = Sensor(width=IMAGE_W, height=IMAGE_H, fps=SENSOR_FPS)
    servo_uart = None

    try:
        sensor.reset()
        sensor.set_hmirror(CAMERA_HMIRROR)
        sensor.set_vflip(CAMERA_VFLIP)
        # OpenCV 透视变换使用 RGB888 数组，配置与提供的 OpenCV 例程一致。
        sensor.set_framesize(width=IMAGE_W, height=IMAGE_H)
        sensor.set_pixformat(Sensor.RGB888)

        Display.init(DISPLAY_TYPE, width=DISPLAY_W, height=DISPLAY_H,
                     fps=DISPLAY_FPS, to_ide=DISPLAY_TO_IDE)
        MediaManager.init()
        sensor.run()

        # 必须放在全部媒体初始化之后，避免摄像头或显示初始化覆盖 FPIOA。
        servo_uart = init_servo_uart()
        uart_fpioa = FPIOA()
        print("UART2 pin map: TX=IO%d RX=IO%d baud=%d" % (
            uart_fpioa.get_pin_num(uart_fpioa.UART2_TXD),
            uart_fpioa.get_pin_num(uart_fpioa.UART2_RXD),
            SERVO_UART_BAUDRATE))
        last_sent_servo_angle = send_servo_angle(
            servo_uart, SERVO_ANGLE_CENTER)  # 启动即回正。
        servo_send_frame_count = 0

        fps = time.clock()
        frame_count = 0
        while True:
            os.exitpoint()
            fps.tick()
            camera_img = sensor.snapshot()
            steering_angle = None

            if CALIBRATION_MODE:
                draw_source_guides(camera_img, source_points)
                output_img = camera_img
                status = "标定模式"
            else:
                camera_np = camera_img.to_numpy_ref()
                bird_np = cv2.warpPerspective(camera_np, homography, (BIRD_W, BIRD_H))
                bird_rgb = image.Image(BIRD_W, BIRD_H, image.RGB888,
                                       alloc=image.ALLOC_REF, data=bird_np)
                output_img = bird_rgb.to_grayscale()
                del bird_rgb

                result = lane_process(output_img)
                if result is None:
                    status = "俯视图未找到赛道"
                else:
                    top_y, centre_line = result
                    start_y = lane_start_y()
                    error = centre_line[start_y] - BIRD_W // 2
                    steering_angle = steering_angle_from_centre_line(centre_line)
                    if LANE_SHOW_CENTER_LINE:
                        for y in range(top_y, BIRD_H - 1):
                            if centre_line[y]:
                                output_img.set_pixel(centre_line[y], y,
                                                    LANE_CENTER_LINE_GRAY)
                    status = "bird lane center=%d error=%d" % (
                        centre_line[start_y], error)
                if SHOW_ALL_TRAJECTORIES:
                    draw_all_trajectories(output_img, steering_angle)
                if SHOW_CAR_MODEL:
                    draw_car_model(output_img)

            # 角度一旦变化就在当前图像循环立即发送。
            # 没有找到赛道时始终保持上一次有效角度，不主动发送 90 回正。
            if steering_angle is not None:
                desired_servo_angle = clamp_servo_angle(
                    SERVO_ANGLE_CENTER + steering_angle)
            else:
                desired_servo_angle = last_sent_servo_angle

            # 严格按图像处理帧计数，每帧发送一次最新角度。
            servo_send_frame_count += 1
            if servo_send_frame_count >= SERVO_SEND_INTERVAL_FRAMES:
                last_sent_servo_angle = send_servo_angle(
                    servo_uart, desired_servo_angle)
                servo_send_frame_count = 0

            # 等比例显示才能保持俯视图中的车体和道路尺寸比例。
            x_scale = DISPLAY_W / output_img.width()
            y_scale = DISPLAY_H / output_img.height()
            if DISPLAY_KEEP_ASPECT:
                display_scale = min(x_scale, y_scale)
                display_x = (DISPLAY_W - int(output_img.width() * display_scale)) // 2
                display_y = (DISPLAY_H - int(output_img.height() * display_scale)) // 2
                x_scale = display_scale
                y_scale = display_scale
            else:
                display_x = 0
                display_y = 0
            display_img = output_img.to_rgb565(
                x_scale=x_scale, y_scale=y_scale)
            Display.show_image(display_img, x=display_x, y=display_y)

            if output_img is not camera_img:
                del output_img
            del display_img
            del camera_img
            frame_count += 1
            if (LOG_INTERVAL_FRAMES > 0 and
                    frame_count % LOG_INTERVAL_FRAMES == 0):
                print("%s, servo=%d, fps=%.1f" % (
                    status, last_sent_servo_angle, fps.fps()))
            if (GC_INTERVAL_FRAMES > 0 and
                    frame_count % GC_INTERVAL_FRAMES == 0):
                gc.collect()
    except KeyboardInterrupt:
        print("用户停止")
    finally:
        if servo_uart is not None:
            try:
                send_servo_angle(servo_uart, SERVO_ANGLE_CENTER)
            except Exception:
                pass
        sensor.stop()
        Display.deinit()
        time.sleep_ms(50)
        MediaManager.deinit()


if __name__ == "__main__":
    main()
