"""SSVEP 刺激几何：以物理视觉角度设计，空间不足时整体缩小并记录实际值。"""

import math


def stimulus_geometry(width, height, display_size, *, diagonal_inches=None,
                      distance_cm=70., size_deg=4., pitch_deg=6., rows=1):
    """PsychoPy 客户区坐标；display_size 必须使用同一坐标单位的整屏尺寸。

    pitch 是相邻目标中心在屏幕上的物理距离所对的角度，不是边缘间距。
    4°/6° 是可调设计起点，不是 FBCCA 规定或已经验证的最优参数。
    """
    if rows not in (1, 2):
        raise ValueError("仅支持一行主菜单或两行六分类")
    if not (0 < size_deg < pitch_deg < 90 and distance_cm > 0):
        raise ValueError("视觉角度须满足 0 < size < pitch < 90，距离为正")
    ratio = math.tan(math.radians(pitch_deg / 2)) / math.tan(math.radians(size_deg / 2))
    units_per_cm = None
    if diagonal_inches is not None:
        if diagonal_inches <= 0:
            raise ValueError("屏幕对角线必须为正")
        units_per_cm = math.hypot(*display_size) / (diagonal_inches * 2.54)
        desired_side = 2 * distance_cm * math.tan(math.radians(size_deg / 2)) * units_per_cm
    else:
        desired_side = height * .25
    # 给标题、状态条和选中描边留出空间；方块、中心间距同比缩放。
    max_side = min(width * .88 / (1 + 2 * ratio),
                   height * .68 / (1 + (rows - 1) * ratio))
    side = min(desired_side, max_side)
    pitch = side * ratio
    actual_size = actual_pitch = side_cm = None
    if units_per_cm is not None:
        side_cm = side / units_per_cm
        actual_size = math.degrees(2 * math.atan(side_cm / (2 * distance_cm)))
        actual_pitch = math.degrees(2 * math.atan(pitch / units_per_cm / (2 * distance_cm)))
    return {"side_pix_units": side, "center_pitch_pix_units": pitch,
            "edge_gap_pix_units": pitch - side, "rows": rows, "columns": 3,
            "requested_size_deg": size_deg, "requested_center_pitch_deg": pitch_deg,
            "actual_size_deg": actual_size, "actual_center_pitch_deg": actual_pitch,
            "side_cm": side_cm, "viewing_distance_cm": distance_cm,
            "screen_diagonal_inches": diagonal_inches,
            "physical_scale_source": "declared_screen_geometry" if units_per_cm else "uncalibrated_window_fraction",
            "fit_scale": side / desired_side, "physical_size_independently_measured": False}


def chair_positions(pitch, center_y):
    """位置与刺激数组的顺序解耦；保留原来的 class/frequency 对应关系。"""
    grid = {"LEFT": (-1, .5), "UP": (0, .5), "RIGHT": (1, .5),
            "MENU": (-1, -.5), "DOWN": (0, -.5), "STOP": (1, -.5)}
    return {symbol: (column * pitch, center_y + row * pitch)
            for symbol, (column, row) in grid.items()}
