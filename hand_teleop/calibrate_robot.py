import json
import os
import time

import numpy as np

# ─────────────────────────────────────────────
# LeRobot 임포트 (최신 API 기준)
# ─────────────────────────────────────────────
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors import Motor, MotorNormMode
    LEROBOT_NEW_API = True
except ImportError:
    try:
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        LEROBOT_NEW_API = False
    except ImportError:
        raise ImportError(
            "LeRobot가 설치되지 않았습니다.\n"
            "설치: pip install lerobot"
        )

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR  = os.path.join(BASE_DIR, "calibration")
CALIB_FILE = os.path.join(CALIB_DIR, "robot_calib.json")

os.makedirs(CALIB_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 모터 설정
# ─────────────────────────────────────────────
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}

# SO101 center 자세 안내 메시지
CENTER_INSTRUCTION = """
  팔꿈치를 어깨 높이로 들고,
  팔꿈치 90도로 굽혀 전완을 정면으로 뻗은 자세.
"""

# 각 모터의 min/max 안내
RANGE_INSTRUCTIONS = {
    "shoulder_pan": {
        "min": "베이스를 오른쪽으로 최대한 회전",
        "max": "베이스를 왼쪽으로 최대한 회전",
    },
    "shoulder_lift": {
        "min": "어깨를 가장 낮게 (팔 내린 자세)",
        "max": "어깨를 가장 높게 (팔 위로 최대)",
    },
    "elbow_flex": {
        "min": "팔꿈치 완전히 펼친 자세",
        "max": "팔꿈치 최대로 굽힌 자세",
    },
    "wrist_flex": {
        "min": "손목 최대 extension (손등 방향)",
        "max": "손목 최대 flexion (손바닥 방향)",
    },
    "wrist_roll": {
        "min": "손목 최대 pronation (손바닥 아래)",
        "max": "손목 최대 supination (손바닥 위)",
    },
    "gripper": {
        "min": "그리퍼 완전히 닫힘",
        "max": "그리퍼 완전히 열림",
    },
}

# ─────────────────────────────────────────────
# 버스 연결
# ─────────────────────────────────────────────

def connect_bus(port: str) -> FeetechMotorsBus:
    """FeetechMotorsBus 연결 (LeRobot API 버전 자동 대응)"""
    if LEROBOT_NEW_API:
        bus = FeetechMotorsBus(
            port=port,
            motors={
                name: Motor(idx, "sts3215", MotorNormMode.RANGE_M100_100)
                for name, idx in MOTOR_IDS.items()
            },
        )
    else:
        config = FeetechMotorsBusConfig(
            port=port,
            motors={name: [idx, "sts3215"] for name, idx in MOTOR_IDS.items()},
        )
        bus = FeetechMotorsBus(config)

    bus.connect()
    return bus


def disable_torque(bus: FeetechMotorsBus):
    """모든 모터 토크 OFF → 손으로 자유롭게 움직일 수 있음"""
    for name in MOTOR_NAMES:
        bus.write("Torque_Enable", name, 0)
    print("  [토크 OFF] 모터를 손으로 자유롭게 움직일 수 있습니다.")


def enable_torque(bus: FeetechMotorsBus):
    """모든 모터 토크 ON"""
    for name in MOTOR_NAMES:
        bus.write("Torque_Enable", name, 1)
    print("  [토크 ON]")


def read_all_positions(bus: FeetechMotorsBus) -> dict:
    """모든 모터의 현재 raw 엔코더 값 읽기 (정규화 없이)"""
    positions = {}
    for name in MOTOR_NAMES:
        try:
            # 최신 LeRobot API: normalize=False로 raw 값 읽기
            val = bus.read("Present_Position", name, normalize=False)
        except TypeError:
            # 구 API는 normalize 파라미터 없음
            val = bus.read("Present_Position", name)
        # read가 배열/단일값 둘 다 반환할 수 있음
        if hasattr(val, "__iter__"):
            val = int(val[0])
        else:
            val = int(val)
        positions[name] = val
    return positions


def print_positions(positions: dict, highlight: str = None):
    """현재 위치 출력. highlight 모터는 강조 표시"""
    print("\n  현재 모터 위치 (raw encoder):")
    for name, val in positions.items():
        marker = "  ◀" if name == highlight else ""
        print(f"    {name:15s}: {val:5d}{marker}")


# ─────────────────────────────────────────────
# 캘리브레이션 메인
# ─────────────────────────────────────────────

def run_calibration(port: str):
    print("\n" + "=" * 60)
    print("  SO101 Robot Calibration")
    print("=" * 60)
    print(f"  포트: {port}")
    print("  조작: Enter → 현재 값 기록 / Ctrl+C → 중단")
    print("=" * 60)

    # 연결
    print("\n로봇 연결 중...")
    bus = connect_bus(port)
    print("  연결 완료.")

    # 토크 OFF
    disable_torque(bus)

    recorded = {}

    try:
        # ── Step 1: CENTER ────────────────────────
        print("\n" + "─" * 60)
        print("  [STEP 1]  CENTER 자세")
        print("─" * 60)
        print(CENTER_INSTRUCTION)
        print("  모든 모터를 center 자세로 맞춘 후 Enter를 누르세요.")

        input("\n  >>> Enter 입력: ")
        center_positions = read_all_positions(bus)
        print_positions(center_positions)
        print("\n  ✓ Center 위치 기록 완료.")

        for name in MOTOR_NAMES:
            recorded[name] = {"center": center_positions[name]}

        # ── Step 2: MIN / MAX per motor ───────────
        print("\n" + "─" * 60)
        print("  [STEP 2]  각 모터 MIN / MAX 범위")
        print("─" * 60)
        print("  각 모터를 하나씩 끝까지 움직이면서 Enter로 기록합니다.")
        print("  (나머지 모터는 가능한 center 위치 유지)")

        for motor_name in MOTOR_NAMES:
            print(f"\n  ── {motor_name} ──")

            # MIN
            min_desc = RANGE_INSTRUCTIONS[motor_name]["min"]
            print(f"  MIN: {min_desc}")
            input("  >>> 이 자세로 맞추고 Enter: ")
            pos = read_all_positions(bus)
            min_val = pos[motor_name]
            recorded[motor_name]["min"] = min_val
            print(f"  ✓ {motor_name}  min = {min_val}")

            # MAX
            max_desc = RANGE_INSTRUCTIONS[motor_name]["max"]
            print(f"\n  MAX: {max_desc}")
            input("  >>> 이 자세로 맞추고 Enter: ")
            pos = read_all_positions(bus)
            max_val = pos[motor_name]
            recorded[motor_name]["max"] = max_val
            print(f"  ✓ {motor_name}  max = {max_val}")

            # min > max 인 경우 경고 (반전된 모터)
            if min_val > max_val:
                print(f"  [INFO] {motor_name}: min > max → 이 모터는 반전 방향입니다. (정상)")

    except KeyboardInterrupt:
        print("\n\n캘리브레이션 중단.")
        bus.disconnect()
        return

    # 토크 복원
    enable_torque(bus)
    bus.disconnect()

    # 저장
    save_calibration(recorded)


def save_calibration(recorded: dict):
    """캘리브레이션 결과 JSON 저장"""

    # raw encoder → degree 변환용 메타 포함
    ENCODER_RESOLUTION = 4096   # STS3215 1회전 = 4096 steps
    DEGREE_PER_STEP    = 360.0 / ENCODER_RESOLUTION  # ≈ 0.0879°/step

    output = {}
    for name in MOTOR_NAMES:
        if name not in recorded:
            continue
        vals = recorded[name]
        center = vals.get("center", 2048)
        mn     = vals.get("min",    0)
        mx     = vals.get("max",    4095)

        # homing_offset: center를 0 기준으로 설정
        homing_offset = center - 2048

        output[name] = {
            "center":         center,
            "min":            mn,
            "max":            mx,
            "homing_offset":  homing_offset,
            "drive_mode":     0,
            "calib_mode":     "DEGREE",
            "motor_resolution": ENCODER_RESOLUTION,
        }

    with open(CALIB_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("  캘리브레이션 완료!")
    print(f"  저장 위치: {CALIB_FILE}")
    print("=" * 60)
    print("\n결과 요약:")
    print(f"  {'모터':15s}  {'center':>7s}  {'min':>7s}  {'max':>7s}  {'homing_offset':>14s}")
    print("  " + "-" * 58)
    for name, vals in output.items():
        print(f"  {name:15s}  {vals['center']:7d}  {vals['min']:7d}  {vals['max']:7d}  {vals['homing_offset']:14d}")
    print()


# ─────────────────────────────────────────────
# 저장된 캘리브레이션 로드 (다른 모듈에서 import)
# ─────────────────────────────────────────────

def load_calibration() -> dict | None:
    """
    저장된 로봇 캘리브레이션 로드.
    텔레오퍼레이션 루프에서 사용.

    Returns:
        dict | None
    """
    if not os.path.exists(CALIB_FILE):
        print(f"[WARN] 로봇 캘리브레이션 파일 없음: {CALIB_FILE}")
        return None
    with open(CALIB_FILE) as f:
        return json.load(f)


def raw_to_degree(raw: int, center: int,
                  resolution: int = 4096) -> float:
    """raw 엔코더값 → degree 변환 (center = 0°)"""
    return (raw - center) * (360.0 / resolution)


def degree_to_raw(degree: float, center: int,
                  resolution: int = 4096) -> int:
    """degree → raw 엔코더값 변환"""
    return int(center + degree * resolution / 360.0)


# ─────────────────────────────────────────────
# 포트 자동 탐색 (Mac 기준)
# ─────────────────────────────────────────────

def find_robot_port() -> str | None:
    """연결된 Feetech 버스 포트 자동 탐색"""
    import glob
    patterns = [
        "/dev/tty.usbmodem*",   # macOS
        "/dev/ttyACM*",         # Linux
        "/dev/ttyUSB*",         # Linux (USB-serial)
    ]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    return None


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SO101 Robot Calibration")
    parser.add_argument(
        "--port", type=str, default=None,
        help="시리얼 포트 (예: /dev/tty.usbmodem585A0076841). "
             "미입력 시 자동 탐색."
    )
    args = parser.parse_args()

    port = args.port
    if port is None:
        port = find_robot_port()
        if port is None:
            print("[ERROR] 로봇 포트를 찾을 수 없습니다.")
            print("  USB 연결을 확인하거나 --port 옵션으로 직접 지정하세요.")
            exit(1)
        print(f"  포트 자동 감지: {port}")

    run_calibration(port)