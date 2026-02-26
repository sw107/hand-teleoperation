import cv2
import json
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
CALIB_DIR   = os.path.join(BASE_DIR, "calibration")
CALIB_FILE  = os.path.join(CALIB_DIR, "human_calib.json")

os.makedirs(CALIB_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Pose 시각화용 연결선/랜드마크
# ─────────────────────────────────────────────
POSE_CONNECTIONS = [
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
]
VISIBLE_POSE_LANDMARKS = {12, 14, 16, 18, 20, 22}

# 손은 엄지 끝(4) - 검지 끝(8) 거리만 표시

# ─────────────────────────────────────────────
# 캘리브레이션 스텝 정의
# ─────────────────────────────────────────────
# (motor_name, step, 안내 메시지)
CALIB_STEPS = [
    # ── shoulder_pan ──────────────────────────
    ("shoulder_pan", "center",
     "CENTER  |  shoulder_pan\n"
     "팔꿈치를 어깨 높이로 들고, 팔꿈치 90도로 굽혀 전완을 앞으로 뻗으세요.\n"
     "(SO101 기본 자세와 동일)"),

    ("shoulder_pan", "min",
     "MIN     |  shoulder_pan\n"
     "팔을 몸 안쪽(오른쪽)으로 최대한 돌리세요."),

    ("shoulder_pan", "max",
     "MAX     |  shoulder_pan\n"
     "팔을 바깥쪽(왼쪽)으로 최대한 돌리세요."),

    # ── shoulder_lift ─────────────────────────
    ("shoulder_lift", "center",
     "CENTER  |  shoulder_lift\n"
     "팔꿈치를 어깨 높이로 들고, 팔꿈치 90도로 굽혀 전완을 앞으로 뻗으세요."),

    ("shoulder_lift", "min",
     "MIN     |  shoulder_lift\n"
     "팔을 자연스럽게 몸 옆으로 완전히 내리세요."),

    ("shoulder_lift", "max",
     "MAX     |  shoulder_lift\n"
     "팔을 머리 위로 최대한 들어올리세요."),

    # ── elbow_flex ────────────────────────────
    ("elbow_flex", "center",
     "CENTER  |  elbow_flex\n"
     "팔꿈치를 정확히 90도로 굽히세요."),

    ("elbow_flex", "min",
     "MIN     |  elbow_flex\n"
     "팔꿈치를 완전히 펴세요 (180도)."),

    ("elbow_flex", "max",
     "MAX     |  elbow_flex\n"
     "팔꿈치를 최대한 굽히세요 (최소각)."),

    # ── wrist_flex ────────────────────────────
    ("wrist_flex", "center",
     "CENTER  |  wrist_flex\n"
     "손목을 중립으로 유지하세요 (전완과 손이 일직선)."),

    ("wrist_flex", "min",
     "MIN     |  wrist_flex\n"
     "손목을 손등 방향으로 최대한 젖히세요 (extension)."),

    ("wrist_flex", "max",
     "MAX     |  wrist_flex\n"
     "손목을 손바닥 방향으로 최대한 굽히세요 (flexion)."),

    # ── wrist_roll ────────────────────────────
    ("wrist_roll", "center",
     "CENTER  |  wrist_roll\n"
     "손바닥이 안쪽(몸 방향)을 향하게 하세요."),

    ("wrist_roll", "min",
     "MIN     |  wrist_roll\n"
     "손바닥이 바닥을 향하게 최대한 회전하세요 (pronation)."),

    ("wrist_roll", "max",
     "MAX     |  wrist_roll\n"
     "손바닥이 천장을 향하게 최대한 회전하세요 (supination)."),

    # ── gripper ───────────────────────────────
    ("gripper", "center",
     "CENTER  |  gripper\n"
     "엄지와 검지를 반쯤 벌린 상태로 유지하세요."),

    ("gripper", "min",
     "MIN     |  gripper\n"
     "엄지와 검지를 완전히 붙이세요 (완전 닫힘)."),

    ("gripper", "max",
     "MAX     |  gripper\n"
     "엄지와 검지를 최대한 벌리세요 (완전 열림)."),
]

# ─────────────────────────────────────────────
# 각도 계산 (mediaPipe_tracker.py와 동일)
# ─────────────────────────────────────────────

def calculate_angle_3d(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_shoulder_pan(shoulder, elbow):
    return np.degrees(np.arctan2(elbow.y - shoulder.y, elbow.x - shoulder.x))

def calculate_shoulder_lift(shoulder, elbow, hip):
    v_body = np.array([shoulder.x - hip.x, shoulder.y - hip.y, shoulder.z - hip.z])
    v_arm  = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z])
    cosine = np.dot(v_body, v_arm) / (np.linalg.norm(v_body) * np.linalg.norm(v_arm) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_wrist_roll(index_mcp, pinky_mcp):
    palm_vector = np.array([
        pinky_mcp.x - index_mcp.x,
        pinky_mcp.y - index_mcp.y,
        pinky_mcp.z - index_mcp.z,
    ])
    horizontal = np.array([1, 0, 0])
    cosine = np.dot(palm_vector, horizontal) / (np.linalg.norm(palm_vector) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_gripper(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    return np.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )

def find_right_hand(wrist_landmark, hand_landmarks_list, w, h):
    """Pose 손목과 가장 가까운 Hand 반환"""
    if not hand_landmarks_list:
        return None
    wrist_pos = np.array([wrist_landmark.x * w, wrist_landmark.y * h])
    best, best_dist = None, float('inf')
    for hand in hand_landmarks_list:
        pos = np.array([hand[0].x * w, hand[0].y * h])
        dist = np.linalg.norm(wrist_pos - pos)
        if dist < best_dist:
            best_dist, best = dist, hand
    return best if best_dist < 100 else None

def get_current_values(pose_landmarks, hand_landmarks_list, w, h):
    """현재 프레임의 6개 관절 값을 모두 계산해서 반환"""
    values = {}

    if len(pose_landmarks) <= 16:
        return values

    shoulder = pose_landmarks[12]
    elbow    = pose_landmarks[14]
    wrist    = pose_landmarks[16]
    hip      = pose_landmarks[24] if len(pose_landmarks) > 24 else pose_landmarks[12]

    values['shoulder_pan']  = calculate_shoulder_pan(shoulder, elbow)
    values['shoulder_lift'] = calculate_shoulder_lift(shoulder, elbow, hip)
    values['elbow_flex']    = calculate_angle_3d(shoulder, elbow, wrist)

    hand = find_right_hand(wrist, hand_landmarks_list, w, h)
    if hand:
        middle_tip  = hand[12]
        index_mcp   = hand[5]
        pinky_mcp   = hand[17]
        values['wrist_flex'] = calculate_angle_3d(elbow, wrist, middle_tip)
        values['wrist_roll'] = calculate_wrist_roll(index_mcp, pinky_mcp)
        values['gripper']    = calculate_gripper(hand)

    return values

# ─────────────────────────────────────────────
# 화면 렌더링
# ─────────────────────────────────────────────

def draw_skeleton(frame, pose_landmarks, hand_landmarks_list, w, h):
    """Pose + Hands 스켈레톤 그리기"""
    # Pose 연결선
    for start, end in POSE_CONNECTIONS:
        x1 = int(pose_landmarks[start].x * w)
        y1 = int(pose_landmarks[start].y * h)
        x2 = int(pose_landmarks[end].x * w)
        y2 = int(pose_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, lm in enumerate(pose_landmarks):
        if i in VISIBLE_POSE_LANDMARKS:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

    # Hands - Pose 손목 기준으로 오른손만 찾아서 표시
    if hand_landmarks_list and pose_landmarks:
        right_wrist = pose_landmarks[16]
        right_hand = find_right_hand(right_wrist, hand_landmarks_list, w, h)
        if right_hand:
            thumb_pos = (int(right_hand[4].x * w), int(right_hand[4].y * h))
            index_pos = (int(right_hand[8].x * w), int(right_hand[8].y * h))
            cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 3)
            cv2.circle(frame, thumb_pos, 8, (255, 255, 0), -1)
            cv2.circle(frame, index_pos, 8, (255, 255, 0), -1)


def draw_instruction_panel(frame, step_index, motor_name, step_type, instruction, countdown):
    """
    상단 안내 패널 + 하단 진행 바
    """
    h_frame, w_frame = frame.shape[:2]
    total_steps = len(CALIB_STEPS)

    # ── 상단 반투명 패널 ──────────────────────
    panel_h = 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 진행 단계
    step_text = f"Step {step_index + 1} / {total_steps}"
    cv2.putText(frame, step_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    # 모터 이름 + 스텝 타입
    type_color = {
        "center": (0, 255, 255),
        "min":    (100, 180, 255),
        "max":    (100, 255, 150),
    }.get(step_type, (255, 255, 255))

    cv2.putText(frame, f"{motor_name}  [{step_type.upper()}]", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, type_color, 2)

    # 안내 문자 (줄 나누기)
    lines = instruction.split("\n")[1:]  # 첫 줄은 헤더이므로 스킵
    for i, line in enumerate(lines):
        cv2.putText(frame, line.strip(), (20, 95 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # ── SPACE 안내 (우측 하단) ────────────────
    if countdown > 0:
        # 카운트다운 중
        cd_text = f"Recording in  {countdown}"
        cv2.putText(frame, cd_text, (w_frame - 260, h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        # 빨간 테두리
        cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), (0, 80, 255), 4)
    else:
        cv2.putText(frame, "SPACE: record    Q: quit", (w_frame - 340, h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # ── 하단 진행 바 ──────────────────────────
    bar_y   = h_frame - 8
    bar_w   = int(w_frame * (step_index / total_steps))
    cv2.rectangle(frame, (0, bar_y - 6), (w_frame, bar_y), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, bar_y - 6), (bar_w, bar_y), type_color, -1)


def draw_current_values(frame, values, target_motor):
    """
    우측 하단에 현재 관절 값 표시.
    캘리브레이션 중인 관절은 강조.
    """
    h_frame, w_frame = frame.shape[:2]
    panel_x = w_frame - 260
    panel_y = 180

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - 10, panel_y - 20),
                  (w_frame - 5, panel_y + 160),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Current Values", (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    items = [
        ("shoulder_pan",  "SH Pan"),
        ("shoulder_lift", "SH Lift"),
        ("elbow_flex",    "Elbow"),
        ("wrist_flex",    "Wr Flex"),
        ("wrist_roll",    "Wr Roll"),
        ("gripper",       "Gripper"),
    ]

    for i, (key, label) in enumerate(items):
        y = panel_y + 25 + i * 23
        is_target = (key == target_motor)
        color = (0, 255, 255) if is_target else (160, 160, 160)

        if key in values:
            if key == "gripper":
                text = f"{label}: {values[key]:.4f}"
            else:
                text = f"{label}: {values[key]:.1f}"
        else:
            text = f"{label}: ---"

        if is_target:
            cv2.rectangle(frame,
                          (panel_x - 5, y - 16),
                          (w_frame - 8, y + 6),
                          (40, 40, 80), -1)

        cv2.putText(frame, text, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if not is_target else 2)


def draw_recorded_summary(frame, recorded):
    """좌측 하단에 기록된 값 요약"""
    h_frame = frame.shape[0]
    y = h_frame - 160

    cv2.putText(frame, "Recorded:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex",   "wrist_roll",    "gripper"]

    for i, motor in enumerate(motors):
        yy = y + 20 + i * 20
        if motor in recorded:
            c = recorded[motor]
            mn = recorded[motor].get("min", "?")
            mx = recorded[motor].get("max", "?")
            ct = recorded[motor].get("center", "?")
            def fmt(v):
                return f"{v:.2f}" if isinstance(v, float) else "?"

            text = f"  {motor[:10]:12s}  C:{fmt(ct)}  mn:{fmt(mn)}  mx:{fmt(mx)}"
            color = (100, 220, 100)
        else:
            text = f"  {motor[:10]:12s}  -"
            color = (80, 80, 80)

        cv2.putText(frame, text, (15, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


# ─────────────────────────────────────────────
# 캘리브레이션 메인
# ─────────────────────────────────────────────

def run_calibration():
    # MediaPipe 설정
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker.task")),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=lambda r, i, t: globals().update(latest_pose=r)
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=lambda r, i, t: globals().update(latest_hands=r)
    )

    global latest_pose, latest_hands
    latest_pose = None
    latest_hands = None

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(0)
    timestamp = 0

    recorded  = {}   # {motor_name: {center/min/max: value}}
    step_idx  = 0
    countdown = 0    # 0 이면 대기중, >0 이면 카운트다운 중
    countdown_start = 0
    COUNTDOWN_SEC = 2

    print("\n" + "="*55)
    print("  Human Arm Calibration for SO101")
    print("="*55)
    print("  SPACE  : 현재 자세 기록 (2초 카운트다운 후 저장)")
    print("  Q      : 종료 (저장 없이)")
    print("="*55 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_landmarker.detect_async(mp_image, timestamp)
        hand_landmarker.detect_async(mp_image, timestamp)

        h, w = frame.shape[:2]

        # 현재 관절 값 계산
        pose_landmarks = None
        hand_landmarks_list = None
        values = {}

        if latest_pose and latest_pose.pose_landmarks:
            pose_landmarks = latest_pose.pose_landmarks[0]
            hand_landmarks_list = latest_hands.hand_landmarks if latest_hands else None
            values = get_current_values(pose_landmarks, hand_landmarks_list, w, h)

            draw_skeleton(frame, pose_landmarks, hand_landmarks_list, w, h)

        # 현재 스텝 정보
        motor_name, step_type, instruction = CALIB_STEPS[step_idx]

        # 카운트다운 처리
        remaining = 0
        if countdown > 0:
            elapsed = time.time() - countdown_start
            remaining = max(0, COUNTDOWN_SEC - int(elapsed))
            if elapsed >= COUNTDOWN_SEC:
                # 기록
                target_val = values.get(motor_name)
                if target_val is not None:
                    if motor_name not in recorded:
                        recorded[motor_name] = {}
                    recorded[motor_name][step_type] = float(target_val)
                    print(f"  [SAVED]  {motor_name}  {step_type:6s}  =  {target_val:.4f}")
                    step_idx += 1
                    if step_idx >= len(CALIB_STEPS):
                        break  # 완료
                else:
                    print(f"  [WARN]  {motor_name} 값 없음. 자세를 확인하세요.")
                countdown = 0

        # UI 렌더링
        draw_instruction_panel(frame, step_idx, motor_name, step_type,
                               instruction, remaining)
        draw_current_values(frame, values, motor_name)
        draw_recorded_summary(frame, recorded)

        cv2.imshow("Human Arm Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n캘리브레이션 중단.")
            break
        elif key == ord(' ') and countdown == 0:
            if motor_name in values:
                countdown = 1
                countdown_start = time.time()
                print(f"  Recording  {motor_name}  {step_type}  in {COUNTDOWN_SEC}s ...")
            else:
                print(f"  [WARN]  {motor_name} 값이 감지되지 않습니다. 자세를 확인하세요.")

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()

    # ── 저장 ──────────────────────────────────
    if step_idx >= len(CALIB_STEPS):
        save_calibration(recorded)
    else:
        print(f"\n캘리브레이션 미완료 ({step_idx}/{len(CALIB_STEPS)} 스텝). 저장하지 않습니다.")


def save_calibration(recorded):
    """캘리브레이션 결과를 JSON으로 저장"""
    output = {}
    for motor, vals in recorded.items():
        center = vals.get("center", 0.0)
        mn     = vals.get("min", 0.0)
        mx     = vals.get("max", 0.0)
        output[motor] = {
            "center": center,
            "min":    mn,
            "max":    mx,
        }

    with open(CALIB_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*55)
    print("  캘리브레이션 완료!")
    print(f"  저장 위치: {CALIB_FILE}")
    print("="*55)
    print("\n결과:")
    for motor, vals in output.items():
        print(f"  {motor:15s}  center={vals['center']:8.3f}  "
              f"min={vals['min']:8.3f}  max={vals['max']:8.3f}")
    print()


def load_calibration():
    """
    저장된 캘리브레이션 로드.
    다른 모듈에서 import해서 사용.

    Returns:
        dict | None
    """
    if not os.path.exists(CALIB_FILE):
        print(f"[WARN] 캘리브레이션 파일 없음: {CALIB_FILE}")
        return None
    with open(CALIB_FILE, "r") as f:
        return json.load(f)


def normalize(value, center, min_val, max_val):
    """
    캘리브레이션 기준으로 -1.0 ~ +1.0 정규화.
    텔레오퍼레이션 루프에서 사용.
    """
    if value >= center:
        denom = max_val - center
        return (value - center) / denom if denom != 0 else 0.0
    else:
        denom = center - min_val
        return (value - center) / denom if denom != 0 else 0.0


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_calibration()