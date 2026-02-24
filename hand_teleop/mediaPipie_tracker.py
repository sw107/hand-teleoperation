import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque

# Pose 연결선 (팔만)
POSE_CONNECTIONS = [
    (12, 14), (14, 16),
    (16, 18), (16, 20),
    (16, 22), (18, 20),
]
VISIBLE_POSE_LANDMARKS = {12, 14, 16, 18, 20, 22}

latest_pose = None
latest_hands = None

# 그리퍼 히스토리 (부드러운 제어용)
gripper_history = {
    'right': deque(maxlen=5),
    # 'left': deque(maxlen=5)
}

def calculate_angle_3d(a, b, c):
    """3개 점으로 각도 계산 (3D)"""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def calculate_shoulder_pan(shoulder, elbow):
    """어깨 좌우 회전 각도"""
    angle = np.arctan2(elbow.y - shoulder.y, elbow.x - shoulder.x)
    return np.degrees(angle)

def calculate_shoulder_lift(shoulder, elbow, hip):
    """어깨 들어올림 각도"""
    v_body = np.array([shoulder.x - hip.x, shoulder.y - hip.y, shoulder.z - hip.z])
    v_arm = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z])
    
    cosine = np.dot(v_body, v_arm) / (np.linalg.norm(v_body) * np.linalg.norm(v_arm) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def calculate_wrist_pitch(elbow, wrist, hand_direction):
    """손목 pitch (위아래 각도)"""
    return calculate_angle_3d(elbow, wrist, hand_direction)

def calculate_wrist_roll(wrist, index_mcp, pinky_mcp):
    """손목 roll (회전 각도)"""
    palm_vector = np.array([
        pinky_mcp.x - index_mcp.x,
        pinky_mcp.y - index_mcp.y,
        pinky_mcp.z - index_mcp.z
    ])
    
    horizontal = np.array([1, 0, 0])
    cosine = np.dot(palm_vector, horizontal) / (np.linalg.norm(palm_vector) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def calculate_gripper_state(hand_landmarks, smooth=True, hand_side='right'):
    """
    엄지-검지 거리로 그리퍼 상태 계산
    
    Args:
        hand_landmarks: 손 랜드마크
        smooth: 부드러운 제어 활성화
        hand_side: 'right' 또는 'left'
    
    Returns:
        float: 0.0 (완전 닫힘) ~ 1.0 (완전 열림)
    """
    # 엄지 끝 (landmark 4)
    thumb_tip = hand_landmarks[4]
    
    # 검지 끝 (landmark 8)
    index_tip = hand_landmarks[8]
    
    # 3D 유클리드 거리 계산
    distance = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )
    
    # 거리를 그리퍼 값으로 정규화
    # 실험적으로 결정된 값 (카메라와 손 거리에 따라 조정 필요)
    MIN_DIST = 0.02   # 완전히 붙었을 때
    MAX_DIST = 0.12   # 최대로 벌렸을 때
    
    # 0~1로 정규화
    gripper = (distance - MIN_DIST) / (MAX_DIST - MIN_DIST)
    gripper = np.clip(gripper, 0.0, 1.0)
    
    # 부드러운 제어 (이동평균)
    if smooth:
        gripper_history[hand_side].append(gripper)
        if len(gripper_history[hand_side]) > 0:
            gripper = np.mean(gripper_history[hand_side])
    
    return gripper

def find_hand_for_wrist(wrist_landmark, hand_landmarks_list, w, h):
    """Pose의 손목과 가장 가까운 Hand 찾기"""
    if not hand_landmarks_list:
        return None
    
    wrist_pos = np.array([wrist_landmark.x * w, wrist_landmark.y * h])
    min_dist = float('inf')
    closest_hand = None
    
    for hand in hand_landmarks_list:
        hand_wrist = hand[0]
        hand_wrist_pos = np.array([hand_wrist.x * w, hand_wrist.y * h])
        
        dist = np.linalg.norm(wrist_pos - hand_wrist_pos)
        if dist < min_dist:
            min_dist = dist
            closest_hand = hand
    
    if min_dist > 100:  # 100픽셀 이상 차이나면 매칭 실패
        return None
    
    return closest_hand

def calculate_arm_and_gripper(pose_landmarks, hand_landmarks_list, w, h):
    """
    팔 각도 + 그리퍼 상태 모두 계산
    """
    angles = {}
    
    # 오른팔
    if len(pose_landmarks) > 16:
        right_shoulder = pose_landmarks[12]
        right_elbow = pose_landmarks[14]
        right_wrist = pose_landmarks[16]
        right_hip = pose_landmarks[24] if len(pose_landmarks) > 24 else pose_landmarks[12]
        
        # 어깨, 팔꿈치 각도
        angles['right_elbow'] = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
        angles['right_shoulder_pan'] = calculate_shoulder_pan(right_shoulder, right_elbow)
        angles['right_shoulder_lift'] = calculate_shoulder_lift(right_shoulder, right_elbow, right_hip)
        
        # 손목 각도 + 그리퍼
        right_hand = find_hand_for_wrist(right_wrist, hand_landmarks_list, w, h)
        if right_hand:
            # 손목 각도
            middle_finger_tip = right_hand[12]
            angles['right_wrist_pitch'] = calculate_wrist_pitch(
                right_elbow, right_wrist, middle_finger_tip
            )
            
            index_mcp = right_hand[5]
            pinky_mcp = right_hand[17]
            angles['right_wrist_roll'] = calculate_wrist_roll(
                right_wrist, index_mcp, pinky_mcp
            )
            
            # 그리퍼 (엄지-검지 거리)
            angles['right_gripper'] = calculate_gripper_state(right_hand, smooth=True, hand_side='right')
        else:
            # 손 안 보이면 그리퍼 열림으로 기본값
            angles['right_gripper'] = 1.0
    
    # # 왼팔
    # if len(pose_landmarks) > 15:
    #     left_shoulder = pose_landmarks[11]
    #     left_elbow = pose_landmarks[13]
    #     left_wrist = pose_landmarks[15]
    #     left_hip = pose_landmarks[23] if len(pose_landmarks) > 23 else pose_landmarks[11]
        
    #     angles['left_elbow'] = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
    #     angles['left_shoulder_pan'] = calculate_shoulder_pan(left_shoulder, left_elbow)
    #     angles['left_shoulder_lift'] = calculate_shoulder_lift(left_shoulder, left_elbow, left_hip)
        
    #     left_hand = find_hand_for_wrist(left_wrist, hand_landmarks_list, w, h)
    #     if left_hand:
    #         middle_finger_tip = left_hand[12]
    #         angles['left_wrist_pitch'] = calculate_wrist_pitch(
    #             left_elbow, left_wrist, middle_finger_tip
    #         )
            
    #         index_mcp = left_hand[5]
    #         pinky_mcp = left_hand[17]
    #         angles['left_wrist_roll'] = calculate_wrist_roll(
    #             left_wrist, index_mcp, pinky_mcp
    #         )
            
    #         angles['left_gripper'] = calculate_gripper_state(left_hand, smooth=True, hand_side='left')
    #     else:
    #         angles['left_gripper'] = 1.0
    
    return angles

def visualize_gripper(frame, hand_landmarks, gripper_value, w, h, hand_side='right'):
    """
    그리퍼 상태 시각화 (엄지-검지 사이 선)
    """
    # 엄지 끝
    thumb_tip = hand_landmarks[4]
    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    
    # 검지 끝
    index_tip = hand_landmarks[8]
    index_pos = (int(index_tip.x * w), int(index_tip.y * h))
    
    # 색상: 초록(열림) → 빨강(닫힘)
    green = int(255 * gripper_value)
    red = int(255 * (1 - gripper_value))
    color = (0, green, red)  # BGR
    
    # 선 굵기: 닫힐수록 얇게
    thickness = max(int(8 * gripper_value), 2)
    
    # 엄지-검지 사이 선
    cv2.line(frame, thumb_pos, index_pos, color, thickness)
    
    # 양 끝에 큰 점
    cv2.circle(frame, thumb_pos, 8, (255, 255, 0), -1)
    cv2.circle(frame, index_pos, 8, (255, 255, 0), -1)
    
    # 그리퍼 값 표시
    mid_x = (thumb_pos[0] + index_pos[0]) // 2
    mid_y = (thumb_pos[1] + index_pos[1]) // 2
    
    text = f"{gripper_value:.2f}"
    cv2.putText(frame, text, (mid_x - 30, mid_y - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 상태 텍스트
    if gripper_value < 0.2:
        status = "CLOSED"
        status_color = (0, 0, 255)
    elif gripper_value > 0.8:
        status = "OPEN"
        status_color = (0, 255, 0)
    else:
        status = "PARTIAL"
        status_color = (0, 255, 255)
    
    cv2.putText(frame, status, (mid_x - 35, mid_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

def draw_angles_on_frame(frame, pose_landmarks, angles, w, h):
    """각도를 화면에 표시"""
    # 오른팔
    if 'right_elbow' in angles:
        elbow = pose_landmarks[14]
        ex, ey = int(elbow.x * w), int(elbow.y * h)
        cv2.putText(frame, f"E: {angles['right_elbow']:.1f}°", (ex + 20, ey), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    if 'right_wrist_pitch' in angles:
        wrist = pose_landmarks[16]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, f"W: {angles['right_wrist_pitch']:.1f}°", (wx + 20, wy + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # # 왼팔
    # if 'left_elbow' in angles:
    #     elbow = pose_landmarks[13]
    #     ex, ey = int(elbow.x * w), int(elbow.y * h)
    #     cv2.putText(frame, f"E: {angles['left_elbow']:.1f}°", (ex - 100, ey), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # if 'left_wrist_pitch' in angles:
    #     wrist = pose_landmarks[15]
    #     wx, wy = int(wrist.x * w), int(wrist.y * h)
    #     cv2.putText(frame, f"W: {angles['left_wrist_pitch']:.1f}°", (wx - 100, wy + 20), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame

def display_angles_panel(frame, angles):
    """화면 왼쪽에 각도 패널 표시"""
    panel_height = 200
    panel_width = 320
    y_start = 50
    
    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y_start), (10 + panel_width, y_start + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 헤더
    y_offset = y_start + 30
    cv2.putText(frame, "=== angles ===", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    # 오른팔
    y_offset += 35
    cv2.putText(frame, "RIGHT ARM:", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    y_offset += 25
    right_joints = [
        ('right_shoulder_pan', 'Shoulder Pan'),
        ('right_shoulder_lift', 'Shoulder Lift'),
        ('right_elbow', 'Elbow'),
        ('right_wrist_pitch', 'Wrist Pitch'),
        ('right_wrist_roll', 'Wrist Roll'),
        ('right_gripper', 'Gripper')
    ]
    
    for joint_key, joint_name in right_joints:
        if joint_key in angles:
            if joint_key == 'right_gripper':
                # 그리퍼는 0~1 값
                text = f"  {joint_name}: {angles[joint_key]:.2f}"
                
                # 색상 코드
                if angles[joint_key] < 0.2:
                    color = (0, 0, 255)  # 빨강 (닫힘)
                elif angles[joint_key] > 0.8:
                    color = (0, 255, 0)  # 초록 (열림)
                else:
                    color = (0, 255, 255)  # 노랑 (중간)
            else:
                # 각도는 도(°) 단위
                text = f"  {joint_name}: {angles[joint_key]:.1f}°"
                color = (0, 255, 255)
            
            cv2.putText(frame, text, (25, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 22
    
    # # 왼팔
    # y_offset += 10
    # cv2.putText(frame, "LEFT ARM:", (20, y_offset), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # y_offset += 25
    # left_joints = [
    #     ('left_shoulder_pan', 'Shoulder Pan'),
    #     ('left_shoulder_lift', 'Shoulder Lift'),
    #     ('left_elbow', 'Elbow'),
    #     ('left_wrist_pitch', 'Wrist Pitch'),
    #     ('left_wrist_roll', 'Wrist Roll'),
    #     ('left_gripper', 'Gripper')
    # ]
    
    # for joint_key, joint_name in left_joints:
    #     if joint_key in angles:
    #         if joint_key == 'left_gripper':
    #             text = f"  {joint_name}: {angles[joint_key]:.2f}"
    #             if angles[joint_key] < 0.2:
    #                 color = (0, 0, 255)
    #             elif angles[joint_key] > 0.8:
    #                 color = (0, 255, 0)
    #             else:
    #                 color = (0, 255, 255)
    #         else:
    #             text = f"  {joint_name}: {angles[joint_key]:.1f}°"
    #             color = (0, 255, 255)
            
    #         cv2.putText(frame, text, (25, y_offset), 
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    #         y_offset += 22
    
    return frame

def on_pose(result, image, timestamp):
    global latest_pose
    latest_pose = result

def on_hands(result, image, timestamp):
    global latest_hands
    latest_hands = result


if __name__ == "__main__":
    # Pose 설정
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='models/pose_landmarker.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=on_pose
    )

    # Hands 설정
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=on_hands
    )
    
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(0)
    timestamp = 0

    print("=" * 50)
    print("    SO101 텔레오퍼레이션 - 팔 각도 + 그리퍼 제어")
    print("=" * 50)
    print("\n제어 방법:")
    print("  - 팔 움직임: 어깨, 팔꿈치, 손목 각도 추적")
    print("  - 그리퍼: 엄지-검지 거리로 제어")
    print("    * 손 펴기 → 그리퍼 열림 (1.0)")
    print("    * 손 오므리기 → 그리퍼 닫힘 (0.0)")
    print("\n단축키:")
    print("  q: 종료")
    print("=" * 50)
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_landmarker.detect_async(mp_image, timestamp)
        hand_landmarker.detect_async(mp_image, timestamp)

        h, w, _ = frame.shape

        # Pose 그리기 + 각도 + 그리퍼 계산
        if latest_pose and latest_pose.pose_landmarks:
            pose_landmarks = latest_pose.pose_landmarks[0]
            
            # 연결선 그리기
            for start, end in POSE_CONNECTIONS:
                x1, y1 = int(pose_landmarks[start].x * w), int(pose_landmarks[start].y * h)
                x2, y2 = int(pose_landmarks[end].x * w), int(pose_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 점 그리기
            for i, lm in enumerate(pose_landmarks):
                if i in VISIBLE_POSE_LANDMARKS:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)
            
            # Hands 데이터
            hand_landmarks_list = latest_hands.hand_landmarks if latest_hands else None
            
            # 팔 각도 + 그리퍼 계산
            angles = calculate_arm_and_gripper(pose_landmarks, hand_landmarks_list, w, h)
            
            # 각도 표시
            frame = draw_angles_on_frame(frame, pose_landmarks, angles, w, h)
            
            # 패널 표시
            frame = display_angles_panel(frame, angles)
            
            # 그리퍼 시각화 (엄지-검지 선)
            if hand_landmarks_list:
                for hand in hand_landmarks_list:
                    # 오른손인지 왼손인지 판별 (간단히 x 좌표로)
                    hand_x = hand[0].x
                    is_right = hand_x < 0.5  # 화면 왼쪽이면 오른손 (미러 모드)
                    
                    if is_right and 'right_gripper' in angles:
                        visualize_gripper(frame, hand, angles['right_gripper'], w, h, 'right')
                    elif not is_right and 'left_gripper' in angles:
                        visualize_gripper(frame, hand, angles['left_gripper'], w, h, 'left')
            
            # 콘솔 출력 (매 30프레임마다)
            if timestamp % 30 == 0:
                print(f"\n--- Frame {timestamp} ---")
                if 'right_elbow' in angles:
                    print("RIGHT ARM:")
                    for key in ['right_shoulder_pan', 'right_shoulder_lift', 'right_elbow', 
                            'right_wrist_pitch', 'right_wrist_roll']:
                        if key in angles:
                            print(f"  {key}: {angles[key]:.2f}°")
                    if 'right_gripper' in angles:
                        print(f"  right_gripper: {angles['right_gripper']:.3f} ({'OPEN' if angles['right_gripper'] > 0.7 else 'CLOSED' if angles['right_gripper'] < 0.3 else 'PARTIAL'})")
                
                # if 'left_elbow' in angles:
                #     print("LEFT ARM:")
                #     for key in ['left_shoulder_pan', 'left_shoulder_lift', 'left_elbow', 
                #                'left_wrist_pitch', 'left_wrist_roll']:
                #         if key in angles:
                #             print(f"  {key}: {angles[key]:.2f}°")
                #     if 'left_gripper' in angles:
                #         print(f"  left_gripper: {angles['left_gripper']:.3f} ({'OPEN' if angles['left_gripper'] > 0.7 else 'CLOSED' if angles['left_gripper'] < 0.3 else 'PARTIAL'})")


        cv2.imshow('SO101 Teleoperation - Full Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n프로그램 종료")