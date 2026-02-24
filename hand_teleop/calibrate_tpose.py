import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import yaml
from datetime import datetime
from collections import deque

# 이전에 만든 함수들 import (또는 여기에 복사)
from mediaPipie_tracker import (
    calculate_angle_3d,
    calculate_shoulder_pan,
    calculate_shoulder_lift,
    calculate_wrist_pitch,
    calculate_wrist_roll,
    find_hand_for_wrist,
    calculate_gripper_state,
    on_hands,
    on_pose,
    calculate_arm_and_gripper,
)

# T-Pose 기준 각도 (오른팔만, 허용 범위)
TPOSE_REFERENCE = {
    'right_shoulder_pan': (80, 100),      # 90° ± 10° (팔을 옆으로)
    'right_shoulder_lift': (80, 100),     # 90° ± 10° (수평)
    'right_elbow': (170, 180),            # 180° (쭉 펴짐)
    'right_wrist_pitch': (80, 100),       # 90° ± 10°
    'right_wrist_roll': (-10, 10),        # 0° ± 10°
}

latest_pose = None
latest_hands = None


def calculate_arm_angles(pose_landmarks, hand_landmarks_list, w, h):
    """오른팔 각도만 계산"""
    angles = {}
    
    if len(pose_landmarks) > 16:
        right_shoulder = pose_landmarks[12]
        right_elbow = pose_landmarks[14]
        right_wrist = pose_landmarks[16]
        right_hip = pose_landmarks[24] if len(pose_landmarks) > 24 else pose_landmarks[12]
        
        angles['right_elbow'] = calculate_angle_3d(right_shoulder, right_elbow, right_wrist)
        angles['right_shoulder_pan'] = calculate_shoulder_pan(right_shoulder, right_elbow)
        angles['right_shoulder_lift'] = calculate_shoulder_lift(right_shoulder, right_elbow, right_hip)
        
        right_hand = find_hand_for_wrist(right_wrist, hand_landmarks_list, w, h)
        if right_hand:
            middle_finger_tip = right_hand[12]
            angles['right_wrist_pitch'] = calculate_wrist_pitch(right_elbow, right_wrist, middle_finger_tip)
            
            index_mcp = right_hand[5]
            pinky_mcp = right_hand[17]
            angles['right_wrist_roll'] = calculate_wrist_roll(right_wrist, index_mcp, pinky_mcp)
    
    return angles

def check_tpose(angles):
    """
    T-Pose 자세인지 확인 (오른팔만)
    
    Returns:
        (bool, dict): (T-Pose 여부, 각 관절별 상태)
    """
    status = {}
    all_good = True
    
    for joint, (min_val, max_val) in TPOSE_REFERENCE.items():
        if joint in angles:
            current = angles[joint]
            in_range = min_val <= current <= max_val
            status[joint] = {
                'current': current,
                'target': (min_val + max_val) / 2,
                'ok': in_range,
                'error': abs(current - (min_val + max_val) / 2)
            }
            if not in_range:
                all_good = False
        else:
            status[joint] = {'ok': False}
            all_good = False
    
    return all_good, status

def draw_tpose_guide(frame, status):
    """T-Pose 가이드 그리기"""
    h, w = frame.shape[:2]
    
    # 반투명 패널
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 420, 50), (w - 20, 420), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 제목
    cv2.putText(frame, "=== T-POSE (RIGHT ARM) ===", 
               (w - 410, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    y_offset = 130
    
    # 각 관절 상태
    for joint, info in status.items():
        joint_name = joint.replace('right_', '').replace('_', ' ').title()
        
        if info['ok']:
            color = (0, 255, 0)  # 초록
            status_text = "✓ OK"
        else:
            color = (0, 0, 255)  # 빨강
            status_text = f"✗ ±{info['error']:.1f}°"
        
        text = f"{joint_name}:"
        cv2.putText(frame, text, (w - 410, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        if 'current' in info:
            value_text = f"{info['current']:.1f}°"
            cv2.putText(frame, value_text, (w - 230, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.putText(frame, status_text, (w - 120, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        y_offset += 35
    
    # 안내 메시지
    y_offset += 20
    cv2.line(frame, (w - 410, y_offset - 10), (w - 30, y_offset - 10), (100, 100, 100), 1)
    y_offset += 10
    
    cv2.putText(frame, "Ready? Press SPACE", 
               (w - 410, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, "Quit: Q", 
               (w - 410, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    return frame

def save_calibration(angles):
    """캘리브레이션 데이터 저장"""
    calibration_data = {
        'metadata': {
            'reference_pose': 'tpose',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'side': 'right',
            'description': 'Right arm T-Pose calibration'
        },
        'human_reference': {
            'right_shoulder_pan': float(angles['right_shoulder_pan']),
            'right_shoulder_lift': float(angles['right_shoulder_lift']),
            'right_elbow': float(angles['right_elbow']),
            'right_wrist_pitch': float(angles.get('right_wrist_pitch', 90.0)),
            'right_wrist_roll': float(angles.get('right_wrist_roll', 0.0)),
        }
    }
    
    with open('calibration.yaml', 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*60)
    print("✓ Calibration saved to calibration.yaml")
    print("="*60)
    print(f"Timestamp: {calibration_data['metadata']['timestamp']}")
    print("\nHuman Reference Angles:")
    for joint, value in calibration_data['human_reference'].items():
        print(f"  {joint}: {value:.2f}°")
    print("="*60)

def main():
    # MediaPipe 초기화
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='models/pose_landmarker.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=on_pose
    )
    
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
    
    print("=" * 60)
    print("    T-Pose Calibration for SO101 (Right Arm Only)")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Stand facing the camera (1-2m distance)")
    print("2. Extend RIGHT ARM to the side (T-Pose)")
    print("   - Arm straight and horizontal")
    print("   - Palm facing down")
    print("3. LEFT ARM: any position (relaxed)")
    print("4. When all joints are GREEN, press SPACE to save")
    print("5. Press Q to quit")
    print("=" * 60)
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
        
        # 각도 계산
        if latest_pose and latest_pose.pose_landmarks:
            pose_landmarks = latest_pose.pose_landmarks[0]
            hand_landmarks_list = latest_hands.hand_landmarks if latest_hands else None
            
            angles = calculate_arm_angles(pose_landmarks, hand_landmarks_list, w, h)
            
            # T-Pose 확인
            is_tpose, status = check_tpose(angles)
            
            # 가이드 그리기
            frame = draw_tpose_guide(frame, status)
            
            # 전체 상태 표시
            if is_tpose:
                cv2.putText(frame, "T-POSE READY!", (50, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
            else:
                cv2.putText(frame, "Adjust right arm...", (50, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
            
            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE
                if is_tpose:
                    save_calibration(angles)
                    print("\n✓ Calibration complete! You can now close this window.")
                    cv2.waitKey(2000)  # 2초 대기
                    break
                else:
                    print("\n✗ Not in T-Pose yet. Adjust your right arm.")
            elif key == ord('q'):
                print("\nCalibration cancelled.")
                break
        else:
            cv2.putText(frame, "No person detected", (50, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            cv2.waitKey(1)
        
        cv2.imshow('T-Pose Calibration - Right Arm', frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()