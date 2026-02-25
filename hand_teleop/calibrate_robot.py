from scservo_sdk import *
import yaml

def get_port_from_user():
    """사용자에게 포트 직접 입력받기"""
    print("\nEnter the robot's USB port:")
    print("  Example (macOS): /dev/tty.usbmodem575E0032081")
    print("  Example (Linux): /dev/ttyACM0")
    print("\nTip: Check available ports with:")
    print("  macOS: ls /dev/tty.usbmodem*")
    print("  Linux: ls /dev/ttyACM*")
    
    port = input("\nPort: ").strip()
    return port

def read_robot_angles(port_handler, packet_handler):
    """로봇 현재 각도 읽기"""
    
    ADDR_PRESENT_POSITION = 56
    
    motor_names = {
        1: 'joint_1',  # Base / shoulder_pan
        2: 'joint_2',  # Shoulder lift
        3: 'joint_3',  # Elbow flex
        4: 'joint_4',  # Wrist flex
        5: 'joint_5',  # Wrist roll
        6: 'joint_6',  # Gripper
    }
    
    robot_angles = {}
    
    print("\nReading motor positions...")
    for motor_id, joint_name in motor_names.items():
        position, result, error = packet_handler.read4ByteTxRx(
            port_handler, motor_id, ADDR_PRESENT_POSITION
        )
        
        if result == COMM_SUCCESS:
            # Position을 각도로 변환 (0~4095 → 0~360)
            angle = (position / 4095.0) * 360.0
            robot_angles[joint_name] = float(angle)
            print(f"  {joint_name}: {angle:6.2f}°")
        else:
            print(f"  {joint_name}: ✗ Read failed")
            robot_angles[joint_name] = 0.0
    
    return robot_angles

def update_calibration(robot_angles):
    """calibration.yaml 업데이트"""
    
    # 기존 calibration 로드
    try:
        with open('calibration.yaml', 'r') as f:
            calib = yaml.safe_load(f)
            if calib is None:
                calib = {}
    except FileNotFoundError:
        print("\ncalibration.yaml not found!")
        print("Please run calibrate_neutral.py first to calibrate human pose.")
        return False
    
    # 로봇 각도 추가
    calib['robot_reference'] = robot_angles
    
    # Offset 계산
    human = calib.get('human_reference', {})
    
    if not human:
        print("\nNo human reference found in calibration.yaml!")
        print("Please run calibrate_neutral.py first.")
        return False
    
    calib['offsets'] = {
        'joint_1': robot_angles['joint_1'] - human.get('right_shoulder_pan', 0),
        'joint_2': robot_angles['joint_2'] - human.get('right_shoulder_lift', 0),
        'joint_3': robot_angles['joint_3'] - human.get('right_elbow', 0),
        'joint_4': robot_angles['joint_4'] - human.get('right_wrist_pitch', 0),
        'joint_5': robot_angles['joint_5'] - human.get('right_wrist_roll', 0),
        'joint_6': 0,  # Gripper는 별도 제어
    }
    
    # 저장
    with open('calibration.yaml', 'w') as f:
        yaml.dump(calib, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*60)
    print("✓ Robot calibration saved to calibration.yaml")
    print("="*60)
    print("\nRobot Reference Angles:")
    for joint, angle in robot_angles.items():
        print(f"  {joint}: {angle:.2f}°")
    
    print("\nOffsets (robot - human):")
    for joint, offset in calib['offsets'].items():
        print(f"  {joint}: {offset:+.2f}°")
    print("="*60)
    
    return True

def main():
    print("="*60)
    print("    SO101 Robot Calibration (Feetech SDK)")
    print("="*60)
    print("\nThis script calibrates the robot arm to match human pose.")
    print("\nInstructions:")
    print("1. Manually move the robot to match your calibrated human pose:")
    print("   - Base rotated ~90° (arm pointing sideways)")
    print("   - Arm forward")
    print("   - Elbow slightly extended (like your pose)")
    print("   - Wrist straight")
    print("   - Gripper open")
    print("2. Press ENTER to read robot angles")
    print("="*60)
    print()
    
    # 포트 입력받기
    port = get_port_from_user()
    
    print(f"\n✓ Using port: {port}")
    
    # Feetech SDK 초기화
    BAUDRATE = 1000000
    PROTOCOL_VERSION = 0
    
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # 포트 열기
    print("\nConnecting to robot...")
    if not port_handler.openPort():
        print("✗ Failed to open port")
        print("\nTroubleshooting:")
        print("  - Check the port name is correct")
        print("  - Make sure robot USB is connected")
        print("  - Try: ls /dev/tty.* | grep usb")
        return
    
    print("✓ Port opened")
    
    if not port_handler.setBaudRate(BAUDRATE):
        print("✗ Failed to set baudrate")
        port_handler.closePort()
        return
    
    print("✓ Baudrate set")
    
    # 사용자 대기
    print("\n" + "="*60)
    print("Move the robot to the calibration pose...")
    print("="*60)
    input("\nPress ENTER when ready...")
    
    # 각도 읽기
    robot_angles = read_robot_angles(port_handler, packet_handler)
    
    # 포트 닫기
    port_handler.closePort()
    
    # 확인
    print("\n" + "="*60)
    confirm = input("\nSave these angles to calibration.yaml? (y/n): ")
    
    if confirm.lower() == 'y':
        if update_calibration(robot_angles):
            print("\n✓ Calibration complete!")
            print("\nNext step: Run teleoperation")
            print("  python teleoperate.py")
        else:
            print("\n✗ Calibration failed")
    else:
        print("\n✗ Calibration cancelled")

if __name__ == "__main__":
    main()