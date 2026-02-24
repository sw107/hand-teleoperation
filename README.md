# Vision-Based Teleoperation for SO101

MediaPipe를 사용한 SO101 로봇팔 비전 기반 텔레오퍼레이션

## 개요

웹캠으로 사람의 팔 움직임을 추적하여 SO101 로봇팔을 제어합니다.
- **팔 추적**: MediaPipe Pose (어깨, 팔꿈치, 손목)
- **그리퍼 제어**: MediaPipe Hands (엄지-검지 거리)
- **실시간 제어**: 30+ FPS

## 요구사항

### 하드웨어
- 웹캠 (720p 이상 권장)
- SO101 로봇팔

### 소프트웨어
- Python 3.8+
- Linux/macOS

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/vision-teleop-so101.git
cd vision-teleop-so101
```

### 2. 패키지 설치
```bash
pip install mediapipe>=0.10.0 opencv-python>=4.8.0 numpy>=1.24.0
```

### 3. MediaPipe 모델 다운로드 ⭐ 필수!

**자동 다운로드 (추천)**:
```bash
python download_models.py
```

**수동 다운로드**:
```bash
mkdir models

# Pose Landmarker
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O models/pose_landmarker.task

# Hand Landmarker
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O models/hand_landmarker.task
```

## 사용 방법

### 추적 테스트
```bash
python test_tracking.py
```
- 웹캠 화면에서 팔 추적 확인
- 각 관절 각도 실시간 표시
- 그리퍼 상태 (엄지-검지 거리)
- `q` 키로 종료

## 제어 방식

### 관절 매핑
| 사람 | SO101 |
|------|-------|
| 어깨 좌우 회전 | Joint 1 |
| 어깨 들어올림 | Joint 2 |
| 팔꿈치 굽힘 | Joint 3 |
| 손목 위아래 | Joint 4 |
| 손목 회전 | Joint 5 |

### 그리퍼
- **손 펴기** → 그리퍼 열림 (1.0)
- **손 오므리기** → 그리퍼 닫힘 (0.0)
- 엄지-검지 끝 거리로 제어

## 파일 구조
```
hand-teleoperation/
├── models/                    # MediaPipe 모델 파일
│   ├── pose_landmarker.task   # Pose 추적 모델
│   └── hand_landmarker.task   # Hand 추적 모델
├── hand_teleop
│   ├── test_tracking.py           # 추적 테스트 스크립트
├── download_models.py         # 모델 다운로드 스크립트
└── README.md
```

## 문제 해결

### 모델 파일 에러
```
FileNotFoundError: pose_landmarker.task not found
```
**해결**: `python download_models.py` 실행

### 카메라 인식 안 됨
```bash
# 사용 가능한 카메라 확인
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# 카메라 번호 변경 (코드에서 camera_index 수정)
cap = cv2.VideoCapture(1)  # 0 대신 1 시도
```

### 팔 인식 안 됨
- 조명 확인 (너무 어둡거나 밝지 않게)
- 카메라 거리 조정 (1-2m 권장)
- 팔이 몸에 가려지지 않게
- 긴팔 옷 착용 권장


## 성능 최적화

### 느릴 때 (Lite 모델 사용)
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -O models/pose_landmarker.task
```

### 정확도 올리기 (Heavy 모델 사용)
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task -O models/pose_landmarker.task
```

## 모델 비교

| 모델 | 크기 | 속도 | 정확도 | 용도 |
|------|------|------|--------|------|
| Lite | 4MB | 60+ FPS | 보통 | 빠른 테스트 |
| Full | 12MB | 30-45 FPS | 좋음 | ⭐ 권장 |
| Heavy | 26MB | 20-30 FPS | 최고 | 정밀 제어 |

## 단축키

| 키 | 동작 |
|----|------|
| `q` | 종료 |

## 참고

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [LeRobot](https://github.com/huggingface/lerobot)
- [SO101](https://github.com/TheRobotStudio/SO-ARM100)

## 라이선스

MIT License