import cv2
import threading
import queue
from ultralytics import YOLO

# -----------------------------
# 설정
# -----------------------------
CAMERA_TYPE = "CSI"  # CSI, ISP, CCTV
VIDEO_SOURCE = 0     # CSI/ISP: 장치 번호, CCTV: RTSP 주소
FRAME_QUEUE_SIZE = 5  # RTSQ 큐 크기

# YOLO 모델 로드 (예: yolov8s.pt)
model = YOLO("yolov8s.pt")

# 큐 생성
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

# 종료 플래그
stop_flag = False

# -----------------------------
# 캡처 스레드
# -----------------------------
def capture_thread():
    global stop_flag
    if CAMERA_TYPE in ["CSI", "ISP"]:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    elif CAMERA_TYPE == "CCTV":
        cap = cv2.VideoCapture(VIDEO_SOURCE)  # RTSP 주소
    
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        # 큐에 넣기
        if not frame_queue.full():
            frame_queue.put(frame)
    
    cap.release()

# -----------------------------
# 전처리/ISP 스레드 (선택)
# -----------------------------
def isp_thread():
    """ISP 카메라 전용 전처리"""
    global stop_flag
    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # 예: 색상 변환, 노이즈 제거 등
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(processed)

# -----------------------------
# AI 분석 스레드
# -----------------------------
def ai_thread():
    global stop_flag
    while not stop_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # YOLO 추론
            results = model.predict(frame)
            # 결과 렌더링
            annotated_frame = results[0].plot()
            # 큐 또는 직접 표시
            cv2.imshow("YOLO Result", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break

# -----------------------------
# 스레드 시작
# -----------------------------
threads = []

# 캡처 스레드
t_capture = threading.Thread(target=capture_thread)
threads.append(t_capture)

# ISP 처리 필요시
if CAMERA_TYPE == "ISP":
    t_isp = threading.Thread(target=isp_thread)
    threads.append(t_isp)

# AI 분석 스레드
t_ai = threading.Thread(target=ai_thread)
threads.append(t_ai)

# 모든 스레드 시작
for t in threads:
    t.start()

# 모든 스레드 종료 대기
for t in threads:
    t.join()

cv2.destroyAllWindows()
