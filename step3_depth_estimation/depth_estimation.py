import cv2
import numpy as np
import subprocess
import shlex
import threading
from queue import Queue, Empty
import triangulation as tri
import calibration
import mediapipe as mp
import time

mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
frame_rate = 5  # FPS 증가
B = 5.5  # 카메라 간 거리 [cm]
f = 3.51  # 초점 거리 [mm]
alpha = 54  # 카메라 시야각 [도]

def capture_stream(camera_id, buffer_queue):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --flush --width 960 --height 960 --framerate {frame_rate} -o - --camera {camera_id}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=4096)

    buffer = b""
    try:
        while process.poll() is None:
            buffer += process.stdout.read(4096)
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if not buffer_queue.full():
                        buffer_queue.put(frame)
    finally:
        process.terminate()

if __name__ == "__main__":
    buffer_queue0 = Queue(maxsize=5)
    buffer_queue1 = Queue(maxsize=5)

    thread0 = threading.Thread(target=capture_stream, args=(0, buffer_queue0), daemon=True)
    thread1 = threading.Thread(target=capture_stream, args=(1, buffer_queue1), daemon=True)
    thread0.start()
    thread1.start()

    while True:
        try:
            frame_left = buffer_queue0.get_nowait()
            frame_right = buffer_queue1.get_nowait()
        except Empty:
            continue

        # 스테레오 이미지 보정
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        # RGB 변환
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Hand tracking
        results_right = mp_hands.process(frame_right_rgb)
        results_left = mp_hands.process(frame_left_rgb)

        wrist_right = wrist_left = None

        if results_right.multi_hand_landmarks:
            h, w, _ = frame_right.shape
            wrist_right = np.array([int(results_right.multi_hand_landmarks[0].landmark[0].x * w),
                                    int(results_right.multi_hand_landmarks[0].landmark[0].y * h)])

            cv2.rectangle(frame_right, (wrist_right[0] - 5, wrist_right[1] - 5),
                          (wrist_right[0] + 5, wrist_right[1] + 5), (255, 0, 0), 2)

        if results_left.multi_hand_landmarks:
            h, w, _ = frame_left.shape
            wrist_left = np.array([int(results_left.multi_hand_landmarks[0].landmark[0].x * w),
                                   int(results_left.multi_hand_landmarks[0].landmark[0].y * h)])

            cv2.rectangle(frame_left, (wrist_left[0] - 5, wrist_left[1] - 5),
                          (wrist_left[0] + 5, wrist_left[1] + 5), (255, 0, 0), 2)

        if wrist_right is not None and wrist_left is not None:
            depth = tri.find_depth(wrist_right, wrist_left, frame_right, frame_left, B, f, alpha)
            cv2.putText(frame_right, f"Depth: {round(depth, 1)} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame_left, f"Depth: {round(depth, 1)} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            print("Depth:", round(depth, 1))
        else:
            cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 프레임 합치기 및 표시
        combined_frame = cv2.hconcat([frame_left, frame_right])
        cv2.imshow("Stereo View", cv2.resize(combined_frame, (960, 480), interpolation=cv2.INTER_LINEAR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
