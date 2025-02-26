import cv2
import numpy as np
import subprocess
import shlex
import threading
from queue import Queue
import triangulation as tri
import calibration
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
frame_rate = 5
B = 5.5  # 카메라 간 거리 [cm]
f = 3.51  # 초점 거리 [mm]
alpha = 54  # 카메라 시야각 [도]

def capture_stream(camera_id, buffer_queue):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 960 --height 960 --framerate {frame_rate} -o - --camera {camera_id}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buffer = b""
    try:
        while True:
            buffer += process.stdout.read(4096)
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)  # 프레임 180도 회전
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

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            if buffer_queue0.empty() or buffer_queue1.empty():
                continue

            frame_left = buffer_queue0.get()
            frame_right = buffer_queue1.get()

            frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

            frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            
            results_right = hands.process(frame_right_rgb)
            results_left = hands.process(frame_left_rgb)

            frame_right = cv2.cvtColor(frame_right_rgb, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left_rgb, cv2.COLOR_RGB2BGR)

            wrist_right = wrist_left = None
            
            if results_right.multi_hand_landmarks:
                for hand_landmarks in results_right.multi_hand_landmarks:
                    h, w, _ = frame_right.shape
                    wrist_right = (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h))
                    cv2.rectangle(frame_right, (wrist_right[0] - 5, wrist_right[1] - 5), (wrist_right[0] + 5, wrist_right[1] + 5), (255, 0, 0), 2)

            if results_left.multi_hand_landmarks:
                for hand_landmarks in results_left.multi_hand_landmarks:
                    h, w, _ = frame_left.shape
                    wrist_left = (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h))
                    cv2.rectangle(frame_left, (wrist_left[0] - 5, wrist_left[1] - 5), (wrist_left[0] + 5, wrist_left[1] + 5), (255, 0, 0), 2)

            if wrist_right and wrist_left:
                depth = tri.find_depth(wrist_right, wrist_left, frame_right, frame_left, B, f, alpha)
                cv2.putText(frame_right, f"Depth: {round(depth,1)} cm", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_left, f"Depth: {round(depth,1)} cm", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                print("Depth:", round(depth,1))
            else:
                cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        
            #frame_left_resized = cv2.resize(frame_left, (480, 480))
            #frame_right_resized = cv2.resize(frame_right, (480, 480))
            combined_frame = cv2.hconcat([frame_left, frame_right])

            cv2.imshow("Stereo View", combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
