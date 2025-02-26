import cv2
import numpy as np
import subprocess
import shlex
import threading
from queue import Queue
import datetime
import time

def rotate_image(image, angle):
    # 이미지의 중심을 기준으로 회전 행렬 생성
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 회전 행렬을 사용하여 이미지를 회전시킴
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def capture_stream(camera_id, buffer_queue, image_save_dir):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 960 --height 960 --framerate 20 -o - --camera {camera_id}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buffer = b""
    last_capture_time = 0
    capture_interval = 2  # 이미지 캡처 간격 (초)

    try:
        while True:
            current_time = time.time()

            buffer += process.stdout.read(4096)
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = buffer[a:b+2]
                buffer = buffer[b+2:]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # 스트리밍을 위한 프레임을 큐에 저장
                    buffer_queue.put(frame)

                    # 일정 간격마다 이미지를 저장
                    if current_time - last_capture_time >= capture_interval:
                        now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
                        print(f"Camera {camera_id} - take picture: {now}")
                        cv2.imwrite(f"{image_save_dir}/{now}.jpg", rotate_image(frame, 180))  # 저장
                        last_capture_time = current_time

    finally:
        process.terminate()

def show_stream(buffer_queue0, buffer_queue1):
    while True:
        if not buffer_queue0.empty() and not buffer_queue1.empty():
            frame0 = buffer_queue0.get()
            frame1 = buffer_queue1.get()

            # 두 이미지를 수평으로 결합
            combined_frame = np.hstack((frame1, frame0))

            cv2.imshow('Stereo Camera Stream', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    buffer_queue0 = Queue(maxsize=5)
    buffer_queue1 = Queue(maxsize=5)

    # 저장 디렉토리 설정
    image_save_dir0 = "../data/image_0"
    image_save_dir1 = "../data/image_1"

    # 멀티스레딩을 이용해 두 카메라의 스트리밍을 동시에 처리
    thread0 = threading.Thread(target=capture_stream, args=(0, buffer_queue0, image_save_dir0), daemon=True)
    thread1 = threading.Thread(target=capture_stream, args=(1, buffer_queue1, image_save_dir1), daemon=True)

    thread0.start()
    thread1.start()

    # 화면에 스트리밍을 동기화하여 출력
    show_stream(buffer_queue0, buffer_queue1)

    thread0.join()
    thread1.join()
