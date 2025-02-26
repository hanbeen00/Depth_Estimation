import cv2
import numpy as np
import subprocess
import shlex
import threading
from queue import Queue
import os
from datetime import datetime

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def capture_stream(camera_id, buffer_queue):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 960 --height 960 --framerate 30 -o - --camera {camera_id}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buffer = b""
    try:
        while True:
            buffer += process.stdout.read(4096)
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = buffer[a:b + 2]
                buffer = buffer[b + 2:]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    buffer_queue.put(frame)
    finally:
        process.terminate()

def save_image(camera_id, frame):
    if camera_id == 0:
        directory = '../data/image_0/'
    else:
        directory = '../data/image_1/'
    
    # 디렉토리가 없으면 생성
    os.makedirs(directory, exist_ok=True)

    # 현재 시간을 기준으로 파일 이름 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(directory, f'{timestamp}.jpg')
    
    # 이미지를 180도 회전
    rotated_frame = rotate_image(frame, 180)

    # 이미지 저장
    cv2.imwrite(filename, rotated_frame)
    print(f'Saved image: {filename}')

def show_stream(buffer_queue1, buffer_queue2):
    while True:
        if not buffer_queue1.empty():
            frame1 = buffer_queue1.get()
            rotated_frame1 = rotate_image(frame1, 180)
            cv2.imshow('Camera 0 Stream', rotated_frame1)

        if not buffer_queue2.empty():
            frame2 = buffer_queue2.get()
            rotated_frame2 = rotate_image(frame2, 180)
            cv2.imshow('Camera 1 Stream', rotated_frame2)

        # 사진을 찍기 위한 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not buffer_queue1.empty():
                frame1 = buffer_queue1.get()
                save_image(0, frame1)
            if not buffer_queue2.empty():
                frame2 = buffer_queue2.get()
                save_image(1, frame2)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    buffer_queue1 = Queue()
    buffer_queue2 = Queue()

    # 멀티스레딩을 이용해 두 카메라의 스트리밍을 동시에 처리
    thread1 = threading.Thread(target=capture_stream, args=(0, buffer_queue1))
    thread2 = threading.Thread(target=capture_stream, args=(1, buffer_queue2))

    thread1.start()
    thread2.start()

    # 화면에 스트리밍을 동기화하여 출력
    show_stream(buffer_queue1, buffer_queue2)

    thread1.join()
    thread2.join()
