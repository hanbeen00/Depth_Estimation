# Depth_Estimation for Raspberry Pi5 using ov5647 

##### 실행하기전 python 가상환경 활성화

------------

#### Step1. 카메라 2대를 이용해서 카메라간 거리가 고정된 상태에서 동시에 사진촬영 진행
  - 실행: python take_picture.py
  - data/image_0, data/image_1에 각각의 카메라 촬영 사진이 저장됨
  - 해상도: 960x960, 캘리브레이션을 위해 각각 20장 이상의 사진 촬영 권장 

------------

#### Step2. 각각의 카메라 캘리브레이션 및 스테레오 캘리브레이션 진행
  - 실행: python stereo_calibration.py
  - data 디렉터리에 캘리브레이션 결과값인 stereoMap.xml이 생성

------------

#### Step3. 캘리브레이션 값 및 삼각측량을 이용해 카메라에서 손까지의 거리를 추정(Depth Estimation)
  - 실행: python depth_estimation.py
  - 실제 카메라 렌즈간 거리(5.5cm, 임의로 설정), 초점거리(3.51mm, 단일 캘리브레이션을 통해 얻은 값), 시야각(54도, 실제 데이터시트 값) 정보 필
