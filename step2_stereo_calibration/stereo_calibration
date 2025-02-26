import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8,6)
frameSize = (960,960)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
square_size = 31  # 체스보드 한 칸의 크기 (예: 31mm)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points in real-world space
imgpointsL = [] # 2D points in image plane (Left Camera)
imgpointsR = [] # 2D points in image plane (Right Camera)

imagesLeft = glob.glob('../data/image_0/*.jpg')
imagesRight = glob.glob('../data/image_1/*.jpg')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points and image points (after refining them)
    if retL and retR:
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        #cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        #cv.imshow('img left', imgL)
        #cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        #cv.imshow('img right', imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()

############## CAMERA CALIBRATION #######################################################

# Left Camera Calibration
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

# Right Camera Calibration
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# 출력 (단일 카메라 RMS 오차)
print(f"Left Camera Reprojection Error (RMS): {retL:.6f}")
print(f"Right Camera Reprojection Error (RMS): {retR:.6f}")

########## STEREO CALIBRATION #####################################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC  # 내부 카메라 파라미터 고정

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 스테레오 캘리브레이션 (Essential & Fundamental Matrix 포함)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

# 출력 (스테레오 RMS 오차)
print(f"Stereo Calibration Reprojection Error (RMS): {retStereo:.6f}")

########## STEREO RECTIFICATION #################################################

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
    newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

# 파라미터 저장
print("Saving parameters!")
cv_file = cv.FileStorage('../data/stereoMap.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])
cv_file.release()
