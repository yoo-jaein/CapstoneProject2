#!/usr/bin/env python

# python tftest.py
# 마스크 인식 최종 코드

# 필수 패키지들을 import한다
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import rospy

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# python 3 import
import cv2
from cv_bridge import CvBridge, CvBridgeError

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import numpy as np
import tensorflow as tf
import argparse
import imutils
import time
import os

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

def detect_and_predict_mask(frame, faceNet, maskNet):
	# blobFromImage를 사용하여 frame을 Caffe 모델에서 사용하는 blob으로 변환한다
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# blob을 신경망의 입력값으로 넣어주고 얼굴 인식 결과를 얻는다
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# 얼굴, 얼굴의 위치, 예측도 배열을 초기화한다
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# i번째 detections의 정확도를 confidence에 넣는다
		confidence = detections[0, 0, i, 2]

		# confidence가 최소 신뢰도보다 클 때
		if confidence > args["confidence"]:
			# 객체를 둘러싸는 박스의 x, y 좌표 값을 계산한다
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 그 박스가 프레임 내부에 존재하는지 범위를 체크한다
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# 얼굴 관심영역(ROI)를 추출하고 BGR에서 RGB로 변환한다
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			
			# 244x244로 크기를 조정하고 사전 처리한다
			face = img_to_array(face)
			face = preprocess_input(face)

			# 얼굴 관심영역과 박스를 리스트에 추가한다
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# 얼굴이 하나 이상 감지될 때
	if len(faces) > 0:
		# 전체에 대해 일괄적으로 예측한다
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# 얼굴의 위치와 정확도 튜플을 반환한다
	return (locs, preds)

def image_callback(ros_image_compressed):
	try:
		np_arr = np.fromstring(ros_image_compressed.data, np.uint8)
		frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	except e:
		print("Error")

	# 받아온 frame을 리사이즈한다
	frame = imutils.resize(frame, width=400)
	frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

	# frame의 얼굴을 감지하고 마스크를 끼고있는지 판별한다
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# 마스크 착용자와 미착용자를 판별하고 분류한다
		if mask > withoutMask:
			label = "Mask"
			pub = rospy.Publisher("alert", String, queue_size=10)
			pub_str = "1"
			pub.publish(pub_str)#rospy.sleep(5)
		else:
			label = "No Mask"
			pub = rospy.Publisher("alert", String, queue_size=10)
			pub_str = "0"
			pub.publish(pub_str)

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# probability를 입력한다
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# 객체에 박스를 씌우고 분류 결과와 퍼센트를 보여준다
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# frame을 보여준다
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

# args를 구성한다
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 감지 모델 faceNet을 로드한다
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 마스크 인식 모델 maskNet을 로드한다
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

rospy.init_node('image_listener')

image_topic = "/raspicam_node/image/compressed"
rospy.Subscriber(image_topic, CompressedImage, image_callback)

print("Sub start")
rospy.spin()
