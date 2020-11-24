# python detect_mask_image.py --image examples/example_01.png

# 필수 패키지들을 import한다
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# args를 구성한다
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 얼굴 감지 모델을 로드한다
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# 마스크 인식 모델을 로드한다
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# 입력 이미지를 로드한다
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# blobFromImage를 사용하여 frame을 Caffe 모델에서 사용하는 blob으로 변환한다
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# blob을 신경망의 입력값으로 넣어주고 얼굴 인식 결과를 얻는다
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

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
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        	# 244x244로 크기를 조정하고 사전 처리한다
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# 마스크 감지 모델에 넣는다
		(mask, withoutMask) = model.predict(face)[0]

		# 마스크 착용자와 미착용자를 판별하고 분류한다
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# probability를 입력한다
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# 객체에 박스를 씌우고 분류 결과와 퍼센트를 보여준다
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# 결과 이미지를 보여준다
cv2.imshow("Output", image)
cv2.waitKey(0)
