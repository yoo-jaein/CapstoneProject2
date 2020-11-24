# python train_mask_detector.py --dataset dataset

# 필수 패키지들을 import한다
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# args를 구성한다
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

#초기 학습률, epoch, 배치 사이즈에 대한 초기값을 설정해준다
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#imagepaths에 있는 모든 데이터셋을 가져온다 
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	#클래스 라벨 추출
	label = imagePath.split(os.path.sep)[-2]

	# 이미지의 사진을 조절 후 로드하여 전처리 과정을 거친다
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# 이미지와 라벨을 추가해준다
	data.append(image)
	labels.append(label)

# 데이터와 라벨을 numPy 배열로 변환한다
data = np.array(data, dtype="float32")
labels = np.array(labels)

#라벨을 원-핫 인코딩 방식으로 표현해준다
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#scikit-learn을 통해 80퍼센트의 데이터는 훈련, 20퍼센트의 데이터는 테스트 용으로 분류한다
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# data augmentation을 위한 과정
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#MobileNetV2 모델을 로드한다
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#헤드 모델을 설계한다
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#헤드를 원래 모델의 헤드가 있는 위치에 넣어준다
model = Model(inputs=baseModel.input, outputs=headModel)

#베이스 레이어는 고정시켜준다
for layer in baseModel.layers:
	layer.trainable = False

# 컴파일해준다
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 네트워크의 헤드 부분을 훈련시킨다
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# 테스트셋에 대한 예측을 만든다
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

#그 중 가장 높은 확률 클래스 라벨을 표시
predIdxs = np.argmax(predIdxs, axis=1)

# classification report 출력
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# 우리가 생성한 모델을 serialize해준다
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot 과정
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
