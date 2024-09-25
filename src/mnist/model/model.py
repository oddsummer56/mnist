import numpy as np
from PIL import Image
from keras.models import load_model
import os

def loading_model(): 
    file = __file__
    fpath = os.path.dirname(file)
    model_path = os.path.join(fpath, 'mnist240924.keras')
    model = load_model(model_path)  # 학습된 모델 파일 경로
    return model

# 사용자 이미지 불러오기 및 전처리
def preprocess_image(file_path):
    img = Image.open(file_path).convert('L')  # 흑백 이미지로 변환
    img = img.resize((28, 28))  # 크기 조정

    # 흑백 반전
    img = 255 - np.array(img)  # 흑백 반전
    # img = np.array(img)
    
    img = img.reshape(1, 28, 28, 1)  # 모델 입력 형태에 맞게 변형
    img = img / 255.0  # 정규화
    return img

# 예측
def predict_digit(file_path):
    img = preprocess_image(file_path)
    model = loading_model()
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    return digit

