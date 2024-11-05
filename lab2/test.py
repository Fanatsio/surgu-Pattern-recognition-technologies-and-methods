import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import joblib  # для загрузки модели

# Пример извлечения признаков HOG (как в train.py)
def extract_hog_features(image):
    gray_image = rgb2gray(image)  # Преобразуем изображение в оттенки серого
    features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, feature_vector=True)
    return features

# Загрузка сохраненной модели
svm = joblib.load('svm_cat_classifier.pkl')  # Загружаем ранее обученную модель
print("Модель загружена.")

# Загрузка новых изображений для тестирования
image_paths = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"]
test_images = [cv2.imread(path) for path in image_paths]

# Извлечение признаков из новых изображений
test_features = [extract_hog_features(img) for img in test_images]

# Предсказание с помощью загруженной модели
predictions = svm.predict(test_features)

# Вывод результатов
for i, pred in enumerate(predictions):
    label = "кот" if pred == 1 else "не кот"
    print(f"Изображение {image_paths[i]}: {label}")
