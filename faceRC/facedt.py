import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from numpy.linalg import norm
from scipy.spatial.distance import cosine



# Bước 1: Tải mô hình VGG16 đã được huấn luyện trước
base_model = VGG16(weights='imagenet')

# Bước 2: Xóa lớp phân loại phía trên cùng để chỉ lấy phần trích xuất đặc trưng
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Hàm phát hiện và cắt khuôn mặt từ ảnh
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Hàm chuẩn bị ảnh để đưa vào VGG16
def preprocess_face(face_img):
    face_img_resized = cv2.resize(face_img, (224, 224))  # VGG16 yêu cầu ảnh 224x224
    face_img_resized = np.expand_dims(face_img_resized, axis=0)  # Thêm batch dimension
    face_img_resized = preprocess_input(face_img_resized)  # Chuẩn hóa cho VGG16
    return face_img_resized

# Hàm trích xuất đặc trưng từ khuôn mặt bằng VGG16
def extract_face_embedding(face_img):
    preprocessed_face = preprocess_face(face_img)
    embedding_vector = model.predict(preprocessed_face)
    return embedding_vector



def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2.T)  # Tính tích vô hướng
    norm_vector1 = norm(vector1)  # Độ dài vector 1
    norm_vector2 = norm(vector2)  # Độ dài vector 2
    similarity = dot_product / (norm_vector1 * norm_vector2)  # Cosine similarity
    return similarity[0][0]  # Lấy giá trị trong mảng


# Main pipeline
image_path1 = r'C:\Users\ATUS\Desktop\Pycharm\faceRC\km1.jpg'
image1 = cv2.imread(image_path1)


image_path2 = r'C:\Users\ATUS\Desktop\Pycharm\faceRC\km1.jpg'
image2 = cv2.imread(image_path2)

# Bước 1: Phát hiện khuôn mặt trong ảnh
faces1 = detect_faces(image1)
faces2 = detect_faces(image2)


# Bước 2: Trích xuất đặc trưng cho mỗi khuôn mặt
# ảnh 1
x1, y1, w1, h1 = faces1[0]
face_img1 = image1[y1:y1+h1, x1:x1+w1]  # Cắt khuôn mặt từ ảnh gốc
cv2.imshow('Detected Face1', face_img1)
vector1 = extract_face_embedding(face_img1)  # Trích xuất vector đặc trưng bằng VGG16
print("Embedding vector:", vector1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ảnh 2
x2, y2, w2, h2 = faces2[0]
face_img2 = image2[y2:y2+h2, x2:x2+w2]  # Cắt khuôn mặt từ ảnh gốc
cv2.imshow('Detected Face2', face_img2)
vector2 = extract_face_embedding(face_img2)  # Trích xuất vector đặc trưng bằng VGG16
print("Embedding vector:", vector2)


similarity = cosine(vector1[0], vector2[0])
print(f"Độ tương tự giữa 2 ảnh: {similarity}") 

threshold = 0.3
if similarity < threshold:
    print("The two faces are likely the same person.")
else:
    print("The two faces are likely different people.")

cv2.waitKey(0)
cv2.destroyAllWindows()