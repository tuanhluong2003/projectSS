from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np

# Tải mô hình VGGFace với kiến trúc VGG16
vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Đọc ảnh đầu vào và chuyển đổi thành kích thước cần thiết (224x224)
img_path = 'path_to_your_image.jpg'  # Đường dẫn tới ảnh bạn muốn trích xuất đặc trưng
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Tiền xử lý ảnh theo chuẩn của VGGFace
img_array = preprocess_input(img_array, version=1)  # Version 1 cho VGGFace VGG16

# Trích xuất đặc trưng từ ảnh đầu vào
features = vggface_model.predict(img_array)

# In ra đặc trưng đã được trích xuất
print("Features shape:", features.shape)
print("Extracted features:", features)
