import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

SIZE = 160
model = load_model('note\my_model.h5')
def colorize_image(input_image):
    # Đảm bảo ảnh là định dạng RGB
    img_rgb = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)

    original_shape = img_rgb.shape[:2]
    # Resize ảnh về kích thước phù hợp cho mô hình
    img_resized = cv2.resize(img_rgb, (160, 160))

    # Tiền xử lý ảnh cho đầu vào của mô hình
    img_input = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    img_input = np.reshape(img_input ,(len(img_input),160,160,3))
    # Dự đoán và tô màu ảnh
    predicted_image = np.clip(model.predict(img_input.reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)

    # Chuẩn hóa lại ảnh kết quả
    #img_colorized = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    img_colorized_resized = cv2.resize(predicted_image, (original_shape[1], original_shape[0]))
    return img_colorized_resized

def main():
    st.title("Tô màu ảnh")

    

    uploaded_file = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1.image(image, caption='Ảnh gốc', use_column_width=True)

        colorized_image = colorize_image(image)
        col2.image(colorized_image, caption='Ảnh đã được tô màu', use_column_width=True)
     
if __name__ == "__main__":
    main()