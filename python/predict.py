import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# 加载训练好的模型
model = tf.keras.models.load_model('my_model.h5')  # 替换为你保存的模型文件路径

# 加载图像并进行预处理
img_path1 = 'test/test1.png' 
img_path2 = 'test/test2.png' 
img_path3 = 'test/test3.png'  # 替换为你要分类的图像路径
img1 = image.load_img(img_path1, target_size=(224, 224))
img2 = image.load_img(img_path2, target_size=(224, 224))
img3 = image.load_img(img_path3, target_size=(224, 224))

x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)

x2 = image.img_to_array(img2)
x2 = np.expand_dims(x2, axis=0)
x2 = preprocess_input(x2)

x3 = image.img_to_array(img3)
x3 = np.expand_dims(x3, axis=0)
x3 = preprocess_input(x3)
# 使用模型进行预测
preds1 = model.predict(x1)
preds2 = model.predict(x2)
preds3 = model.predict(x3)
predicted_class1 = np.argmax(preds1, axis=1)[0]
predicted_class2 = np.argmax(preds2, axis=1)[0]
predicted_class3 = np.argmax(preds3, axis=1)[0]

# 打印预测结果
print(f'Predicted class: {predicted_class1}')
print(f'Predicted class: {predicted_class2}')
print(f'Predicted class: {predicted_class3}')
3