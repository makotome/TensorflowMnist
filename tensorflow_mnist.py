import tensorflow as tf
import numpy as np
import tf2onnx
import os
from datetime import datetime


# 下载 MNIST 数据集 默认下载到 ~/.keras/datasets/mnist.npz
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化数据
x_train, x_test = x_train / 255.0, x_test / 255.0

# 调整数据形状以适应卷积层
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 加载自定义图片数据 增加自己的训练数据
custom_data_dir = 'data/train'
custom_dataset = tf.keras.utils.image_dataset_from_directory(
    custom_data_dir,
    image_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    label_mode='int'
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
  custom_data_dir,
  color_mode='grayscale',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(28, 28),
  batch_size=10)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_valid_ds = valid_ds.map(lambda x, y: (normalization_layer(x), y))

# # 将自定义图片数据转换为 NumPy 数组
custom_images = []
custom_labels = []
for images, labels in custom_dataset:
    custom_images.append(images.numpy())
    custom_labels.append(labels.numpy())

custom_images = np.concatenate(custom_images, axis=0)
custom_labels = np.concatenate(custom_labels, axis=0)

# # 归一化自定义图片数据
custom_images = custom_images / 255.0

# # 调整自定义图片数据形状以适应卷积层
custom_images = custom_images.reshape(-1, 28, 28, 1)

# # 合并自定义图片数据与 MNIST 数据集
x_train = np.concatenate((x_train, custom_images), axis=0)
y_train = np.concatenate((y_train, custom_labels), axis=0)

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=15, verbose=1)

# 评估模型
loss, accuracy = model.evaluate(normalized_valid_ds, verbose=2)
print(f"Test accuracy: {accuracy}")

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 将模型转换为 ONNX 格式
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
output_path = f"./onnx/mnist_model_{timestamp}.onnx"

# 指定输出节点名称这个是必须的，为了不报错
model.output_names = ["output"]

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"ONNX 模型已保存到 {output_path}")
