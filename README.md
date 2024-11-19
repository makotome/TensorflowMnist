# 使用Tensorflow 训练生成图片数字识别模型给Unity使用

1. 生成格式为onnx
2. Unity 使用Barracuda包来加载模型
3. 使用Mnist原始数据识别精度不够，自己自己的一些图片来加强训练

## 增加数据的时候，注意标签的问题
1. 将文件夹 "10" 重命名为 "a" 或 "z" 等，确保它在字母顺序中是最后一个
```
custom_dataset = tf.keras.utils.image_dataset_from_directory(
    custom_data_dir,
    image_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    label_mode='int'
)
```

2. 第二种方法是使用 class_names 参数明确指定类别顺序：
```
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
custom_dataset = tf.keras.utils.image_dataset_from_directory(
    custom_data_dir,
    image_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    label_mode='int',
    class_names=class_names
)
```
