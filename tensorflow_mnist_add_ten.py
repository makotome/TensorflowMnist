import tensorflow as tf
import numpy as np
import tf2onnx
import os
from datetime import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pathlib import Path

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

def create_data_augmentation():
    """创建数据增强层"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])

def augment_class_10(images, labels, target_size, augmentation_layer):
    """
    专门对数字10的样本进行数据增强
    """
    # 找出所有数字10的样本
    mask_10 = (labels == 10)
    images_10 = images[mask_10]
    labels_10 = labels[mask_10]
    
    # 显示前 5 张图像
    # plt.figure(figsize=(10, 2))
    # for i in range(min(5, len(images_10))):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(images_10[i].reshape(28, 28), cmap='gray')
    #     plt.title(f'Label: {labels_10[i]}')
    #     plt.axis('off')
    # plt.show()

    # 计算需要增强的数量
    current_size = len(images_10)
    augment_times = int(np.ceil(target_size / current_size))
    
    print(f"当前数字10的样本数量: {current_size}")
    print(f"目标数量: {target_size}")
    print(f"需要增强的倍数: {augment_times}")
    
    # 存储增强后的图片和标签
    augmented_images = [images_10]
    augmented_labels = [labels_10]
    
    # 进行数据增强
    for i in range(augment_times - 1):
        # 确保输入张量的形状正确
        augmented = augmentation_layer(tf.convert_to_tensor(images_10), training=True)
        augmented_images.append(augmented.numpy())
        augmented_labels.append(labels_10)
    
    # 合并所有增强后的数据
    augmented_images = np.concatenate(augmented_images, axis=0)
    augmented_labels = np.concatenate(augmented_labels, axis=0)
    
    # 如果生成的数量超过目标数量，随机截取所需数量
    if len(augmented_images) > target_size:
        indices = np.random.choice(len(augmented_images), target_size, replace=False)
        augmented_images = augmented_images[indices]
        augmented_labels = augmented_labels[indices]
    
    return augmented_images, augmented_labels

def prepare_balanced_dataset(x_train, y_train, custom_images, custom_labels):
    """准备平衡的数据集"""
    # 合并MNIST数据和自定义数据
    all_images = np.concatenate((x_train, custom_images), axis=0)
    all_labels = np.concatenate((y_train, custom_labels), axis=0)
    
    # 计算各个类别的样本数量
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print("\n各个类别的原始样本数量：")
    for label, count in zip(unique_labels, counts):
        print(f"数字 {label}: {count} 张图片")
    
    # 计算除了10以外其他类别的平均样本数量
    mask_not_10 = (unique_labels != 10)
    avg_count = int(np.mean(counts[mask_not_10]))
    print(f"\n其他类别的平均样本数量: {avg_count}")
    
    # 创建数据增强层
    data_augmentation = create_data_augmentation()
    
    # 对数字10进行数据增强
    augmented_10_images, augmented_10_labels = augment_class_10(
        all_images, all_labels, avg_count, data_augmentation
    )
    
    # 将原始数据中的10替换为增强后的数据
    mask_not_10 = (all_labels != 10)
    final_images = np.concatenate([all_images[mask_not_10], augmented_10_images])
    final_labels = np.concatenate([all_labels[mask_not_10], augmented_10_labels])
    
    # 随机打乱数据
    final_images, final_labels = shuffle(final_images, final_labels, random_state=42)
    
    return final_images, final_labels

def create_model():
    """创建改进的CNN模型"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(11, activation='softmax')  # 11个类别（0-10）
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_augmented_samples(images, labels, num_samples=5):
    """可视化增强后的样本"""
    mask_10 = (labels == 10)
    images_10 = images[mask_10][:num_samples]
    
    # plt.figure(figsize=(15, 3))
    # for i in range(num_samples):
    #     plt.subplot(1, num_samples, i + 1)
    #     plt.imshow(images_10[i].reshape(28, 28), cmap='gray')
    #     plt.axis('off')
    #     plt.title(f'Sample {i+1}')
    # plt.show()

def evaluate_model(model, test_images, test_labels):
    """详细评估模型性能"""
    # 预测
    predictions = model.predict(test_images)
    
    # 每个类别的准确率
    for i in range(11):  # 0-10
        mask = test_labels == i
        if np.any(mask):
            class_acc = np.mean(np.argmax(predictions[mask], axis=1) == test_labels[mask])
            print(f"类别 {i} 的准确率: {class_acc:.4f}")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, np.argmax(predictions, axis=1))
    print("\n混淆矩阵:")
    print(cm)

def evaluate_model1(model,normalized_valid_ds):
    # 评估模型并打印准确率
    loss, accuracy = model.evaluate(normalized_valid_ds, verbose=2)
    print(f"Test accuracy: {accuracy}")

    # 获取预测结果
    predictions = model.predict(normalized_valid_ds)

    # 将预测结果转换为标签
    predicted_labels = np.argmax(predictions, axis=1)

    # 获取实际标签
    actual_labels = np.concatenate([y for x, y in normalized_valid_ds], axis=0)

    # 找出预测错误的样本
    incorrect_indices = np.where(predicted_labels != actual_labels)[0]

    # 打印预测错误的标签
    print("Incorrectly predicted labels:")
    for i in incorrect_indices:
        print(f"Index: {i}, Predicted: {predicted_labels[i]}, Actual: {actual_labels[i]}")
    
def main():
    # 创建必要的目录
    Path("./onnx").mkdir(parents=True, exist_ok=True)
    
    # 下载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 归一化数据
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # 调整数据形状
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # 加载自定义数据（数字10）
    custom_data_dir = 'data/train'
    custom_dataset = tf.keras.utils.image_dataset_from_directory(
        custom_data_dir,
        image_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        label_mode='int',
        class_names=class_names

    )
    
    # 准备验证集
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        custom_data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(28, 28),
        batch_size=32,
        label_mode='int',
        class_names=class_names
    )
    
    # 转换自定义数据集为numpy数组
    custom_images = []
    custom_labels = []
    for images, labels in custom_dataset:
        custom_images.append(images.numpy())
        custom_labels.append(labels.numpy())
    
    custom_images = np.concatenate(custom_images, axis=0)
    custom_labels = np.concatenate(custom_labels, axis=0)
    
    # 归一化自定义数据
    custom_images = custom_images / 255.0
    
    # 准备平衡的数据集
    balanced_images, balanced_labels = prepare_balanced_dataset(
        x_train, y_train, custom_images, custom_labels
    )
    
    # 创建和编译模型
    model = create_model()
    
    # 添加回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # 训练模型
    history = model.fit(
        balanced_images,
        balanced_labels,
        epochs=10,
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    evaluate_model(model, balanced_images, balanced_labels)

    # 评估模型
    loss, accuracy = model.evaluate(valid_ds, verbose=2)
    print(f"\nValidation accuracy: {accuracy}")

    # evaluate_model1
    
    # 可视化一些增强后的样本
    # visualize_augmented_samples(balanced_images, balanced_labels)
    
    # 保存为ONNX模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    output_path = f"./onnx/mnist_model_{timestamp}.onnx"
    
    model.output_names = ["output"]
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"\nONNX model saved to {output_path}")


if __name__ == "__main__":
    main()