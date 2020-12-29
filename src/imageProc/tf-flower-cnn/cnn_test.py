from skimage import io, transform
import glob  # 查找目录和文件模块
import os  # 操作文件夹模块
import tensorflow as tf  # tens框架
import numpy as np  # 数组函数包
import time  # 时间模块

path1 = "E:/flower_photos/dandelion/822355d2f_n.jpg"
path2 = "E:/flower_photos/dandelion/7355d3078_m.jpg"
path3 = "E:/flower_photos/roses/39492cf8d_n.jpg"
path4 = "E:/flower_photos/sunflowers/69536bf4ea3.jpg"
path5 = "E:/flower_photos/tulips/107912168491604.jpg"

flower_dict = {0: 'dasiy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

w = 100
h = 100
c = 3


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('E:/花朵分类/ckpt_dir/.meta')
    saver.restore(sess, tf.train.latest_checkpoint('E:/花朵分类/ckpt_dir/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i + 1, "朵花预测:" + flower_dict[output[i]])
