
import numpy as np
import tensorflow as tf
import PIL.Image


SIZE = 1280
WIDTH = 32
NUM_CLASSES = 6
HEIGHT = 40
iterations = 200

PROVINCES_SAVER_DIR = r"C:\\Users\Administrator\Desktop\666\park\parkmanage\parkmanage1\train-saver\province"

PROVINCES = ("京", "闽", "粤", "苏", "沪", "浙")
nProvinceIndex = 0

#默认图
tf.reset_default_graph()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


# saver = tf.train.import_meta_graph(
#     "C:\\Users\Administrator\Desktop\\666\park\parkmanage\parkmanage1\\train-saver\province\model.ckpt.meta")


def predict_province():
    saver = tf.train.import_meta_graph("C:\\Users\Administrator\Desktop\\666\park\parkmanage\parkmanage1\\train-saver\province\model.ckpt.meta")
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(PROVINCES_SAVER_DIR)
        saver.restore(sess, model_file)

        # 第一个卷积层
        W_conv1_province = sess.graph.get_tensor_by_name("W_conv1_province:0")
        b_conv1_province = sess.graph.get_tensor_by_name("b_conv1_province:0")
        conv_strides_province = [1, 1, 1, 1]
        kernel_size_province = [1, 2, 2, 1]
        pool_strides_province = [1, 2, 2, 1]
        L1_pool_province = conv_layer(x_image, W_conv1_province, b_conv1_province,
                                      conv_strides_province, kernel_size_province, pool_strides_province, padding='SAME')

        # 第二个卷积层
        W_conv2_province = sess.graph.get_tensor_by_name("W_conv2_province:0")
        b_conv2_province = sess.graph.get_tensor_by_name("b_conv2_province:0")
        conv_strides_province = [1, 1, 1, 1]
        kernel_size_province = [1, 1, 1, 1]
        pool_strides_province = [1, 1, 1, 1]
        L2_pool_province = conv_layer(L1_pool_province, W_conv2_province, b_conv2_province, conv_strides_province, kernel_size_province, pool_strides_province, padding='SAME')

        # 全连接层
        W_fc1_province = sess.graph.get_tensor_by_name("W_fc1_province:0")
        b_fc1_province = sess.graph.get_tensor_by_name("b_fc1_province:0")
        h_pool2_flat_province = tf.reshape(L2_pool_province, [-1, 16 * 20 * 32])
        h_fc1_province = full_connect(h_pool2_flat_province, W_fc1_province, b_fc1_province)

        # dropout
        keep_prob_province = tf.placeholder(tf.float32)

        h_fc1_drop_province = tf.nn.dropout(h_fc1_province, keep_prob_province)

        # readout层
        W_fc2_province = sess.graph.get_tensor_by_name("W_fc2_province:0")
        b_fc2_province = sess.graph.get_tensor_by_name("b_fc2_province:0")

        # 定义优化器和训练op
        conv_province = tf.nn.softmax(tf.matmul(h_fc1_drop_province, W_fc2_province) + b_fc2_province)

        for n in range(1):
            path = "D:/license_data/" + str(n) + ".bmp"
            img = PIL.Image.open(path)
            width = img.size[0]
            height = img.size[1]
            img_data = [[0] * SIZE for i in range(1)]
            # print("img_data:",type(img_data))
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0

            result = sess.run(conv_province, feed_dict={x: np.array(img_data), keep_prob_province: 1.0})
            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j] > max2) and (result[0][j] <= max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j] > max3) and (result[0][j] <= max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue
            nProvinceIndex = max1_index
    #         print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
    #             PROVINCES[max1_index], max1 * 100, PROVINCES[max2_index], max2 * 100, PROVINCES[max3_index],
    #             max3 * 100))
    # print("省份简称是: %s" % PROVINCES[nProvinceIndex])
    f2 = open(r"C:\\Users\Administrator\Desktop\666\park\parkmanage\out.txt", 'w', encoding="utf-8")
    f2.write(PROVINCES[nProvinceIndex])
    f2.close()
    tf.Graph().as_default()
    return PROVINCES[nProvinceIndex]




