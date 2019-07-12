import tensorflow
import numpy as np
# import tensorflow as tf
import PIL.Image


SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 200

DIGITS_SAVER_DIR = r"C:\\Users\Administrator\Desktop\666\park\parkmanage\parkmanage1\train-saver\digits"

LETTERS_DIGITS = (
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
        "N", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

tensorflow.reset_default_graph()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tensorflow.placeholder(tensorflow.float32, shape=[None, SIZE])
y_ = tensorflow.placeholder(tensorflow.float32, shape=[None, NUM_CLASSES])

x_image = tensorflow.reshape(x, [-1, WIDTH, HEIGHT, 1])

# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tensorflow.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tensorflow.nn.relu(L1_conv + b)
    return tensorflow.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')

# 定义全连接层函数
def full_connect(inputs, W, b):
    return tensorflow.nn.relu(tensorflow.matmul(inputs, W) + b)


def predict_digits():

    license_num = ""
    saver = tensorflow.train.import_meta_graph(r"C:\\Users\Administrator\Desktop\666\park\parkmanage\parkmanage1\train-saver\digits\model.ckpt.meta")
    with tensorflow.Session() as sess:
        model_file = tensorflow.train.latest_checkpoint(DIGITS_SAVER_DIR)
        saver.restore(sess, model_file)
        # 第一个卷积层
        W_conv1_digits = sess.graph.get_tensor_by_name("W_conv1_digits:0")
        b_conv1_digits = sess.graph.get_tensor_by_name("b_conv1_digits:0")
        conv_strides_digits = [1, 1, 1, 1]
        kernel_size_digits = [1, 2, 2, 1]
        pool_strides_digits = [1, 2, 2, 1]
        L1_pool_digits = conv_layer(x_image, W_conv1_digits, b_conv1_digits, conv_strides_digits, kernel_size_digits, pool_strides_digits, padding='SAME')

        # 第二个卷积层
        W_conv2_digits = sess.graph.get_tensor_by_name("W_conv2_digits:0")
        b_conv2_digits = sess.graph.get_tensor_by_name("b_conv2_digits:0")
        conv_strides_digits = [1, 1, 1, 1]
        kernel_size_digits = [1, 1, 1, 1]
        pool_strides_digits = [1, 1, 1, 1]
        L2_pool_digits = conv_layer(L1_pool_digits, W_conv2_digits, b_conv2_digits, conv_strides_digits, kernel_size_digits, pool_strides_digits, padding='SAME')

        # 全连接层
        W_fc1_digits = sess.graph.get_tensor_by_name("W_fc1_digits:0")
        b_fc1_digits = sess.graph.get_tensor_by_name("b_fc1_digits:0")
        h_pool2_flat_digits = tensorflow.reshape(L2_pool_digits, [-1, 16 * 20 * 32])
        h_fc1_digits = full_connect(h_pool2_flat_digits, W_fc1_digits, b_fc1_digits)

        # dropout
        keep_prob_digits = tensorflow.placeholder(tensorflow.float32)

        h_fc1_drop_digits = tensorflow.nn.dropout(h_fc1_digits, keep_prob_digits)

        # readout层
        W_fc2_digits = sess.graph.get_tensor_by_name("W_fc2_digits:0")
        b_fc2_digits = sess.graph.get_tensor_by_name("b_fc2_digits:0")

        # 定义优化器和训练op
        conv_digits = tensorflow.nn.softmax(tensorflow.matmul(h_fc1_drop_digits, W_fc2_digits) + b_fc2_digits)

        for n in range(1, 7):
            path = "D:/license_data/" + str(n) + ".bmp"
            img = PIL.Image.open(path)
            width = img.size[0]
            height = img.size[1]
            img_data = [[0] * SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0
            result = sess.run(conv_digits, feed_dict={x: np.array(img_data), keep_prob_digits: 1.0})
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

            license_num = license_num + LETTERS_DIGITS[max1_index]
        #     print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
        #         LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100,
        #         LETTERS_DIGITS[max3_index],
        #         max3 * 100))
        #print("车牌编号是: 【%s】" % license_num)
        f2 = open(r"C:\\Users\Administrator\Desktop\666\park\parkmanage\out.txt", 'a')
        f2.write(license_num)
        f2.close()
        tensorflow.Graph().as_default()
        return license_num


