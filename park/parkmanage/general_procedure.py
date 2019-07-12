import positioning

positioning.get_img_data(r"C:\Users\Administrator\Desktop\666\park\static\img\temp.jpg")
import tensorflow as tf
from license_province import predict_province
predict_province()
from license_digits import predict_digits
predict_digits()
tf.Graph().as_default()
#
# tf.reset_default_graph()


def output_license():
    f2 = open("out.txt", 'r', encoding="utf-8")
    license_data = f2.read()
    f2.close()
    print(license_data)
    return license_data

output_license()
# positioning.get_img_data(r"C:\Users\Administrator\Desktop\666\park\static\img\temp1.jpg")
# # from license_province import predict_province
# predict_province()
# # from license_digits import predict_digits
# predict_digits()
# output_license()