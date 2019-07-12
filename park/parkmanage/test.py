import other,cv2
from PIL import Image
import PIL

path = "D:/license_data/" + str(0) + ".bmp"
img = PIL.Image.open(path)
width = img.size[0]
height = img.size[1]
img_data = [[0] * 1280 for i in range(1)]
print("img_data:",type(img_data))
print(len(img_data[0]))
# # 使用的是HyperLPR已经训练好了的分类器
# watch_cascade = cv2.CascadeClassifier('model/cascade.xml')
# print(type(watch_cascade))
#     # 先读取图片
# image = cv2.imread("E://img//8 (3).jpg")
#
#     # resize_h = 1000
#     # height = image.shape[0]
#     # scale = image.shape[1]/float(image.shape[0])
#     # image = cv2.resize(image, (int(scale*resize_h), resize_h))
#     # print("111")
# image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(19, 5), maxSize=(36*40, 9*40))
#
# print("检测到车牌数", len(watches))
# print(watches.shape)
# rect=[]
# for (x, y, w, h) in watches:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
#     rect.append([x, y, x + w, y + h])
#
# cv2.imshow("image", image)
#     # return rect
# cv2.waitKey(0)
# cv2.destroyAllWindows()