import cv2

def cascade(image):
    # 使用的是HyperLPR已经训练好了的分类器
    print(image.shape)
    watch_cascade = cv2.CascadeClassifier(r'C:\Users\Administrator\Desktop\666\park\parkmanage\model\cascade.xml')
    print(watch_cascade)
    # 先读取图片
    # image = cv2.imread("E://img//8 (3).jpg")

    # resize_h = 1000
    # height = image.shape[0]
    # scale = image.shape[1]/float(image.shape[0])
    # image = cv2.resize(image, (int(scale*resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print("imggrag:",image_gray.shape)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2, 2, minSize=(19, 5), maxSize=(36*40, 9*40))

    #print("检测到车牌数", len(watches))
    #print(watches.shape)
    rect=[]
    for (x, y, w, h) in watches:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        rect.append([x, y, x + w, y + h])

    #cv2.imshow("image", image)
    return rect
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()