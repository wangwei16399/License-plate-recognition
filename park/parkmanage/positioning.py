import cv2
import numpy as np
import separate
import other



def stretch(img):
    max = float(img.max())
    min = float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max - min)) * img[i, j] - (255 * min) / (max - min)
    return img

def dobinaryzation(img):
    max = float(img.max())
    min = float(img.min())
    x = max - ((max - min) / 2)#确定分割的阈值
    ret, threshedimg = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    #print("ret:",ret)
    #print(threshedimg.shape)
    return threshedimg


def find_retangle(contour):
    y, x = [], []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]

def choose_area(contours):
    area_list=[]
    for i in range(len(contours)):
        r=find_retangle(contours[i])
        #print("r",r)
        width=r[2]-r[0]
        height=r[3]-r[1]
        area=width*height
        aspect_ratio=width/height
        if aspect_ratio>2.0 and aspect_ratio<5:#长宽比为;2——5
            area_list.append([r,area])
        else:
            pass
    return area_list

def find_maxArea(list):
    for i in range(len(list)):
        for j in range(i,len(list)):
            if list[i][1]<list[j][1]:
                temp=list[i][1]
                list[i][1]=list[j][1]
                list[j][1]=temp
    return list[-3:]

def white_counts(img, area):
    counts = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                counts += 1

    return counts / area

def is_color(list,img):
    MIN_AREA = 1000
    rectangular = []
    for i in range(len(list)):
        r = list[i][0]  # 坐标
        s = list[i][1]  # 面积
        # print("r", r)
        # print("s", s)
        if s>MIN_AREA:
            image = img[r[1]:r[3], r[0]:r[2]]
            #print('image',image.shape)
            #cv2.imshow("image", image)

            # # RGB转HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #print("hsv.shape", hsv.shape)
            # print('hsv',hsv)
            # 蓝色车牌范围
            blue_lower = np.array([100, 43, 46])
            blue_upper = np.array([124, 255, 255])
            # 绿色范围
            green_lower = np.array([35, 43, 46])
            green_upper = np.array([77, 255, 255])
            # 根据阈值构建掩模
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            # print(mask.shape)
            # cv2.imshow("mask",mask)
            # cv2.waitKey()
            if white_counts(mask, s) > 0.7:
                #print("blue")
                rectangular.append([r, "blue"])
                return rectangular
            else:
                mask = cv2.inRange(hsv, green_lower, green_upper)
                if white_counts(mask, s) > 0.7:
                    #print("green")
                    rectangular.append([r, "green"])
                    return rectangular
                else:
                    pass
        else:
            pass

def xiangjiao(rect1,rect2):
    M_x=max(rect1[1],rect2[1])
    M_y=max(rect1[0],rect2[0])
    N_x=min(rect1[3],rect2[3])
    N_y=min(rect1[2],rect2[2])
    if M_x<N_x and M_y<N_y:
        s=(N_x-M_x)*(N_y-M_y)
        s1=(rect1[2]-rect1[0])*(rect1[3]-rect1[1])
        s2=(rect2[2]-rect2[0])*(rect2[3]-rect1[1])
        s_min=min(s1,s2)
        if s/s_min >0.7:

            return [M_y,M_x,N_y,N_x]
        else:
            x_min=min(rect1[1],rect2[1])
            y_min=min(rect1[0],rect2[0])
            x_max=max(rect1[3],rect2[3])
            y_max=max(rect1[2],rect2[2])
            return [y_min,x_min,y_max,x_max]
    else:
        return rect1



# def extract_license_plate(img):
#
#     # 压缩图像
#     img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))
#     # cv2.imshow('resize', img)
#     # RGB转灰色
#     grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # guassimg=cv2.GaussianBlur(src=grayimg,ksize=(3,3),sigmaX=0,sigmaY=0)
#     # cv2.imshow("guass",guassimg)
#     # middleImg=cv2.medianBlur(src=grayimg,ksize=3)
#     # cv2.imshow("middle",middleImg)
#     # 灰度拉伸
#     stretchedimg = stretch(grayimg)
#     # cv2.imshow('stretchedimg', stretchedimg)
#     # 进行开运算，用来去噪声
#     r = 16
#     h = w = r * 2 + 1
#     kernel = np.zeros((h, w), dtype=np.uint8)  # 定义卷积核 33*33 全0
#     cv2.circle(kernel, (r, r), r, 1, -1)  # 卷积核 33*33 半径为r的圆区域全为1
#
#     openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)  # 开运算 先腐蚀后膨胀
#     print(openingimg[1][:10])
#     strtimg = cv2.absdiff(stretchedimg, openingimg)  # 两幅图做差分
#     print(strtimg[1][:10])
#
#     # 图像二值化
#     binaryimg = dobinaryzation(strtimg)
#     # cv2.imshow("binary",binaryimg)
#
#     rectca = cascade(grayimg)
#
#     if len(rectca)<1:
#         print("未检测到车牌")
#         return 0
#     elif len(rectca)==1:
#         return binaryimg[rectca[0][1]:rectca[0][3],rectca[0][0]:rectca[0][2]]
#     else:
#
#
#         cannyimg = cv2.Canny(binaryimg, binaryimg.shape[0], binaryimg.shape[1])
#         cv2.imshow("canny", cannyimg)
#
#         sobelimg=cv2.Sobel(binaryimg,ddepth=cv2.CV_8U,dx=1,dy=0)
#         cv2.imshow("sobel",sobelimg)
#         ''' 消除小区域，保留大块区域，从而定位车牌'''
#         # 进行闭运算
#         kernel = np.ones((11, 13), np.uint8)
#         closingimg = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
#         cv2.imshow("close", closingimg)
#             # 进行开运算
#         openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)
#         cv2.imshow("open", openingimg)
#             # 再次进行开运算
#         kernel = np.ones((7,5), np.uint8)
#         openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)
#         cv2.imshow("open1", openingimg)
#             # 消除小区域，定位车牌位置
#         # rect = locate_license(openingimg, img)
#         #查找轮廓
#         contours, hierarchy = cv2.findContours(openingimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         #选取矩形区域
#
#         area_list=choose_area(contours)
#         print("area_list",area_list)
#
#         area_list=sorted(area_list,key=lambda b: b[1])[-3:]
#         print("maxarea",area_list)
#         rect = is_color(area_list, img)
#
#
#         rects=[]
#         for i in range(len(rectca)):
#             for j in range(len(rect)):
#                 rect_temp=xiangjiao(rectca[i],rect[j][0])
#                 rects.append(rect_temp)
#
#
#         print("rects",type(rects))
#         print("rects[0]", rects[0])
#
#         # print("车牌颜色：",rect[0][1])
#         chepai_img=binaryimg[rects[0][1]:rects[0][3],rects[0][0]:rects[0][2]]
#         print(chepai_img.shape)
#         cv2.imshow("chepai",chepai_img)
#         return chepai_img

def get_max_XY(mask,x,y):
    x_max = x
    y_max = y
    while mask[x,y]==255:
        x_max=x
        y_max=y
        if mask[x+1,y+1]==255:
            x+=1
            y+=1
        elif mask[x+1,y]==255:
            x+=1
        elif mask[x,y+1]==255:
            y+=1
        else:
            x+1
            y+=1
    return x_max,y_max

def get_min_XY(mask,x,y):
    x_min=x
    y_min=y
    while mask[x,y]==255:
        x_min=x
        y_min=y
        if mask[x-1,y-1]==255:
            x-=1
            y-=1
        elif mask[x-1,y]==255:
            x-=1
        elif mask[x,y-1]==255:
            y-=1
        else:
            x-1
            y-=1
    return x_min,y_min

def get_rect(img,rects):
    rect_re=[]
    for i in range(len(rects)):
        #print("rect",rects[i])
        # x0=int(rect[i][0]+(rect[i][2]-rect[i][0])/2)
        # y0=int(rect[i][1]+(rect[i][3]-rect[i][1])/2)
        # # RGB转HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #print("hsv.shape", hsv.shape)
        # print('hsv',hsv)
        # 蓝色车牌范围
        blue_lower = np.array([100, 120, 120])
        blue_upper = np.array([124, 255, 255])
        # 绿色范围
        # green_lower = np.array([35, 43, 46])
        # green_upper = np.array([77, 255, 255])
        # 根据阈值构建掩模
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        # 进行开闭运算
        kernel = np.ones((9, 11), np.uint8)
        mask_blue_open = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("close", mask_blue_open)
        kernel = np.ones((7, 7), np.uint8)
        mask_blue_close = cv2.morphologyEx(mask_blue_open, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("open", mask_blue_close)
        # mask_green = cv2.inRange(hsv, green_lower, green_upper)
        #cv2.imshow("blue",mask_blue)

        contours, hierarchy = cv2.findContours(mask_blue_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 选取矩形区域

        area_list = choose_area(contours)
        #print("area_list", area_list)

        rect_blue = sorted(area_list, key=lambda b: b[1])[-2:]
        rect_blue=rect_blue[0][0]
        #print("maxarea", rect_blue)
        # rect = is_color(area_list, img)
        rect1=rects[i]
        rect_re=xiangjiao(rect1,rect_blue)

        #cv2.waitKey(0)

    return rect_re



# if __name__=="__main__":
def get_img_data(path):
    img = cv2.imread(path)
    print(img.shape)
    img = cv2.resize(img,dsize=(600,500))
    rects=other.cascade(img)
    if len(rects)>0:
        rect=get_rect(img,rects)
        #print("jieguo",rect)
        graImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # medianImg=cv2.medianBlur(graImg,ksize=5)
        chepai=graImg[rect[1]:rect[3],rect[0]:rect[2]]
        chepai=dobinaryzation(chepai)
        #cv2.imshow("bin",chepai)
        separate.fenge(chepai)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("flase")




    #字符识别

