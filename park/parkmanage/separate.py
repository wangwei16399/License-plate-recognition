import cv2
import numpy as np


def jishu(img):
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img.shape[0]
    width = img.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img[j][i] == 255:
                s += 1
            if img[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
    # print("white", sum(white))
    # print("black", sum(black))
    return white, black, height, width, white_max, black_max


def save_zifu(img, count):
    s = img.shape[0] * img.shape[1]
    c = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 10:
                c += 1
    if c / s > 0.15:
        img = cv2.resize(img, dsize=(32, 40))
        cv2.imwrite("D:/license_data/%s" % count + ".bmp", img)
        # 保存图片
        print("保存字符%s" % count)
        count += 1
    else:
        pass
    return count


def fenge(img):
    white, black, height, width, white_max, black_max = jishu(img)

    start = 0
    i = 0
    count = 0
    while i < width and count < 7:
        if i == width - 2 and i - start > 5:
            # characters.append(img[0:height, start:i])
            #print("字符长度：", i - start)
            zifu = img[0:height, start:i + 2]
            # zifu = cv2.resize(zifu, dsize=(32, 40))
            #cv2.imshow("zifu%s" % i, zifu)
            # cv2.imwrite("E://img//zifu//%s" % count + ".bmp", zifu)
            # count += 1
            count = save_zifu(zifu, count)
            break
        elif white[i] > 0.05 * black_max:
            i += 1
        elif i < width - 2 and white[i + 1] <= 0.05 * black_max and white[i + 2] <= 0.05 * black_max:
            if i - start > 4:
                # characters.append(img[0:height,start:i])
                #print("字符长度：", i - start)
                zifu = img[0:height, start:i]
                # zifu=cv2.resize(zifu,dsize=(32,40))
                #cv2.imshow("zifu%s" % i, zifu)
                # cv2.imwrite("E://img//zifu//%s"%count+".bmp",zifu)
                # count+=1
                count = save_zifu(zifu, count)
                # cv2.waitKey(0)
                start = i + 2
                i = start
            else:
                i += 1
                start = i
        else:
            i += 1

    # return to_standard_img(characters)


def to_standard_img(characters):
    characters_1 = []
    for i in range(len(characters)):
        # print("zi", characters[i].shape)
        character = characters[i]
        if character.shape[1] < 8:
            character1 = np.zeros(shape=(character.shape[0], character.shape[1] * 4))
            i = int(character.shape[1] * 1.5)
            j = 0
            while j < character.shape[1]:
                for k in range(character.shape[0]):
                    character1[k][i + j] = character[k][j]
                j += 1
            character1 = cv2.resize(character1, dsize=(40, 32))
            character1 = cv2.resize(character1, dsize=(1, 1280))
            characters_1.append(character1)
        else:
            character = cv2.resize(character, dsize=(40, 32))
            character = cv2.resize(character, dsize=(1, 1280))
            characters_1.append(character)

        f1 = open('out.txt', 'w')

        characters1 = np.array(characters_1)
        rows, cols, dreep = characters1.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(dreep):
                    if (characters1[i, j, k] <= 5):
                        characters1[i, j, k] = 0
                    else:
                        characters1[i, j, k] = 1
                    f1.write(str(characters1[i, j, k]))
        f1.close()
    return characters1