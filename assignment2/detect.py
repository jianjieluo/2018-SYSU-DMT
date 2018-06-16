import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('./input.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 通过阀值进行前后景的分割
    # 这里的170是试出来的，为的是让所有的线段都可以被分离出来
    ret, thresh = cv2.threshold(gray, 170, 255, 0)

    # 形态学图像处理(膨胀腐蚀)
    # 这里迭代的次数2是试出来的，kernel的设置是参考了网上的教程的
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(thresh, kernel, iterations=2)

    # 轮廓检测
    image, contours, hierarchy = cv2.findContours(
        erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = len(contours)
    outputs = []
    # conturs的第一个元素是框出整个图片的大矩阵，故不选
    for i in range(1, n):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 使用ROI来进行分割
        # 这里两边的坐标体系有点迷
        part_img = img[y:y + h, x:x + w]
        outputs.append(part_img)

    # while(1):
    #     cv2.imshow('img', img)
    #     cv2.imshow('erosion', erosion)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # 同过ROI来一个一个输出
    for i in range(0, len(outputs)):
        cv2.imwrite('./output/' + str(i) + '.jpg', outputs[i])
