import cv2
import numpy as np


def getA4():
    img = cv2.imread('./input.jpg', 0)

    edges = cv2.Canny(img, 120, 240)
    # cv2.imwrite('./edges.jpg', edges)

    raw_lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)

    lines = raw_lines

    # for line in lines:
    #     rho, theta = line[0]
    #     l_tan = np.tan(theta)
    #     k = 1 / l_tan
    #     print(k)

    # 寻找4个角点
    n = len(lines)
    points = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]

            l1_tan = np.tan(theta1)
            l2_tan = np.tan(theta2)

            k1 = -1 / l1_tan
            k2 = -1 / l2_tan

            # 斜率乘积为-1，使用范围进行近似判断
            if -2 <= k1 * k2 < -0.05 or 0.05 <= k1 * k2 <= 3:
                # 解出方程组直接得到交点
                x = int((rho1 / np.cos(theta1) - rho2 /
                         np.cos(theta2)) / (l1_tan - l2_tan))
                y = int((rho1 / np.sin(theta1) - rho2 / np.sin(theta2)) /
                        (1 / l1_tan - 1 / l2_tan))

                flag = True
                for y0, x0 in points:
                    dis = (x - x0)**2 + (y - y0)**2
                    if dis < 100:
                        # print("this point isn't selected:", x, y)
                        flag = False
                        break
                if flag is True:
                    # 通过debug发现opencv的坐标映射和我想的有点出入，需要交换x和y
                    points.append((y, x))

    # 在ponits中进行透视变换，输出为原图大小
    points = sorted(points, key=lambda x: x[1])
    height, width = img.shape

    pts1 = np.float32([points[1], points[0], points[3], points[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (width, height))

    cv2.imwrite('./getA4.jpg', dst)

    return dst


if __name__ == '__main__':
    getA4()
