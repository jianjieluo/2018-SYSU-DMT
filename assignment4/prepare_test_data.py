import cv2
import numpy as np


def get_row_hist(binary_img):
    m, n = binary_img.shape
    row_hist = [0] * m
    for i in range(m):
        row_hist[i] = n - np.count_nonzero(binary_img[i, :])

    return row_hist


def split_row(img, row_hist):
    up = []
    down = []

    for i in range(10, len(row_hist)):
        if row_hist[i] > 0 and row_hist[i - 1] == 0:
            up.append(i - 10)
        elif row_hist[i] == 0 and row_hist[i - 1] > 0:
            down.append(i + 10)

    res = []
    if len(up) == len(down):
        for i in range(len(down)):
            res.append(img[up[i]:down[i], :])

    return res


def sort_contours(contours):
    temp = [cv2.boundingRect(cnt)[0] for cnt in contours]
    a = {temp[i]: i for i in range(len(temp))}
    temp = sorted(temp)
    res = []

    for i in range(len(temp)):
        res.append(contours[a[temp[i]]])

    return res


def img2mnist(origin):
    m, n = origin.shape
    # origin = cv2.GaussianBlur(origin, (3, 3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    origin = cv2.erode(origin, kernel, iterations=2)

    # 图片反转变换
    def _reverse(t): return 255 - t
    vfunc = np.vectorize(_reverse)
    temp = vfunc(origin.reshape(1, m * n)).reshape(m, n)

    # 将res转成28*28的图片
    scale = max(m, n) / 28
    res = cv2.resize(temp, (int(n / scale), int(m / scale)),
                     interpolation=cv2.INTER_NEAREST)

    m, n = res.shape
    if m != 28 and n != 28:
        print("m and n are not 28 at the same time!")
        exit(-1)
    else:
        if m == 28 and n < 28:
            # 横着pad
            pad_width = int((28 - n) / 2)
            final_res = np.lib.pad(res, ((0, 0), (pad_width, 28 - n - pad_width)),
                                   'constant', constant_values=0)
        elif m < 28 and n == 28:
            pad_height = int((28 - m) / 2)
            final_res = np.lib.pad(res, ((pad_height, 28 - m - pad_height), (0, 0)),
                                   'constant', constant_values=0)
        else:
            print('error: transforming to mnist img', m, n)
            exit(-1)

    if final_res.shape != (28, 28):
        print('error: transforming to mnist img', m, n)
        exit(-1)

    return final_res


def split(img, row_id):
    _, binary = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('./process/binary.jpg', binary)

    # 形态学图像处理(膨胀腐蚀)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erosion = cv2.erode(binary, kernel, iterations=2)

    # 轮廓检测
    image, contours, hierarchy = cv2.findContours(
        erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = len(contours)

    contours = sort_contours(contours)

    outputs = []
    for i in range(0, n):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 55 and h <= 55:
            pass
        else:
            if w > 500:
                pass
            else:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                part_img = img[y:y + h, x:x + w]
                outputs.append(part_img)

    for i in range(0, len(outputs)):
        res = img2mnist(outputs[i])
        cv2.imwrite('./process/%d_%d.jpg' % (row_id, i), res)


def prepare_test_data():
    img = cv2.imread('./getA4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 通过阀值进行前后景的分割
    # 这里的170是试出来的，为的是让所有的线段都可以被分离出来
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    thresh = thresh[70:-70, 70:-70]

    row_hist = get_row_hist(thresh)
    rows = split_row(thresh, row_hist)

    outputs = []
    for i in range(len(rows)):
        outputs.append(split(rows[i], i))

    return outputs


if __name__ == '__main__':
    res = prepare_test_data()
