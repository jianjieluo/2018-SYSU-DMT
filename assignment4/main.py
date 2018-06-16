from getA4 import *
from prepare_test_data import *
from mnist_deep import *
from identify import *


if __name__ == '__main__':
    # A4纸的校正
    getA4()
    # 数字分割
    prepare_test_data()

    # 训练模型
    # tf.app.run(main=main)

    # 使用模型来进行识别并输出
    res = identify_test()
    print(res[0:6])
    print(res[6:14])
    print(res[14:])
