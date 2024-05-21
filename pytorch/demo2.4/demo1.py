
import numpy as np
##使用一张图片来展示经过卷积后的图像效果
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

##读取图像-->转化为灰度图片-->转化为Numpy数组
myim = Image.open("../data/test1.png")
myimgray = np.array(myim.convert("L"), dtype=np.float32)
##可视化图片
plt.figure(figsize=(6, 6))
plt.imshow(myimgray, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

