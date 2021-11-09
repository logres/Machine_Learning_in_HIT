里面全是50像素*50像素的图片，灰度的

from PIL import Image
import numpy as np

im = Image.open($path_to_picture)
x = np.array(im)

你就可以得到x，一个50*50的ndarray

im = Image.fromarray(x)
im = im.convert('L')
im.save($file_name)

你就可以把一个二维的ndarray保存为一张灰度图片