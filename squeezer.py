from skimage.io import imread
import numpy as np
from PIL import Image
#from skimage.io import imread
from skimage.util import img_as_float
from Xlib import display, X
from skimage.io._plugins.pil_plugin import (pil_to_ndarray, ndarray_to_pil, _palette_is_grayscale)
import skimage.io as sio
import time

sio.use_plugin('pil')

SRC_W = 640
SRC_H = 480
SRC_D = 3
OFFSET_X = 400
OFFSET_Y = 240

IMG_W = 200
IMG_H = 66
IMG_D = 3



# def take_screenshot():
#     screen = wx.ScreenDC()
#     bmp = wx.Bitmap(Screenshot.SRC_W, Screenshot.SRC_H)
#     mem = wx.MemoryDC(bmp)
#     mem.Blit(0, 0, Screenshot.SRC_W, Screenshot.SRC_H, screen, Screenshot.OFFSET_X, Screenshot.OFFSET_Y)
#     return bmp


######### my take screenshot #########
# filename = str(time.time())
# dsp = display.Display()
# root = dsp.screen().root
# raw = root.get_image(0, 0, SRC_W, SRC_H, X.ZPixmap, 0xffffffff)
# image = Image.frombytes("RGB", (SRC_W, SRC_H), raw.data, "raw", "BGRX")
# image.save(filename,'png')

# def prepare_image(img):
#     if(type(img) == wx._core.Bitmap):
#         img.CopyToBuffer(Screenshot.image_array)
#         img = np.frombuffer(Screenshot.image_array, dtype=np.uint8) # shape (921600,)

#     img = img.reshape(Screenshot.SRC_H, Screenshot.SRC_W, Screenshot.SRC_D) # shape (480, 640, 3)

# 	  array to image
#     im = Image.fromarray(img) 

#     image resize
#     im = im.resize((Screenshot.IMG_W, Screenshot.IMG_H))

# 	  image to bytes to array
#     im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)

# 	  array reshape
#     im_arr = im_arr.reshape((Screenshot.IMG_H, Screenshot.IMG_W, Screenshot.IMG_D))
#     return img_as_float(im_arr)

image_file = 'img_41.png'
img = imread(image_file) # returns ndarray
img = Image.fromarray(img)
img = img.resize((IMG_W, IMG_H))
img.save('tiny.png','png')

######### my prepare image #########


# # image to array
# ndarray = pil_to_ndarray(image)
# # print "ndarray type"
# print(ndarray.shape) # array dimensions (480, 640, 3)
# print(ndarray.size) # Number of elements in the array = 480 * 640 * 3

# # array reshape useless
# ndar = ndarray.reshape(SRC_H, SRC_W, SRC_D) # useless
# print(ndar.shape) # array dimensions (480, 640, 3)
# print(ndar.size) # Number of elements in the array = 480 * 640 * 3

# # array to image
# im = Image.fromarray(ndar) 
# print(type(im))

# # image resize (rescale) 
# im = im.resize((IMG_W, IMG_H))
# print(type(im))

# # image to bytes to array
# im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8) # in object exposing buffer interface out ndarray
# print(im_arr.shape) # array dimensions (39600,)
# print(im_arr.size) # Number of elements in the array = 39600

# # array reshape 
# im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D)) # 66, 200, 3
# print(im_arr.shape) # array dimensions (66, 200, 3)
# print(im_arr.size) # Number of elements in the array = 39600
# out = img_as_float(im_arr) # in ndarray out ndarray of float64 
# # print(type(out))



# # image resize (rescale) 
# image = image.resize((IMG_W, IMG_H)) # 200, 66
# filename = str(time.time())
# image.save(filename,'png')

# # ndarray = pil_to_ndarray(image)
# # print(ndarray.shape) # array dimensions (66, 200, 3)
# # print(ndarray.size) # Number of elements in the array = 39600

# # # print(im_arr.shape) # array dimensions (66, 200, 3)
# # # print(im_arr.size) # Number of elements in the array = 39600
# # out = img_as_float(im_arr) # in ndarray out ndarray of float64 
# # print(type(out))
