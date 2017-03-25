#!/usr/bin/env python

import sys
import time
import array
from Xlib import display, X
import wx
wx.App()

from skimage.transform import resize
from skimage.util import img_as_float
from skimage.io._plugins.pil_plugin import (pil_to_ndarray, ndarray_to_pil, _palette_is_grayscale)
import skimage.io as sio

import numpy as np

from PIL import Image


# CONSTANTS #
SRC_W = 640
SRC_H = 480
SRC_D = 3
OFFSET_X = 400
OFFSET_Y = 240

IMG_W = 200
IMG_H = 66
IMG_D = 3


def original_take_screenshot():
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.Bitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    return bmp.GetSubBitmap(wx.Rect([0,0],[SRC_W,SRC_H]))


def modified_take_screenshot():
    screen = wx.ScreenDC()
    bmp = wx.Bitmap(SRC_W, SRC_H)
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, SRC_W, SRC_H, screen, OFFSET_X, OFFSET_Y)
    return bmp

def alternative_take_screenshot():
    dsp = display.Display()
    root = dsp.screen().root
    raw = root.get_image(0, 0, SRC_W, SRC_H, X.ZPixmap, 0xffffffff)
    image = Image.frombytes("RGB", (SRC_W, SRC_H), raw.data, "raw", "BGRX")
    return image


def original_prepare_image(img):
    buf = img.ConvertToImage().GetData()
    img = np.frombuffer(buf, dtype='uint8')

    img = img.reshape(SRC_H, SRC_W, SRC_D)
    img = resize(img, [IMG_H, IMG_W])

    return img


arr = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));
def modified_prepare_image(img):
    img.CopyToBuffer(arr)
    img = np.frombuffer(arr, dtype=np.uint8)

    img = img.reshape(SRC_H, SRC_W, SRC_D)

    im = Image.fromarray(img)
    im = im.resize((IMG_W, IMG_H))

    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D))

def alternative_prepare_image(image):
    if str(type(image)) == "<class 'PIL.Image.Image'>":
      image = pil_to_ndarray(image)
    ndar = image.reshape(SRC_H, SRC_W, SRC_D)
    im = Image.fromarray(ndar)
    im = im.resize((IMG_W, IMG_H))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8) # in object exposing buffer interface out ndarray
    im_arr = im_arr.reshape((IMG_H, IMG_W, IMG_D)) # 200, 66, 3



def call_original():
    bmp = original_take_screenshot()
    vec = original_prepare_image(bmp)


def call_modified():
    bmp = modified_take_screenshot()
    vec = modified_prepare_image(bmp)

def call_alternative():
    bmp = alternative_take_screenshot()
    vec = alternative_prepare_image(bmp)


if __name__ == '__main__':
  import timeit

  try:
    n = int(sys.argv[1])
  except (ValueError, IndexError) as e:
    n = 100

  print("# Running tests " + str(n) + " times")

  # print("#")
  # print("# ORIGINAL:")
  # print(timeit.timeit("call_original()", setup="from __main__ import call_original;", number=n))

  # print("#")
  # print("# MODIFIED:")
  # print(timeit.timeit("call_modified()", setup="from __main__ import call_modified;", number=n))
 
  print("#")
  print("# ALTERNATIVE:")
  print(timeit.timeit("call_alternative()", setup="from __main__ import call_alternative;", number=n))

######################################################
# SOME RESULTS #
#
# Running tests 10000 times
#
# ORIGINAL:
# 1210.20094013
#
# MODIFIED:
# 313.987584114
#
#
# Running tests 10000 times
#
# ORIGINAL:
# 1074.97350001
#
# MODIFIED:
# 270.604922056
#

# Running tests 10 times
#
# ALTERNATIVE:
# 0.150660037994
  
# Running tests 100 times
#
# ALTERNATIVE:
# 1.30269503593

# Running tests 200 times
#
# ALTERNATIVE:
# 2.69465994835
# Running tests 50 times
#
# ALTERNATIVE:
# 0.64985203743
# 150
# 1.98458385468

######################################################
  

######################################################
# RESULTS DURING ACTUAL UTILS.PY PREPARE RUN #
#
# Preparing 4493 samples (8 races)
#
# ORIGINAL CODE: ~280s
#
# MODIFIED CODE: ~90s
#
######################################################
  

######################################################
# RESULTS DURING ACTUAL PLAY.PY RUN #
#
# ORIGINAL CODE:
#                  Screenshot        Prepare Image   Model Eval
# Avg Times (500): 0.000318816661835 0.136291568279  0.0443236446381
#
# MODIFIED CODE:
#                  Screenshot        Prepare Image   Model Eval
# Avg Times (500): 0.000203844547272 0.0492500219345 0.0412494616508
#
# IMPROVEMENT (AS % DECREASE OVER ORIGINAL):
#                  Screenshot        Prepare Image   Model Eval
#                  36.06%            63.86%          6.94% (execution variance - no changes)
#
######################################################


