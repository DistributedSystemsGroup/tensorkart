#!/usr/bin/env python
import random
import sys
import array
# import pygame
# import wx
# wx.App()

import numpy as np

from PIL import Image
# from PyQt4.QtGui import QPixmap, QApplication
from datetime import datetime
from Xlib import display, X

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float
from skimage.io._plugins.pil_plugin import (pil_to_ndarray, ndarray_to_pil, _palette_is_grayscale)
import skimage.io as sio


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time



def take_screenshot():

    # ORIGINAL
    # screen = wx.ScreenDC()
    # bmp = wx.Bitmap(Screenshot.SRC_W, Screenshot.SRC_H)
    # mem = wx.MemoryDC(bmp)
    # mem.Blit(0, 0, Screenshot.SRC_W, Screenshot.SRC_H, screen, Screenshot.OFFSET_X, Screenshot.OFFSET_Y)
    # return bmp

    # ALTERNATIVE
    # http://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
    dsp = display.Display()
    root = dsp.screen().root
    raw = root.get_image(0, 50, Screenshot.SRC_W, Screenshot.SRC_H, X.ZPixmap, 0xffffffff)
    image = Image.frombytes("RGB", (Screenshot.SRC_W, Screenshot.SRC_H), raw.data, "raw", "BGRX")
    # date = datetime.now()
    # filename = date.strftime('%Y-%m-%d_%H-%M-%S.png')
    # filename = str(time.time())
    # print(type(image))
    # image.save(filename,'png')
    return image


def prepare_image(img):

    # ORIGINAL
    # if(type(img) == wx._core.Bitmap):
    #     img.CopyToBuffer(Screenshot.image_array) # img = pixel data
    #     img = np.frombuffer(Screenshot.image_array, dtype=np.uint8) # in object exposing buffer interface out ndarray
    # img = img.reshape(Screenshot.SRC_H, Screenshot.SRC_W, Screenshot.SRC_D)
    # im = Image.fromarray(img)
    # im = im.resize((Screenshot.IMG_W, Screenshot.IMG_H))
    # im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8) # in object exposing buffer interface out ndarray
    # im_arr = im_arr.reshape((Screenshot.IMG_H, Screenshot.IMG_W, Screenshot.IMG_D)) # 200, 66, 3
    # return img_as_float(im_arr) # in ndarray out ndarray of float64 

    # ALTERNATIVE
    if str(type(img)) == "<class 'PIL.Image.Image'>":
        img = img.resize((Screenshot.IMG_W, Screenshot.IMG_H)) # 200, 66
    else:
        img = img.reshape(Screenshot.SRC_H, Screenshot.SRC_W, Screenshot.SRC_D)
        img = Image.fromarray(img)
        img = img.resize((Screenshot.IMG_W, Screenshot.IMG_H))
    im_arr = pil_to_ndarray(img)
    return img_as_float(im_arr) # in ndarray out ndarray of float64 # shape is unchanged

class Screenshot:
    SRC_W = 640
    # SRC_H = 480
    SRC_H = 220
    SRC_D = 3

    OFFSET_X = 0
    OFFSET_Y = 0

    IMG_W = 200
    IMG_H = 66
    IMG_D = 3

    image_array = array.array('B', [0] * (SRC_W * SRC_H * SRC_D));


# class XboxController:
#     def __init__(self):
#         try:
#             pygame.init()
#             self.joystick = pygame.joystick.Joystick(0)
#             self.joystick.init()
#         except:
#             print('unable to connect to Xbox Controller')


#     def read(self):
#         pygame.event.pump()
#         x = self.joystick.get_axis(0)
#         y = self.joystick.get_axis(1)
#         a = self.joystick.get_button(0)
#         b = self.joystick.get_button(2)
#         rb = self.joystick.get_button(5)
#         return [x, y, a, b, rb]


#     def manual_override(self):
#         pygame.event.pump()
#         return self.joystick.get_button(4) == 1


class Data(object):
    def __init__(self, test=0):
        if test == 1:
            self._y = np.load("test_data/y.npy")
            # take only steering
            # self._y = np.delete(self._y, 1, axis=1)
            self._X = np.load("test_data/X.npy")
        else:        
            self._y = np.load("data/y.npy")
            # take only steering
            # self._y = np.delete(self._y, 1, axis=1)
            self._X = np.load("data/X.npy")
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._X.shape[0]

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


def load_sample(sample):
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
#    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1,3)) # default float
    return image_files, joystick_values


# # training data viewer
# def viewer(sample):
#     image_files, joystick_values = load_sample(sample)

#     plotData = []

#     plt.ion()
#     plt.figure('viewer', figsize=(16, 6))

#     for i in range(len(image_files)):

#         # joystick
#         print i, " ", joystick_values[i,:]

#         # format data
#         plotData.append( joystick_values[i,:] )
#         if len(plotData) > 30:
#             plotData.pop(0)
#         x = np.asarray(plotData)

#         # image (every 3rd)
#         if (i % 3 == 0):
#             plt.subplot(121)
#             image_file = image_files[i]
#             img = mpimg.imread(image_file)
#             plt.imshow(img)

#         # plot
#         plt.subplot(122)
#         plt.plot(range(i,i+len(plotData)), x[:,0], 'r')
#         plt.hold(True)
#         plt.plot(range(i,i+len(plotData)), x[:,1], 'b')
#         plt.plot(range(i,i+len(plotData)), x[:,2], 'g')
#         plt.plot(range(i,i+len(plotData)), x[:,3], 'k')
#         plt.plot(range(i,i+len(plotData)), x[:,4], 'y')
#         plt.draw()
#         plt.hold(False)

#         plt.pause(0.0001) # seconds
#         i += 1


# prepare training data
def prepare(samples, test=0):
    print("Preparing data")
    if test == 1:
        print("for TEST")
    X = []
    y = []

    for sample in samples:
        print(sample) # e.g. luigi_raceway1

        # load sample
        image_files, joystick_values = load_sample(sample)

        # add joystick values to y
        y.append(joystick_values) # list of ndarrays: steering and throttle
        # y is a list of num_folders ndarrays, each of which has size (num_samples_in_one_folder, 2) 
        for image_file in image_files:
            # print image_file
            image = imread(image_file) # returns ndarray
            vec = prepare_image(image) 
            X.append(vec)


    X = np.asarray(X) # before: list of ndarrays with shape (200, 60)
    y = np.concatenate(y) 
    if test == 0:
        assert len(X) == len(y)
        # shuffle list of images
        print("Shuffling...")

        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

    print("Saving to file...")
    if test == 1:
        np.save("test_data/X", X)
        np.save("test_data/y", y)
    else:
        np.save("data/X", X)
        np.save("data/y", y)

    print("Done!")
    return

if __name__ == '__main__':
    if sys.argv[1] == 'viewer':
        viewer(sys.argv[2])
    elif sys.argv[1] == 'prepare': # python utils.py prepare samples/*
        prepare(sys.argv[2:])
    elif sys.argv[1] == 'prepare_test': # python utils.py prepare_test samples/*
        prepare(sys.argv[2:], 1)

