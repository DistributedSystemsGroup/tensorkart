import PIL.ImageOps 
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

filename_in = 'img_51.png'
filename_out = 'img_51_edit.png'

def darken_images(samples):
	# edit brightness
	R, G, B = 0, 1, 2
	constant = 1/(1.5*2.2)	 # constant > 1  will darken the image
	print("Darkening images from ...")
	for sample in samples:
		print(sample) # e.g. luigi_raceway1
		# load sample
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		# load, prepare and add images to X
		for image_file in image_files:
			im = Image.open(image_file)
			source = im.split()
			Red = source[R].point(lambda i: i/constant)
			Green = source[G].point(lambda i: i/constant)
			Blue = source[B].point(lambda i: i/constant)
			im = Image.merge(im.mode, (Red, Green, Blue))
			im.save(image_file, 'PNG', quality=100)
	print("Done!")

def flip_images(samples):
	print("Flipping images from ...")
	for sample in samples:
		print(sample)
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		for image_file in image_files:
			# flip along vertical axis
			im = Image.open(image_file).transpose(Image.FLIP_LEFT_RIGHT)
			im.save(image_file, 'PNG', quality=100)
	print("Done!")

def invert_colors(samples):
	print("Inverting colors of images from ...")
	for sample in samples:
		print(sample)
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		for image_file in image_files:
			im = Image.open(image_file)
			inverted_image = PIL.ImageOps.invert(im)
			inverted_image.save(image_file, 'PNG', quality=100)
	print("Done!")

def squeeze_images(samples):
	IMG_W = 200
	IMG_H = 66
	print("Squeezing images from ...")
	for sample in samples:
		print(sample)
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		for image_file in image_files:
			img = Image.open(image_file)
			img = img.resize((IMG_W, IMG_H))
			img.save(image_file, 'PNG', quality=100)
	print("Done!")

def crop_images(samples):
	print("Cropping images from ...")
	for sample in samples:
		print(sample)
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		for image_file in image_files:
			im = Image.open(image_file)
			w, h = im.size
			im = im.crop((0,185,w,h-75)) # (0,0,w,h) is the image, (0,20,w,h-10) removes 20 pixels above and 10 below
			# im = im.crop((0,135,w,h-125)) # (0,0,w,h) is the image, (0,20,w,h-10) removes 20 pixels above and 10 below
			im.save(image_file, 'PNG', quality=100)

	print("Done!")

def remove_mario(samples):
	print("Removing Mario from ...")
	for sample in samples:
		print(sample)
		image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,)) #luigi_raceway1/data.csv
		for image_file in image_files:
			im = Image.open(image_file)
			w, h = im.size
			draw = ImageDraw.Draw(im)
			# draw.rectangle([(w/2.)-70, (h/2.)-30, (w/2.)+70, h], fill=0) # [x0,y0,x1,y1]
			draw.rectangle([(w/2.)-80, (h/2.), (w/2.)+80, h], fill=0) # [x0,y0,x1,y1]
			del draw
			im.save(image_file, 'PNG', quality=100)

	print("Done!")

if __name__ == '__main__':
	# run the following command, then flipper.sh
	if sys.argv[1] == 'flip': 
		flip_images(sys.argv[2:])
	elif sys.argv[1] == 'darken': # python image_editing.py darken samples/* 
		darken_images(sys.argv[2:])
	elif sys.argv[1] == 'invert': # python image_editing.py invert samples/* 
		invert_colors(sys.argv[2:])
	elif sys.argv[1] == 'crop': # python image_editing.py crop samples/* 
		crop_images(sys.argv[2:])
	elif sys.argv[1] == 'remove_mario': # python image_editing.py remove_mario samples/* 
		remove_mario(sys.argv[2:])
	elif sys.argv[1] == 'squeeze_images': # python image_editing.py squeeze_images samples/* 
		squeeze_images(sys.argv[2:])


