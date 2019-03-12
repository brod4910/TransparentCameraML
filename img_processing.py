import time
import os,sys,inspect
from PIL import Image
import numpy as np

def average_frames(direct, dirname):
    # Access all PNG files in directory
    allfiles = os.listdir(os.path.join(direct, dirname))
    imlist = [os.path.join(direct, dirname, filename) for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

    # Assuming all images are the same size, get dimensions of first image
    w,h = Image.open(imlist[0]).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = numpy.zeros((h,w,3),numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr = numpy.array(Image.open(im),dtype=numpy.float)
        arr = arr + imarr/N

    # Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    out.save("train2017/{}.png".format(dirname))

def convert_png_to_jpg(direct, new_direct):
    allfiles = os.listdir(direct)

    for i, filename in enumerate(allfiles):
        if not filename.startswith('.'):
            im = Image.open(os.path.join(direct, filename))
            rgb_im.save('{}.tiff'.format(os.path.join(new_direct, filename[:-4])))

def main():
    convert_png_to_jpg('/scratch/kingspeak/serial/u0853593/train2017', '/scratch/kingspeak/serial/u0853593/train2017_jpg')

if __name__ == '__main__':
    main()