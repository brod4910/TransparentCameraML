import time
import os,sys,inspect

from toupcam.camera import ToupCamCamera


import numpy, PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def main():

    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    cam = ToupCamCamera()
    cam.set_exposure_time(150)
    cam.open()

    # wait for camera to startup
    time.sleep(2)

    # capture n images
    n = 10
    wait = 1
    # every t seconds
    t = .5
    
    directory = "val2017"
    new_dir = 'tpval2017_10frames_r'

    path = os.path.join(os.getcwd(), directory)
    allfiles=os.listdir(path)
    imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"] and '_' not in filename]
    print(imlist)
    for i, img in enumerate(imlist):
        im = plt.imread(os.path.join(path, img))
        if i == 0:
            image = plt.imshow(im)
        else:
            image.set_data(im)
        if i == 0:
            plt.pause(180)
        else:
            plt.pause(.1)

        plt.draw()

        new_im_dir = img[:-4]
        new_path = os.path.join(new_dir, new_im_dir)

        time.sleep(wait)

        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print('Capturing image {}'.format(img))
            for j in range(10):
        
                p = new_path + '\\{:03d}.jpg'.format(j)
                cam.save(p)
                time.sleep(.05)
        else:
            continue


if __name__ == '__main__':
    main()
