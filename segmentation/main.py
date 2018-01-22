from __future__ import print_function, division
import os
import skimage.data
import selectivesearch
import glob
import random

from skimage import io

import warnings
warnings.filterwarnings("ignore")

def get_images(path, count, default = True):
    all_path = os.path.join(path, '*.jpg')
    files = glob.glob(all_path)
    if default:
        count = len(files)
    else:
        random.shuffle(files,random.random)

    if count>len(files):
        return files
    else:
        return files[:count]

def segment_images(imgs, out_dir):
    print("check this one")
    for img in imgs:
        image = io.imread(img)
        size = int((image.shape[0]*image.shape[1])*0.1)
        print(size)
        img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=size)
        rects = []
        for item in regions:
            tmp = item['rect']
            if tmp not in rects:
                print(item['size'])
                rects.append(tmp)
        for segment in rects:
            x1 = segment[0]
            y1 = segment[1]
            x2 = segment[2]
            y2 = segment[3]
            seg = image[y1:y2, x1:x2]
            try:
                io.imshow(seg)
                io.show()
            except ValueError:
                print(y1,y2, x1,x2)
                print('no segment')

def main():
    input_img_path = '../data/fl27/images'
    output_img_path = '../data/fl27/segmented'
    imgs = get_images(input_img_path, 2, False)
    segment_images(imgs, output_img_path)
if __name__ == '__main__':
    main()
