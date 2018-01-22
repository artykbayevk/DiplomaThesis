from __future__ import print_function, division
import os
import skimage.data
import selectivesearch
import glob
import random
import time
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

def segment_images(imgs, out_dir, mode):
    print('smth')
    for img in imgs:
        image = io.imread(img)
        size = int((image.shape[0]*image.shape[1])*0.1)

        img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=size)
        rects = []
        if mode == 1:
            new_folder = (img.split('/')[-1]).split('.')[0]
            os.mkdir(os.path.join(out_dir, new_folder))

        for item in regions:
            tmp = item['rect']
            if tmp not in rects:
                rects.append(tmp)
        for i,segment in zip(range(len(rects)), rects):
            x1 = segment[0]
            y1 = segment[1]
            x2 = segment[2]
            y2 = segment[3]
            seg = image[y1:y2, x1:x2]
            try:
                if mode == 1:
                    new_f_name = (img.split('/')[-1]).split('.')[0]+'_'+str(i)+'.jpg'
                    new_path = os.path.join(out_dir, new_folder, new_f_name)
                    try:
                        io.imsave(new_path, seg)
                    except IndexError:
                        print('No axes')
                else:
                    io.imshow(seg)
                    io.show()
                    time.sleep(0.5)
            except ValueError:
                print(y1,y2, x1,x2)
                print('no segment')

def main():
    input_img_path = '../data/fl27/images'
    output_img_path = '../data/fl27/segmented'
    imgs = get_images(input_img_path, 4, False)
    segment_images(imgs, output_img_path, 1)

if __name__ == '__main__':
    # main()
    pass
