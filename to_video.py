from common import *
import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i')
parser.add_argument('--output_path', '-o')
parser.add_argument('--fps', default=5, type=int)
parser.add_argument('--size', default=5)
args = parser.parse_args()



frame_array = []
image_paths = glob.glob(args.input_dir+'/*.jpg')
def sort_fn(x):
    return int(os.path.basename(x).split('s')[2].split('.')[0])

image_paths = list(sorted(image_paths, key=sort_fn))
print(image_paths)
# import ipdb; ipdb.set_trace()
for image_path in image_paths:
    #reading each files
    img = cv2.imread(image_path)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc(*'DIVX'), args.fps, size)
for frame in frame_array:
    # writing to a image array
    out.write(frame)
out.release()
