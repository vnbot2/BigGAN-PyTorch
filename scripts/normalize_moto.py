from common import *

paths = glob.glob('./downloads/motobike/*')

# for path in paths:
def f(path):
    try:
        cv2.imwrite(path, cv2.imread(path))
    except:
        print('remove: ', path)
        os.remove(path)

multi_thread(f, paths, verbose=1)