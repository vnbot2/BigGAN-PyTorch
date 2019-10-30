from common import *

paths = glob.glob('./downloads/motobike/*')

# for path in paths:
def f(path):
    img = pv.imread(path)
    if img is not None:
        pv.imwrite(path, img)
    else:
        print('remove: ', path)
        os.remove(path)

multi_thread(f, paths, verbose=1)