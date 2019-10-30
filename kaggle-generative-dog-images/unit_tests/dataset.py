from dataset import BikeDataset
from common import *
# import ipdb; ipdb.set_trace()
paths = glob.glob('/dataset/moto/motobike/*')
ds = BikeDataset(paths)
i = 0
x = ds.__getitem__(i)
cv2.imwrite('cache/{}.png'.format(i), x['img'])
