# from models.common import *
# from tools.utils import *
from common import *



def autocontrast(img, cutoff=1): #cutoff[%]
    if np.random.rand() < 0.5:
        img = ImageOps.autocontrast(img, cutoff)
    return img

def sharpen(img, magnitude=1):
    factor = np.random.uniform(1.0-magnitude, 1.0+magnitude)
    img    = ImageEnhance.Sharpness(img).enhance(factor)
    return img

def detect_background_color(img):
    h, w = img.shape[:2]
    # if w > h:
    region = img[:10]
    return int(region.mean()> 128) *255    

def pad_if_needed(img, values=255):
    h, w = img.shape[:2]
    s = max(h, w)
    pad = np.ones([s,s, 3], dtype=img.dtype)*values
    start_h = (s-h)//2
    start_w = (s-w)//2
    pad[start_h:start_h+h, start_w:start_w+w] = img
    return pad


def data_preprocessing(img_path):
    img = pv.imread(img_path)
    bg_color = detect_background_color(img)
    img = pad_if_needed(img, bg_color)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 127.5 - 1
    img = img.transpose([2,0,1])
    return img.astype(np.float32) 


class MotoDataset(Dataset):
    def __init__(self, img_list, load_in_mem, augmentor=None):
        self.img_list  = img_list
        self.augmentor = augmentor
        self.load_in_mem = load_in_mem
        if self.load_in_mem:
            print('Load in mem...')
            # self.data = [data_preprocessing(full_img_path) for full_img_path in tqdm(img_list)]
            self.data = multi_thread(data_preprocessing, img_list, verbose=1)
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,idx):
        if self.load_in_mem:
            img = self.data[idx]
        else:
            full_img_path = self.img_list[idx]
            img = data_preprocessing(full_img_path)

        if self.augmentor is not None:
            img = self.augmentor(img)
        # else:
        #     # flip | [3, 64, 64]
        #     if pu.do_by_chance(50):# 50 % flip
        #         img = img[:, :, ::-1]
        return {'img':img, 'label':0}


class MotoMaskDataset(Dataset):
    def __init__(self, img_list, load_in_mem, augmentor=None):
        self.img_list  = img_list
        self.augmentor = augmentor
        self.load_in_mem = load_in_mem
        if self.load_in_mem:
            print('Load in mem...')
            self.data = multi_thread(self.data_preprocessing, img_list, verbose=1)
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,idx):
        if self.load_in_mem:
            img = self.data[idx]
        else:
            full_img_path = self.img_list[idx]
            img = data_preprocessing(full_img_path)

        if self.augmentor is not None:
            img = self.augmentor(img)
        return {'img':img, 'label':0}

    def data_preprocessing(self, img_path):
        _img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = _img[:,:,3]/255.0
        mask = np.stack([mask]*3, axis=-1)
        img = _img[:,:,:3]*mask
        img = img.astype('uint8')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bg_color = detect_background_color(img)
        img = pad_if_needed(img, bg_color)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 127.5 - 1
        img = img.transpose([2,0,1])
        return img.astype(np.float32) 

def get_data_loaders(data_root=None, label_root=None, batch_size=32, num_workers=16, shuffle=True,
                     pin_memory=True, drop_last=True, load_in_mem=False, mask_out=True):
    print('Using dataset root location %s' % data_root)
    # train_set = DogsDataSet(data_root, label_root, create_runtime_tfms())
    img_paths = glob.glob(f'{data_root}/*.*')
    if mask_out:
        train_set = MotoMaskDataset(img_paths, load_in_mem=load_in_mem)
    else:
        train_set = MotoDataset(img_paths, load_in_mem=load_in_mem)
    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last}  # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders