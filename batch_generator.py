import os
import numpy as np

from skimage import io, img_as_float
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.color import gray2rgb

def remove_center_area(img,area_size=(30, 30)):
    h, w = img.shape[:2]
    corupted_img = img.copy()
    try:
        corupted_img[h // 2 - area_size[0] // 2:h // 2 + area_size[0] // 2,w // 2 - area_size[1] // 2:w // 2 + area_size[1] // 2, :] = 0
    except:
        print("ALERT")
        print(corupted_img.shape)
    return corupted_img
    

class BatchGenerator(Dataset):
    def __init__(self, root_dir, transform=None, test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(root_dir)
        self.filenames = list(
            map(lambda name: os.path.join(root_dir, name), self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        while(1):
            try:
                image = img_as_float(io.imread(img_name))
                break
            except:
                print(img_name)
                idx += 1
                img_name = self.filenames[idx]
        h, w = image.shape[:2]
        if len(image.shape) != 3:
            image = gray2rgb(image)
        area_size = (np.random.randint(h // 2), np.random.randint(w // 2))
        corrupted_image = remove_center_area(image, area_size)
        mask = np.zeros((h, w))
        mask[h // 2 - area_size[0] // 2:h // 2 + area_size[0] // 2,w // 2 - area_size[1] // 2:w // 2 + area_size[1] // 2] = 1
        
        mask = resize(mask, (32, 32))
       

        return corrupted_image, image, mask