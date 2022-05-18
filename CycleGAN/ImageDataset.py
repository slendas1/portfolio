import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

path_monet = '/home/hulk/slendas/DL/kaggle_gan/gan-getting-started/monet_jpg'
path_photo = '/home/hulk/slendas/DL/kaggle_gan/gan-getting-started/photo_jpg'


class ImageDataset(Dataset):

    def __init__(self, type_of_img, img_size=256, normalize=True):
        if (type_of_img == 'Monet'):
            self.img_path = path_monet
        else:
            self.img_path = path_photo

        if (normalize):
            self.transform = transforms.Compose([transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])])
        else:
            self.transform = transforms.Compose([transforms.Resize(img_size),
                                                 transforms.ToTensor()])
        self.imgs = dict()
        for ind, img in enumerate(os.listdir(self.img_path)):
            self.imgs[ind] = img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        img_path = os.path.join(self.img_path, self.imgs[ind])
        img = Image.open(img_path)
        img = self.transform(img)

        return img
