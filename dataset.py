from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

class NPY_datasets(Dataset):
    def __init__(self, config, train=True, path_Data=None):
        super(NPY_datasets, self).__init__()
        self.status = train

        # Use default dataset path if not provided
        if path_Data is None:
            path_Data = '/content/drive/MyDrive/KaggleDatasets/ISIC_Datasets/ISIC_2018_Dataset/'

        # Set subfolder path
        base_dir = os.path.join(path_Data, 'train' if train else 'val')
        images_dir = os.path.join(base_dir, 'images')
        masks_dir = os.path.join(base_dir, 'masks')

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Image folder not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Mask folder not found: {masks_dir}")

        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))

        self.data = []

        if train:
            points_dir = os.path.join(base_dir, 'points_boundary2')
            if not os.path.exists(points_dir):
                raise FileNotFoundError(f"Points folder not found: {points_dir}")
            points_list = sorted(os.listdir(points_dir))
            for i in range(len(images_list)):
                img_path = os.path.join(images_dir, images_list[i])
                mask_path = os.path.join(masks_dir, masks_list[i])
                point_path = os.path.join(points_dir, points_list[i])
                self.data.append([img_path, mask_path, point_path])
            self.transformer = config.train_transformer
        else:
            for i in range(len(images_list)):
                img_path = os.path.join(images_dir, images_list[i])
                mask_path = os.path.join(masks_dir, masks_list[i])
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        if self.status:
            img_path, msk_path, pnt_path = self.data[indx]
            img = np.array(Image.open(img_path).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
            pnt = np.expand_dims(np.array(Image.open(pnt_path).convert('L')), axis=2) / 255
            img, msk, pnt = self.transformer((img, msk, pnt))
            return img, msk, pnt
        else:
            img_path, msk_path = self.data[indx]
            img = np.array(Image.open(img_path).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
            img, msk = self.transformer((img, msk))
            return img, msk

    def __len__(self):
        return len(self.data)


class Test_datasets(Dataset):
    def __init__(self, config, path_Data=None):
        super(Test_datasets, self).__init__()

        if path_Data is None:
            path_Data = '/content/drive/MyDrive/KaggleDatasets/ISIC_Datasets/ISIC_2018_Dataset/'

        base_dir = os.path.join(path_Data, 'test')
        images_dir = os.path.join(base_dir, 'images')
        masks_dir = os.path.join(base_dir, 'masks')

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Test image folder not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Test mask folder not found: {masks_dir}")

        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))

        self.data = []
        for i in range(len(images_list)):
            img_path = os.path.join(images_dir, images_list[i])
            mask_path = os.path.join(masks_dir, masks_list[i])
            self.data.append([img_path, mask_path])

        self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
