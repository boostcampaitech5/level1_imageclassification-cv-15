from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import glob

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class MaskDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.df_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        self.image_dir = os.path.join(data_dir, 'images')
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*', '*'))
        self.train = train
        if self.train:
            self.labels = self.path_to_label(self.image_paths)
        self.transform = transform
    
    def rescale_age(self, age):
        return (age - self.df_data.age.min()) / (self.df_data.age.max() - self.df_data.age.min()
)
    def path_to_label(self, paths):
        labels = []
        for path in paths:
            label_num = []
            label = path.split(os.path.sep)[-2:]
            if 'incorrect' in label[1]:
                label_num.append(2)
            elif 'mask' in label[1]:
                label_num.append(1)
            else: label_num.append(0)

            gender_age = label[0].split('_')
            gender = 0 if gender_age[1] == 'female' else 1
            age = self.rescale_age(int(gender_age[-1]))
            label_num.extend([gender, age])
            labels.append(label_num)
        return labels

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)

        if self.train:
            label = self.labels[index]
            return image, label
        
        return image

    def __len__(self):
        return len(self.image_paths)


class MaskDataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((512, 384), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
        self.data_dir = data_dir
        self.dataset = MaskDataset(self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == "__main__":
    train_dataset = MaskDataset("/opt/ml/level1/mask_data/train")
    print(train_dataset[0])
    print(train_dataset[1])
    val_dataset = MaskDataset("/opt/ml/level1/mask_data/train", train=False)
    print(val_dataset[10])
    print(next(iter(MaskDataLoader("/opt/ml/level1/mask_data/train", 1))))

