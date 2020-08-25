from torchvision.datasets.folder import *
import numpy as np
import ipdb


class ImageFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, return_path=False, split=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        
        if split:
            assert "/" in split
            i, j = split.split("/")
            N = len(self.samples)
            idx = np.array_split(np.arange(N), int(j))[int(i)-1]
            self.samples = [self.samples[i] for i in idx]

        self.imgs = self.samples
        self.idx_to_class = {v:k for k, v in self.class_to_idx.items()}
        self.return_path = return_path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return path, sample, target
        else:
            return sample, target

