import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


# Seams Network dataset mode
class SNDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths, self.I_paths, self.B_paths, self.C_paths, self.D_paths = self.get_paths(opt)
        if opt.serial_batches:
            self.A_paths.sort()
            self.I_paths.sort()
            self.B_paths.sort()
            self.C_paths.sort()
            self.D_paths.sort()

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        A = A.convert('L')
        I_path = self.I_paths[index]
        I = Image.open(I_path)
        I = I.convert('L')
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        C_path = self.C_paths[index]
        C = Image.open(C_path)
        D_path = self.D_paths[index]
        D = Image.open(D_path)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        I = A_transform(I)
        B = B_transform(B)
        C = B_transform(C)
        D = B_transform(D)

        if self.opt.seams_map:
            A = torch.cat((A, I), 0)
            A = torch.cat((A, C), 0)
            A = torch.cat((A, D), 0)
        else:
            A = torch.cat((C, D), 0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def get_paths(self, opt):
        A_dir = os.path.join(opt.dataroot, "map")
        if opt.phase == 'train':
            A_dir = os.path.join(A_dir, 'training')
        else:
            A_dir = os.path.join(A_dir, 'val')

        A_paths = make_dataset(A_dir, opt.max_dataset_size)

        I_dir = A_dir.replace("/map", "/ins")
        I_paths = make_dataset(I_dir, opt.max_dataset_size)

        B_dir = A_dir.replace("/map", "/real")
        B_paths = make_dataset(B_dir, opt.max_dataset_size)

        assert len(A_paths) == len(B_paths), "The #images in %s and %s do not match. Is there something wrong?"

        C_dir = A_dir.replace("/map", "/s")
        C_paths = make_dataset(C_dir, opt.max_dataset_size)

        D_dir = A_dir.replace("/map", "/ns")
        D_paths = make_dataset(D_dir, opt.max_dataset_size)

        return A_paths, I_paths, B_paths, C_paths, D_paths