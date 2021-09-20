"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image, ImageFilter
import util.util as util
import os
from data.image_folder import make_dataset
import copy
import numpy as np
import skimage
import random


class INSGBDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.optCG = copy.deepcopy(opt)
        self.optCG.load_size = opt.cg_size
        self.optCG.crop_size = opt.cg_size

        label_paths, image_paths, instance_paths, cg_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)
        util.natural_sort(cg_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.cg_paths = cg_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        cg_dir = image_dir.replace("real", "guidance")
        cg_paths = make_dataset(cg_dir, recursive=False, read_cache=True)

        return label_paths, image_paths, instance_paths, cg_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        # FIXME: convert('L') is due to current label image is 3-channel
        label = label.convert('L')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # color guidance
        if self.opt.phase == 'train':
            cg_path = self.cg_paths[index]
        else:
            # cg_path = self.cg_paths[index + 3000]
            # for nz3 dataset
            # cg_path = self.cg_paths[index + 1100]
            # for tc (test color guidance) dataset
            cg_path = self.cg_paths[index]
        assert self.paths_match(image_path, cg_path), \
            "The cg_path %s and image_path %s don't match." % \
            (image_path, cg_path)
        cg = Image.open(cg_path)
        cg = cg.convert('RGB')
        cg = cg.resize((256, 256), Image.LANCZOS)
        if self.opt.gbk_size > 0:
            cg = cg.filter(ImageFilter.GaussianBlur(radius = self.opt.gbk_size))
        transform_cg = get_transform(self.optCG, params)
        cg_tensor = transform_cg(cg)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            # FIXME: convert('L') is due to current label image is 3-channel
            instance = instance.convert('L')
            instance_tensor = transform_label(instance) * 255
            instance_tensor = instance_tensor.long()

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'cg': cg_tensor,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
