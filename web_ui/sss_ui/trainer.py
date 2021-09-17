"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

from .options.test_options import TestOptions
from .models.pix2pix_model import Pix2PixModel
from .models.cg_model import CGModel
import torch

from .soptions.test_options import STestOptions
from .sdata import create_dataset
from .smodels import create_model
from .sutil.visualizer import save_images
from itertools import islice
from .sutil import html

def init_model(name):
    # print("enter!")
    # print(TestOptions())
    opt = TestOptions().parse()
    opt.name = name
    opt.load_size = 256
    opt.crop_size = 256
    opt.label_nc = 13
    # opt.no_instance = True
    opt.use_vae = True
    opt.ins_edge = True
    opt.no_flip = True
    opt.random_z = True
    opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)
    model = Pix2PixModel(opt)
    model.eval()
    # print("initialized!")
    # print(next(model.parameters()).device)
    return model, opt

def init_model1(name):
    # print("enter!")
    # print(TestOptions())
    opt = TestOptions().parse()
    opt.name = name
    opt.load_size = 256
    opt.label_nc = 13
    # opt.no_instance = True
    opt.ins_edge = True
    opt.cg = True
    opt.netG = "spadebranchn"
    opt.cg_size = 256
    opt.gbk_size = 8
    opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)
    model = CGModel(opt)
    model.eval()
    # print("initialized!")
    # print(next(model.parameters()).device)
    return model, opt

def init_model2(name):
    opt = STestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle

    opt.name = name
    opt.model = "sn"
    opt.load_size = 256
    # opt.dataset_mode = "sn"
    opt.input_nc = 8
    opt.seam_map = True
    opt.no_flip = True
    opt.ndf = 32
    opt.conD = True
    opt.forced_mask = True

    # create dataset
    # dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    # print('Loading model %s' % opt.model)

    return model, opt