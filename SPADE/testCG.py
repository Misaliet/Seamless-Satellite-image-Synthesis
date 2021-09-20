"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.cg_model import CGModel
from util.visualizer import Visualizer
from util import html

import torch

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = CGModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

if opt.random_z:
    # FIXME: magic number
    z_samples = torch.randn(3, opt.z_dim, dtype=torch.float32)

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    
    # using random z
    if opt.random_z:
        generated = model(data_i, mode='random_z', z=z_samples)
        img_path = data_i['path']
        for b in range(generated[0].shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('ground_truth', data_i['image'][b]),
                                   ('color_guidance', data_i['cg'][b]),
                                   ('synthesized_encode_image', generated[0][b]),
                                   ('synthesized_random_image1', generated[1][b]),
                                   ('synthesized_random_image2', generated[2][b]),
                                   ('synthesized_random_image3', generated[3][b])])
            if opt.ins_edge:
                if not opt.no_instance:
                    instance_edges = model(data_i, mode='instance_edges')
                    instance_edges = instance_edges.squeeze(0)
                else:
                    instance_edges = torch.zeros(data_i['label'][0].size())
                visuals.update({'ins_map': instance_edges})
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        continue
    
    # nomarl process
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('ground_truth', data_i['image'][b]),
                               ('synthesized_image', generated[b])])
        if opt.cg:
            visuals.update({'color_guidance': data_i['cg'][b]})
        if opt.ins_edge:
            if not opt.no_instance:
                instance_edges = model(data_i, mode='instance_edges')
                instance_edges = instance_edges.squeeze(0)
            else:
                instance_edges = torch.zeros(data_i['label'][0].size())
            visuals.update({'ins_map': instance_edges})
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
