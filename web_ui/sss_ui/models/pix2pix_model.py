"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from . import networks as networks
from ..misc import load_network
import torch.nn.functional as F


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, data_i, mode, z=None):
        input_semantics = self.preprocess_input(data, data_i)

        if mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics)
            return fake_image
        elif mode == 'random_z':
            with torch.no_grad():
                fake_images = self.generate_fake1(input_semantics, z=z)
            return fake_images
        else:
            raise ValueError("|mode| is invalid")

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data, data_i):
        # move to GPU and change data types
        data = data.long()
        data_i = data_i.long()

        # create one-hot label map
        label_map = data
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map
        self.instance_edge_map = self.FloatTensor(bs, 1, h, w).zero_()
        inst_map = data_i
        if self.opt.ins_edge:
            instance_edge_map = self.get_edges1(inst_map)
        else:
            instance_edge_map = self.get_edges(inst_map)
        self.instance_edge_map = instance_edge_map
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image=None, compute_kld_loss=False, cg=None):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    def generate_fake1(self, input_semantics, real_image=None, z=None):
        fake_image = self.netG(input_semantics, z=z)

        return fake_image

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def get_edges1(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        one = torch.tensor(1, dtype=torch.int8)
        zero = torch.tensor(0, dtype=torch.int8)
        if self.use_gpu():
            one = one.cuda()
            zero = zero.cuda()
        edge = torch.where(t==self.opt.label_nc+1, one, zero)
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
