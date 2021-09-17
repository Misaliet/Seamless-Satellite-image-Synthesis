"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from . import networks as networks
# import util.util as util
import torch.nn.functional as F
import copy
from ..misc import load_network


class CGModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt2 = copy.deepcopy(self.opt)
        # self.opt2.netD = "NLayer"
        self.opt2.netD = "n"

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if self.opt.doubleD:
            self.netD2 = self.initialize_networks2(self.opt2)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.doubleD:
                if self.opt2.netD == "NLayer":
                    self.criterionGAN2 = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
                else:
                    # self.criterionGAN2 = networks.GANLoss('w', tensor=self.FloatTensor, opt=self.opt)
                    self.criterionGAN2 = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if opt.lambda_c != 0:
                self.cLoss = torch.nn.MSELoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, data_i, data_cg, mode, z=None):
        input_semantics = self.preprocess_input(data, data_i, data_cg)
        if mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def create_optimizers2(self, opt):
        if opt.isTrain:
            D2_params = list(self.netD2.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            D_lr = opt.lr
        else:
            D_lr = opt.lr * 2

        optimizer_D2 = torch.optim.Adam(D2_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_D2

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

    def initialize_networks2(self, opt):
        netD2 = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            if opt.isTrain:
                netD2 = load_network(netD2, 'D2', opt.which_epoch, opt)

        return netD2

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data, data_i, data_cg):
        # move to GPU and change data types
        data = data.long()
        data_i = data_i.long()
        self.cg = data_cg

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

        input_semantics = torch.cat((input_semantics, data_cg), 1)

        return input_semantics

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if self.opt.doubleD:
            if self.opt2.netD == "NLayer":
                pred_fake_rf, _ = self.discriminate2(input_semantics, fake_image, real_image)
                G_losses['GAN_rf'] = self.criterionGAN2(pred_fake_rf, True, for_discriminator=False)
            else:
                pred_fake_rf, _ = self.discriminate3(input_semantics, fake_image, real_image)
                G_losses['GAN_rf'] = self.criterionGAN2(pred_fake_rf, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        if self.opt.lambda_c != 0:
            # FIXME: magic number
            cg_t = F.interpolate(self.cg, size=64, mode='bicubic', align_corners=True)
            fake_image_t = F.interpolate(fake_image, size=64, mode='bicubic', align_corners=True)
            G_losses['Color'] = self.cLoss(cg_t, fake_image_t) * self.opt.lambda_c

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        if self.opt.doubleD:
            if self.opt2.netD == "NLayer":
                pred_fake_rf, pred_real_rf = self.discriminate2(input_semantics, fake_image, real_image)
                D_losses['D_Fake_rf'] = self.criterionGAN2(pred_fake_rf, False, for_discriminator=True)
                D_losses['D_real_rf'] = self.criterionGAN2(pred_real_rf, True, for_discriminator=True)
            else:
                pred_fake_rf, pred_real_rf = self.discriminate3(input_semantics, fake_image, real_image)
                D_losses['D_Fake_rf'] = self.criterionGAN2(pred_fake_rf, False, for_discriminator=True)
                D_losses['D_real_rf'] = self.criterionGAN2(pred_real_rf, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, compute_kld_loss=False):
        z = None
        if self.opt.netG == 'spadebranchn':
            fake_image = self.netG(input_semantics, z=z, cg=self.cg)
        else:
            fake_image = self.netG(input_semantics, z=z)

        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        input_semantics = input_semantics[:, 0:input_semantics.size()[1]-3, :, :]
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        if self.opt.conD:
            fake_concat = torch.cat([fake_concat, self.mask], dim=1)
            real_concat = torch.cat([real_concat, self.mask], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def discriminate2(self, input_semantics, fake_image, real_image):
        input_semantics = input_semantics[:, 0:input_semantics.size()[1]-3, :, :]
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        if self.opt.conD:
            fake_concat = torch.cat([fake_concat, self.mask], dim=1)
            real_concat = torch.cat([real_concat, self.mask], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        # FIXME: magic number 70 (Receptive Field is 70)
        fake_and_real = F.interpolate(fake_and_real, size=70, mode='bicubic', align_corners=True)

        discriminator_out = self.netD2(fake_and_real)
        # a dumb way to fix list size must be 2 issue.
        discriminator_out = [discriminator_out, discriminator_out]

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def discriminate3(self, input_semantics, fake_image, real_image):
        input_semantics = input_semantics[:, 0:input_semantics.size()[1]-3, :, :]
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        if self.opt.conD:
            fake_concat = torch.cat([fake_concat, self.mask], dim=1)
            real_concat = torch.cat([real_concat, self.mask], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        # FIXME: magic number 70 (Receptive Field is 70)
        fake_and_real = F.interpolate(fake_and_real, size=70, mode='bicubic', align_corners=True)

        discriminator_out = self.netD2(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

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
