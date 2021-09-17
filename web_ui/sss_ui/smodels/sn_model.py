import torch
from .base_model import BaseModel
from . import networks
from PIL import Image
from ..misc import get_params1, get_transform1
import random


# Seams Network model
class SNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.more_samples:
            self.loss_names = ['G_GAN_blend', 'D_blend', 'D_blend_1', 'G_L1_blend', 'ml', 'G_L2', 'G2', 'G2_m']
        else:
            self.loss_names = ['G_GAN_blend', 'D_blend', 'G_L1_blend', 'ml', 'G_L2', 'G2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.forced_mask:
            # self.visual_names = ['real_A_map', 'real_A_ins', 'fake_B_encoded_0', 'real_B_encoded', 'fake_mask', 'forced_mask','fake_B_encoded', 'fake_B_encoded_blend', 'fake_t_g', 'fake_C_encoded', 'fake_t_d', 'fake_C_pred_1', 'C_t']
            self.visual_names = ['fake_B_encoded_0', 'real_B_encoded', 'fake_mask', 'forced_mask','fake_B_encoded', 'fake_B_encoded_blend', 'fake_t_g', 'fake_C_encoded', 'fake_t_d', 'fake_C_pred_1', 'C_t']
        else:
            # self.visual_names = ['real_A_map', 'real_A_ins', 'fake_B_encoded_0', 'real_B_encoded', 'fake_mask','fake_B_encoded', 'fake_B_encoded_blend', 'fake_t_g', 'fake_C_encoded', 'fake_t_d', 'fake_C_pred_1', 'C_t']
            self.visual_names = ['fake_B_encoded_0', 'real_B_encoded', 'fake_mask','fake_B_encoded', 'fake_B_encoded_blend', 'fake_t_g', 'fake_C_encoded', 'fake_t_d', 'fake_C_pred_1', 'C_t']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names += ['G']
        if self.opt.g1_mask:
            opt.input_nc += 1
        self.netG = networks.define_G(opt.input_nc, 1, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        # self.model_names += ['G2']
        # self.netG2 = networks.define_G(3, 1, opt.nz, 32, netG="unet_128",
        #                               norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
        #                               gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        # D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        # self.model_names += ['D']
        # if opt.conD:
        #     self.netD = networks.define_D(D_output_nc + 1, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
        #                                   init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        # else:
        #     self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
        #                                   init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G2)

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
        
        # TODO: use parameter to control mask path
        mask_path = './sss_ui/mask/T.png'
        self.loss_mask = Image.open(mask_path)
        transform_params = get_params1(self.opt, self.loss_mask.size)
        mask_transform = get_transform1(self.opt, transform_params, grayscale=True)
        self.loss_mask = mask_transform(self.loss_mask)
        self.loss_mask = self.loss_mask.expand(1, self.loss_mask.size()[0], self.loss_mask.size()[1], self.loss_mask.size()[2]).to(self.device)

        c_path = './sss_ui/mask/c.png'
        self.c_mask = Image.open(c_path)
        self.c_mask = mask_transform(self.c_mask)
        self.real_C_encoded = self.c_mask.expand(1, self.c_mask.size()[0], self.c_mask.size()[1], self.c_mask.size()[2]).to(self.device)

        w_path = './sss_ui/mask/white.png'
        self.w_mask = Image.open(w_path)
        self.w_mask = mask_transform(self.w_mask)
        self.real_W_encoded = self.w_mask.expand(1, self.w_mask.size()[0], self.w_mask.size()[1], self.w_mask.size()[2]).to(self.device)

        tfg1_path = './sss_ui/mask/TfG1.png'
        self.tfg1_mask = Image.open(tfg1_path)
        self.tfg1_mask = mask_transform(self.tfg1_mask)
        self.real_TfG1_encoded = self.tfg1_mask.expand(1, self.tfg1_mask.size()[0], self.tfg1_mask.size()[1], self.tfg1_mask.size()[2]).to(self.device)


    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B = self.real_A
        # self.real_G = input['G'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        z = torch.zeros((batch_size, nz))
        return z.to(self.device)

    def get_z_zero(self, batch_size, nz, random_type='gauss'):
        z = torch.zeros((batch_size, nz))
        return z.to(self.device)

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            z0 = self.get_z_zero(self.real_A.size(0), self.opt.nz)

            self.fake_B_encoded_0 = self.real_A[:, self.real_A.size()[1]-6:self.real_A.size()[1]-3, :, :]
            self.fake_B_encoded = self.real_A[:, self.real_A.size()[1]-3:self.real_A.size()[1], :, :]
            if self.opt.seams_map:
                self.real_A_map = self.real_A[:, 0:1, :, :]
                self.real_A_ins = self.real_A[:, 1:2, :, :]

            self.m = self.netG(self.real_A, z0)
            mask = (self.m + 1.0) / 2.0
            if self.opt.forced_mask:
                mask = mask * (1-(self.loss_mask + 1.0) / 2.0)
                mask = mask + (self.loss_mask + 1.0) / 2.0
                forced_mask = mask * 2.0 - 1.0

            self.fake_B_blend = self.fake_B_encoded * (1 - mask) + self.fake_B_encoded_0 * mask

            if self.opt.forced_mask:
                return self.real_A, self.fake_B_encoded_0, self.fake_B_encoded, self.m, self.fake_B_blend, self.real_B, forced_mask
            else:
                return self.real_A, self.fake_B_encoded_0, self.fake_B_encoded, self.m, self.fake_B_blend, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A
        self.real_B_encoded = self.real_B
        self.fake_B_encoded_0 = self.real_A[:, self.real_A.size()[1]-6:self.real_A.size()[1]-3, :, :]
        self.fake_B_encoded = self.real_A[:, self.real_A.size()[1]-3:self.real_A.size()[1], :, :]
        if self.opt.seams_map:
                self.real_A_map = self.real_A[:, 0:1, :, :]
                self.real_A_ins = self.real_A[:, 1:2, :, :]
        
        # self.ls_encoded = self.real_G[:, 0:1, :, :]
        # self.s_encoded = self.real_G[:, 1:4, :, :]
        # self.n_encoded = self.real_G[:, 4:7, :, :]

        # get z
        self.z = self.get_z_zero(self.real_A_encoded.size(0), self.opt.nz)

        if self.opt.g1_mask:
            self.real_A_encoded = torch.cat((self.real_A_encoded, self.real_TfG1_encoded), 1)
        self.fake_mask = self.netG(self.real_A_encoded, self.z)
        self.mask = (self.fake_mask + 1.0) / 2.0
        if self.opt.forced_mask:
            self.mask = self.mask * (1-(self.loss_mask + 1.0) / 2.0)
            self.mask = self.mask + (self.loss_mask + 1.0) / 2.0
            self.forced_mask = self.mask * 2.0 - 1.0

        self.fake_B_encoded_blend = self.fake_B_encoded * (1 - self.mask) + self.fake_B_encoded_0 * self.mask

    def backward_D(self, netD, real, fake):
        if self.opt.conD:
            # Fake, stop backprop to the generator by detaching fake_B
            pred_fake = netD(torch.cat((fake.detach(), self.loss_mask), 1))
            # real
            pred_real = netD(torch.cat((real, self.loss_mask), 1))
        else:
            # Fake, stop backprop to the generator by detaching fake_B
            pred_fake = netD(fake.detach())
            # real
            pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            if self.opt.conD:
                pred_fake = netD(torch.cat((fake, self.loss_mask), 1))
            else:
                pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN_blend = self.backward_G_GAN(self.fake_B_encoded_blend, self.netD, self.opt.lambda_GAN)
        # 2, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1_blend = self.criterionL1(self.fake_B_encoded_blend, self.real_B_encoded) * self.opt.lambda_L1
            # self.loss_G_L1_blend = self.criterionL1(self.fake_B_encoded_blend, self.fake_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1_blend = 0.0
        # 3, mask loss
        self.tm = (self.fake_mask + 1)/2.0 * (self.loss_mask + 1)/2.0
        self.tm = self.tm * 2.0 - 1.0
        self.loss_ml = self.criterionL2(self.tm, self.loss_mask) * self.opt.lambda_ml
        # 4, 2th generator loss
        self.loss_G_L2 = self.backward_G_G2(self.fake_B_encoded_blend)

        self.loss_G = self.loss_G_GAN_blend + self.loss_G_L1_blend + self.loss_ml + self.loss_G_L2
        self.loss_G.backward(retain_graph=True)

    # TODO: add map data
    def backward_G_G2(self, fake):
        loss_G_L2 = 0.0
        for i in range(0, 4):
            x = random.randint(0, 128)
            y = random.randint(0, 128)
            self.fake_t_g = fake[:, :, x:x+128, y:y+128]
            W_t = self.real_W_encoded[:, :, x:x+128, y:y+128]

            self.fake_C_encoded = self.netG2(self.fake_t_g, self.z)
            if self.opt.lambda_t1 > 0.0:
                loss_G_L2 += self.criterionL2(self.fake_C_encoded, W_t) * self.opt.lambda_t1
            else:
                loss_G_L2 += 0.0
        
        return loss_G_L2
        
    # TODO: add map data
    def backward_G2(self, fake):
        loss_G2_L2_fake_1 = 0.0
        for i in range(0, 4):
            x = random.randint(0, 128)
            y = random.randint(0, 128)
            self.fake_t_d = fake[:, :, x:x+128, y:y+128]
            self.C_t = self.real_C_encoded[:, :, x:x+128, y:y+128]
        
            self.fake_C_pred_1 = self.netG2(self.fake_t_d.detach(), self.z)
            if self.opt.lambda_t2 > 0.0:
                loss_G2_L2_fake_1 += self.criterionL2(self.fake_C_pred_1, self.C_t) * self.opt.lambda_t2
            else:
                loss_G2_L2_fake_1 += 0.0

        loss_G2_L2 = loss_G2_L2_fake_1
        loss_G2_L2.backward()

        return loss_G2_L2

    def update_D(self):
        self.set_requires_grad([self.netD], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            # self.loss_D_blend, self.losses_D_blend = self.backward_D(self.netD, self.real_data_encoded, self.fake_B_encoded_blend)
            self.loss_D_blend, self.losses_D_blend = self.backward_D(self.netD, self.fake_B_encoded, self.fake_B_encoded_blend)
            self.optimizer_D.step()
        if self.opt.more_samples:
            self.optimizer_D.zero_grad()
            self.loss_D_blend_1, self.losses_D_blend_1 = self.backward_D(self.netD, self.fake_B_encoded, self.fake_B_encoded_0)
            self.optimizer_D.step()

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()

    def update_G2(self):
        self.set_requires_grad([self.netG2], True)
        self.optimizer_G2.zero_grad()
        self.loss_G2 = self.backward_G2(self.fake_B_encoded_blend)
        self.optimizer_G2.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_G2()
        self.update_D()
        # print(' ------ optimize_parameters once ------')