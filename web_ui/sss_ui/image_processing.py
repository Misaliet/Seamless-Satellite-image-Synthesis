from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
import os
from .misc import get_params, get_transform, get_numpy, get_image, get_params1, get_transform1
import torch

BASE_DIR = Path(__file__).resolve(strict=True).parent

target_img_size = 512
single_img_size = 256
zoom_level = 4
half = 2

buffer_multiple = 2

class imageProcessing:

    def __init__(self):
        # print(os.getcwd())
        # print(BASE_DIR)
        self.path_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A.png")
        self.path_B = os.path.join(BASE_DIR, "static", "runtime/images/B/B.png")
        self.buff_A = None
        self.buff_B = None
        self.path_buffer_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b.png")
        self.path_buffer_B = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b.png")
        self.path_buffer_B_s = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_s.png")
        self.path_buffer_B_ns = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_ns.png")
        self.path_buffer_B_2 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_2.png")
        self.path_buffer_B_3 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_3.png")

        self.path_label_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_l.png")
        self.buff_label_A = None
        self.path_buffer_label_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b_l.png")

        self.path_ins_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_i.png")
        self.buff_ins_A = None
        self.path_buffer_ins_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b_i.png")

        self.path_cg_2 = os.path.join(BASE_DIR, "static", "runtime/images/A/cg_2.png")
        self.path_cg_3 = os.path.join(BASE_DIR, "static", "runtime/images/A/cg_3.png")
        

    def setup(self):
        if os.path.exists(self.path_A):
            os.remove(self.path_A)
        if os.path.exists(self.path_B):
            os.remove(self.path_B)

        target_A = Image.new('RGB',(target_img_size, target_img_size))
        target_B = Image.new('RGB',(target_img_size, target_img_size))
        self.save(target_A, target_B)
        try:
            img_temp = Image.open(self.path_buffer_B)
        except FileNotFoundError:
            img_temp = Image.new('RGB',(1024, 1024))
            self.save_img(img_temp, self.path_buffer_B)
        self.buff_B = img_temp
    
    def save_img(self, img, path):
        img.save(path)

    def save(self, img_A, img_B):
        img_A.save(self.path_A)
        img_B.save(self.path_B)

    def save_cg(self, x, y, z):
        img = self.path_buffer_B.crop((y, x, y+256, x+256))
        if z == 2:
            cg_path = self.path_cg_2
        else:
            cg_path = self.path_cg_3
        img.save(cg_path)
    
    # for test (skip initial generation)
    def load_buffer_B(self):
        self.buff_B = Image.open(self.path_buffer_B)

    def zoom_in_B(self, x, y, z):
        target_B = Image.open(self.path_B)
        target_B = target_B.resize((2048, 2048), Image.LANCZOS)
        if z == 2:
            cg_path = self.path_cg_2
        else:
            cg_path = self.path_cg_3
        interval = int(target_B.size[0]/4)
        img_temp = target_B.crop((interval, interval, interval*3, interval*3))
        self.save_img(img_temp, cg_path)

        target_B = target_B.crop((y*zoom_level, x*zoom_level, y*zoom_level+512, x*zoom_level+512))

        self.save_img(target_B, self.path_B)
        self.buff_B = img_temp

    def zoom_out_B(self, z):
        if z == 2:
            self.buff_B = Image.open(self.path_buffer_B)
        else:
            # z == 3
            self.buff_B = Image.open(self.path_buffer_B_2)

    def zoom_out_z1(self, x, y, img_A_paths, img_B_paths):
        return 0

    def zoom_out_z2(self, x, y, img_A_paths, img_B_paths):
        return 0

    def refresh_A(self, x, y):
        target_A = self.buff_A.crop((y, x, y+512, x+512))
        self.save_img(target_A, self.path_A)

        target_label_A = self.buff_label_A.crop((y, x, y+512, x+512))
        self.save_img(target_label_A, self.path_label_A)

        target_ins_A = self.buff_ins_A.crop((y, x, y+512, x+512))
        self.save_img(target_ins_A, self.path_ins_A)

    def refresh_buffer_A(self, z, img_A_paths):
        img_A_paths.sort()

        target_A = Image.new('RGB',(target_img_size*buffer_multiple, target_img_size*buffer_multiple))
        imgs_A = []

        for p in img_A_paths:
            try:
                img_temp = Image.open(str(BASE_DIR) + "/static/runtime/images/sA/z" + str(z) + '/' + str(p).zfill(5) + ".png")
            except FileNotFoundError:
                img_temp = Image.new('RGB',(single_img_size, single_img_size))
            imgs_A.append(img_temp)
        
        target_A = self.img_paste(target_A, imgs_A)
        self.buff_A = target_A
        self.save_img(target_A, self.path_buffer_A)

        # handle label
        target_label_A = Image.new('RGB',(target_img_size*buffer_multiple, target_img_size*buffer_multiple))
        imgs_label_A = []

        for p in img_A_paths:
            try:
                img_temp = Image.open(str(BASE_DIR) + "/static/runtime/images/sAL/z" + str(z) + '/' + str(p).zfill(5) + ".png")
            except FileNotFoundError:
                img_temp = Image.new('RGB',(single_img_size, single_img_size))
            imgs_label_A.append(img_temp)
        
        target_label_A = self.img_paste(target_label_A, imgs_label_A)
        self.buff_label_A = target_label_A
        self.save_img(target_label_A, self.path_buffer_label_A)

        # handle ins
        target_ins_A = Image.new('RGB',(target_img_size*buffer_multiple, target_img_size*buffer_multiple))
        imgs_ins_A = []

        for p in img_A_paths:
            try:
                img_temp = Image.open(str(BASE_DIR) + "/static/runtime/images/sAI/z" + str(z) + '/' + str(p).zfill(5) + ".png")
            except FileNotFoundError:
                img_temp = Image.new('RGB',(single_img_size, single_img_size))
            imgs_ins_A.append(img_temp)
        
        target_ins_A = self.img_paste(target_ins_A, imgs_ins_A)
        self.buff_ins_A = target_ins_A
        self.save_img(target_ins_A, self.path_buffer_ins_A)

    def refresh_B(self, x, y):
        target_B = self.buff_B.crop((y, x, y+512, x+512))

        self.save_img(target_B, self.path_B)

    # def refresh_buffer_B(self, z, img_B_paths):
        # img_B_paths.sort()

        # target_B = Image.new('RGB',(target_img_size*buffer_multiple, target_img_size*buffer_multiple))
        # imgs_B = []

        # for p in img_B_paths:
        #     try:
        #         img_temp = Image.open(str(BASE_DIR) + "/static/runtime/images/dB/z" + str(z) + '/' + str(p).zfill(5) + ".png")
        #     except FileNotFoundError:
        #         img_temp = Image.new('RGB',(single_img_size, single_img_size))
        #     imgs_B.append(img_temp)
        
        # target_B = self.img_paste(target_B, imgs_B)
        # self.buff_B = target_B
        # self.save_buffer_B(target_B)

    def im2tensor(self, imgs, transform, nc):
        tensors = []
        for i in imgs:
            tensor = transform(i) * 255.0
            tensor[tensor == 255] = nc
            tensor = tensor.expand(1, tensor.size()[0], tensor.size()[1], tensor.size()[2])
            tensors.append(tensor)
        
        return tensors
    
    def im2tensor_c(self, imgs, imgs_i, imgs_s, imgs_ns, transform1, transform2):
        tensors = []
        for i, ii, s, ns in zip(imgs, imgs_i, imgs_s, imgs_ns):
            it = transform1(i)
            iit = transform1(ii)
            st = transform2(s)
            nst = transform2(ns)
            ft = torch.cat((it, iit), 0)
            ft = torch.cat((ft, st), 0)
            ft = torch.cat((ft, nst), 0)
            ft = ft.expand(1, ft.size()[0], ft.size()[1], ft.size()[2])
            tensors.append(ft)
        
        return tensors

    def im2tensor_i(self, imgs, transform, gbk_size):
        tensors = []
        for i in imgs:
            i = i.filter(ImageFilter.GaussianBlur(radius = gbk_size))
            tensor = transform(i)
            tensor = tensor.expand(1, tensor.size()[0], tensor.size()[1], tensor.size()[2])
            tensors.append(tensor)
        
        return tensors
        
    def generate(self, imgs, imgs_i, model, opt, z_sample):
        params = get_params(opt, (single_img_size, single_img_size))
        transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        
        label_tensors = self.im2tensor(imgs, transform_label, opt.label_nc)
        ins_tensors = self.im2tensor(imgs_i, transform_label, opt.label_nc)
        
        output_tensors = []
        for l, li in zip(label_tensors, ins_tensors):
            if opt.gpu_ids != []:
                l = l.cuda()
                li = li.cuda()
            generated = model(l, li, mode='random_z', z=z_sample)
            output_tensors.append(generated)
        # print(output_tensors)
        output_tensor = []
        for o in output_tensors:
            for b in range(o.shape[0]):
                output_tensor.append(o[b])
        output_imgs = []
        for o in output_tensor:
            n = get_numpy(o)
            img = get_image(n)
            output_imgs.append(img)

        # target_B = Image.new('RGB',(single_img_size*4, single_img_size*4))
        s = int(pow(len(imgs), 0.5))
        target_B = Image.new('RGB',(single_img_size*s, single_img_size*s))
        self.img_paste(target_B, output_imgs)
        
        return target_B

    def generate_seamless(self, imgs, imgs_i, imgs_s, imgs_ns, smodel, sopt):
        transform_params = get_params1(sopt, (single_img_size, single_img_size))
        A_transform = get_transform1(sopt, transform_params, grayscale=True)
        B_transform = get_transform1(sopt, transform_params, grayscale=False)
        
        input_tensors = self.im2tensor_c(imgs, imgs_i, imgs_s, imgs_ns, A_transform, B_transform)
        
        output_tensors = []
        for t in input_tensors:
            if sopt.gpu_ids != []:
                t = t.cuda()
            smodel.set_input(t)
            z_sample = smodel.get_z_zero(1, sopt.nz)
            _, _, _, _, blend_B, _, _ = smodel.test(z_sample, encode=False)
            # print(blend_B.shape)
            output_tensors.append(blend_B)
        # print(output_tensors)
        output_tensor = []
        for o in output_tensors:
            for b in range(o.shape[0]):
                output_tensor.append(o[b])
        output_imgs = []
        for o in output_tensor:
            n = get_numpy(o)
            img = get_image(n)
            output_imgs.append(img)

        s = int(pow(len(imgs), 0.5))
        target_B = Image.new('RGB',(single_img_size*s, single_img_size*s))
        self.img_paste(target_B, output_imgs)
        
        return target_B

    def refresh_buffer_B_s(self, model, opt, z_sample):
        source_A = Image.open(self.path_buffer_label_A)
        source_A = source_A.convert('L')
        imgs = self.image_cut(source_A)

        source_I = Image.open(self.path_buffer_ins_A)
        source_I = source_I.convert('L')
        imgs_i = self.image_cut(source_I)
        
        target_B = self.generate(imgs, imgs_i, model, opt, z_sample)

        # self.buff_B = target_B
        self.save_img(target_B, self.path_buffer_B_s)

    def refresh_buffer_B_ns(self, model, opt, z_sample):
        source_A = Image.open(self.path_buffer_label_A)
        source_A = source_A.convert('L')
        source_A = self.image_crop(source_A)
        imgs = self.image_cut(source_A)

        source_I = Image.open(self.path_buffer_ins_A)
        source_I = source_I.convert('L')
        source_I = self.image_crop(source_I)
        imgs_i = self.image_cut(source_I)
        
        target_B = self.generate(imgs, imgs_i, model, opt, z_sample)

        # self.buff_B = target_B
        self.save_img(target_B, self.path_buffer_B_ns)
    
    def remove_seams(self, smodel, sopt):
        source_A = Image.open(self.path_buffer_label_A)
        source_A = source_A.convert('L')
        source_A = self.image_crop(source_A)
        imgs = self.image_cut(source_A)

        source_I = Image.open(self.path_buffer_ins_A)
        source_I = source_I.convert('L')
        source_I = self.image_crop(source_I)
        imgs_i = self.image_cut(source_I)

        # s
        source_s = Image.open(self.path_buffer_B_s)
        source_s = self.image_crop(source_s)
        imgs_s = self.image_cut(source_s)

        # ns
        source_ns = Image.open(self.path_buffer_B_ns)
        imgs_ns = self.image_cut(source_ns)

        target_B = self.generate_seamless(imgs, imgs_i, imgs_s, imgs_ns, smodel, sopt)
        self.save_img(target_B, os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_ns_f.png"))

        return target_B

    def refresh_buffer_B(self, model, opt, smodel, sopt, z_sample):
        self.refresh_buffer_B_s(model, opt, z_sample)
        self.refresh_buffer_B_ns(model, opt, z_sample)
        target_B = self.remove_seams(smodel, sopt)
        target_B_s = Image.open(self.path_buffer_B_s)
        target_B = self.img_paste_whole(target_B_s, target_B)
        self.buff_B = target_B
        self.save_img(target_B, self.path_buffer_B)

    def img_paste(self, img, imgs):
        w, h = img.size
        i = int(h/single_img_size)
        j = int(w/single_img_size)
        index = 0
        for ii in range(0, i):
            for jj in range(0, j):
                img.paste(imgs[index], (single_img_size*jj, single_img_size*ii))
                index += 1
        return img

    def random_A(self, r_level, r_index, r_x, r_y):
        target_A = Image.open(self.path_A)
        r_A = Image.open(str(BASE_DIR) + "/static/runtime/images/sA/z" + str(r_level) + '/' + str(r_index).zfill(5) + ".png")
        # FIXME: replace magic number
        target_A.paste(r_A, (128, 128))

        self.save_img(target_A, self.path_A)

    def transfer_generate(self, imgs, imgs_i, imgs_cg, model, opt):
        params = get_params(opt, (single_img_size, single_img_size))
        transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        
        label_tensors = self.im2tensor(imgs, transform_label, opt.label_nc)
        ins_tensors = self.im2tensor(imgs_i, transform_label, opt.label_nc)

        transform_image = get_transform(opt, params)
        cg_tensors = self.im2tensor_i(imgs_cg, transform_image, opt.gbk_size)
        
        output_tensors = []
        for l, li, cg in zip(label_tensors, ins_tensors, cg_tensors):
            if opt.gpu_ids != []:
                l = l.cuda()
                li = li.cuda()
                cg = cg.cuda()
            generated = model(l, li, cg, mode='inference')
            output_tensors.append(generated)
        # print(output_tensors)
        output_tensor = []
        for o in output_tensors:
            for b in range(o.shape[0]):
                output_tensor.append(o[b])
        output_imgs = []
        for o in output_tensor:
            n = get_numpy(o)
            img = get_image(n)
            output_imgs.append(img)

        # target_B = Image.new('RGB',(single_img_size*4, single_img_size*4))
        s = int(pow(len(imgs), 0.5))
        target_B = Image.new('RGB',(single_img_size*s, single_img_size*s))
        self.img_paste(target_B, output_imgs)
        
        return target_B

    def transfer_s(self, model, opt, z):
        source_A = Image.open(self.path_buffer_label_A)
        source_A = source_A.convert('L')
        imgs = self.image_cut(source_A)

        source_I = Image.open(self.path_buffer_ins_A)
        source_I = source_I.convert('L')
        imgs_i = self.image_cut(source_I)

        # CG
        if z == 2:
            cg_path = self.path_cg_2
        else:
            cg_path = self.path_cg_3
        source_cg = Image.open(cg_path)
        source_cg = source_cg.convert('RGB')
        imgs_cg = self.image_cut(source_cg)
        
        target_B = self.transfer_generate(imgs, imgs_i, imgs_cg, model, opt)
        self.save_img(target_B, self.path_buffer_B_s)
    
    def transfer_ns(self, model, opt, z):
        source_A = Image.open(self.path_buffer_label_A)
        source_A = source_A.convert('L')
        source_A = self.image_crop(source_A)
        imgs = self.image_cut(source_A)

        source_I = Image.open(self.path_buffer_ins_A)
        source_I = source_I.convert('L')
        source_I = self.image_crop(source_I)
        imgs_i = self.image_cut(source_I)

        # CG
        if z == 2:
            cg_path = self.path_cg_2
        else:
            cg_path = self.path_cg_3
        source_cg = Image.open(cg_path)
        source_cg = source_cg.convert('RGB')
        source_cg = self.image_crop(source_cg)
        imgs_cg = self.image_cut(source_cg)
        
        target_B = self.transfer_generate(imgs, imgs_i, imgs_cg, model, opt)
        self.save_img(target_B, self.path_buffer_B_ns)

    def transfer(self, model, opt, smodel, sopt, z):
        self.transfer_s(model, opt, z)
        self.transfer_ns(model, opt, z)
        target_B = self.remove_seams(smodel, sopt)
        target_B_s = Image.open(self.path_buffer_B_s)
        target_B = self.img_paste_whole(target_B_s, target_B)
        self.buff_B = target_B
        if z == 2:
            self.save_img(target_B, self.path_buffer_B_2)
        else:
            self.save_img(target_B, self.path_buffer_B_3)

    def image_cut(self, source_A):
        imgs = []
        w, h = source_A.size
        i = int(h/single_img_size)
        j = int(w/single_img_size)
        for ii in range(0, i):
            for jj in range(0, j):
                img_temp = source_A.crop((single_img_size*jj, single_img_size*ii, single_img_size+single_img_size*jj, single_img_size+single_img_size*ii))
                imgs.append(img_temp)

        return imgs

    def image_crop(self, source_A):
        cut = int(single_img_size/2)
        w, h = source_A.size
        img = source_A.crop((cut, cut, w-cut, h-cut))

        return img

    def img_paste_whole(self, img, img2):
        cut = int(single_img_size/2)
        w, h = img.size
        img.paste(img2, (cut, cut))

        return img
