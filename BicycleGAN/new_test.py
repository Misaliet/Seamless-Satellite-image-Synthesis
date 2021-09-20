import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html


# options
opt = TestOptions().parse()
opt.num_threads = 0   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    print('process input image %5.5d/%5.5d' % (i, opt.num_test))
    z_samples = model.get_z_zero(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        if opt.forced_mask:
            real_A, fake_B_0, fake_B, mask, blend_B, real_B, forced_mask = model.test(z_samples[[nn]], encode=encode)
        else:
            real_A, fake_B_0, fake_B, mask, blend_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            if opt.forced_mask:
                images = [real_B, fake_B_0, fake_B, mask, blend_B, forced_mask]
                names = ['ground truth', 'fake_0', 'fake_1', 'mask','blended', 'forced_mask']
            else:
                images = [real_B, fake_B_0, fake_B, mask, blend_B]
                names = ['ground truth', 'fake_0', 'fake_1', 'mask','blended']
        else:
            images.append(blend_B)
            names.append('random_sample%2.2d' % nn)

    img_path = 'input_%5.5d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()