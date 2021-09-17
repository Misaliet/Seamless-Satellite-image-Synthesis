from django.shortcuts import render, HttpResponse, redirect
from django.template import Context, Template
from . import get_navi
from . import get_img_p
from . import clear
from .trainer import init_model, init_model1, init_model2
from pathlib import Path
import os
import torch

# Create your views here.
navi = get_navi()
img_p = get_img_p()
model, opt = init_model("z1")
model2, opt2 = init_model1("z2_cg")
model3, opt3 = init_model1("z3_cg")

smodel1, sopt1 = init_model2("z1sn")
smodel2, sopt2 = init_model2("z2sn")
smodel3, sopt3 = init_model2("z3sn")

z_sample = torch.randn(1, opt.z_dim, dtype=torch.float32)

BASE_DIR = Path(__file__).resolve(strict=True).parent
path_buffer_B = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b.png")
path_buffer_B_2 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_2.png")
path_buffer_B_3 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_3.png")

clear()
# img_p.setup()
# navi.setup()
# img_A_paths, _, = navi.get_imgs_index()
# x, y = navi.get_coordinate()
# z = navi.get_level()
# img_p.refresh_buffer_A(z, img_A_paths)
# img_p.refresh_A(x, y)
# img_p.refresh_buffer_B(model, opt, smodel1, sopt1, z_sample)
# img_p.load_buffer_B()


def index(request):
    clear()
    img_p.setup()
    navi.setup()
    img_A_paths, _, = navi.get_imgs_index()
    x, y = navi.get_coordinate()
    z = navi.get_level()
    img_p.refresh_buffer_A(z, img_A_paths)
    img_p.refresh_A(x, y)
    img_p.refresh_buffer_B(model, opt, smodel1, sopt1, z_sample)

    context = {}
    web_page = render(request, 'sss_ui.html', context)
    return web_page

def random(request):
    navi.random()
    r_level, r_index, r_x, r_y = navi.get_random()
    img_p.random_A(r_level, r_index, r_x, r_y)

    response = HttpResponse()
    response.status_code = 200
    return response

def restore(request):
    x, y = navi.get_coordinate()
    img_p.refresh_A(x, y)

    response = HttpResponse()
    response.status_code = 200
    return response

def left(request):
    navi.left()
    x, y = navi.get_coordinate()
    # print(x, y)
    img_p.refresh_A(x, y)
    # print(navi.refresh_buffer())
    if navi.refresh_buffer():
        img_A_paths, img_B_paths = navi.get_imgs_index()
        navi.update_buffer_coordinate()
        navi.update_frame_coordinate()
        x,y = navi.get_coordinate()
        z = navi.get_level()
        # print(img_A_paths)
        img_p.refresh_buffer_A(z, img_A_paths)
        # img_p.refresh_buffer_B(z, img_B_paths)

    response = HttpResponse()
    response.status_code = 200
    return response

def up(request):
    navi.up()
    x, y = navi.get_coordinate()
    # print(x, y)
    img_p.refresh_A(x, y)
    if navi.refresh_buffer():
        img_A_paths, img_B_paths = navi.get_imgs_index()
        navi.update_buffer_coordinate()
        navi.update_frame_coordinate()
        x,y = navi.get_coordinate()
        z = navi.get_level()
        # print(img_A_paths)
        img_p.refresh_buffer_A(z, img_A_paths)
        # img_p.refresh_buffer_B(z, img_B_paths)

    response = HttpResponse()
    response.status_code = 200
    return response

def down(request):
    navi.down()
    x, y = navi.get_coordinate()
    # print(x, y)
    img_p.refresh_A(x, y)
    if navi.refresh_buffer():
        img_A_paths, img_B_paths = navi.get_imgs_index()
        navi.update_buffer_coordinate()
        navi.update_frame_coordinate()
        x,y = navi.get_coordinate()
        z = navi.get_level()
        # print(img_A_paths)
        img_p.refresh_buffer_A(z, img_A_paths)
        # img_p.refresh_buffer_B(z, img_B_paths)

    response = HttpResponse()
    response.status_code = 200
    return response

def right(request):
    navi.right()
    x, y = navi.get_coordinate()
    # print(x, y)
    img_p.refresh_A(x, y)
    if navi.refresh_buffer():
        img_A_paths, img_B_paths = navi.get_imgs_index()
        navi.update_buffer_coordinate()
        navi.update_frame_coordinate()
        x,y = navi.get_coordinate()
        z = navi.get_level()
        # print(img_A_paths)
        img_p.refresh_buffer_A(z, img_A_paths)
        # img_p.refresh_buffer_B(z, img_B_paths)

    response = HttpResponse()
    response.status_code = 200
    return response

def transfer(request):
    x, y = navi.get_coordinate()
    if os.path.exists(path_buffer_B):
        img_p.refresh_B(x, y)
    else:
        img_p.refresh_buffer_B(model, opt, smodel1, sopt1, z_sample)
        img_p.refresh_B(x, y)

    response = HttpResponse()
    response.status_code = 200
    return response

def zoom_in(request):
    z = navi.get_level()
    if z > 2:
        response = HttpResponse()
        response.status_code = 200
        return response

    navi.zoom_in()
    img_A_paths, img_B_paths = navi.get_imgs_index()
    navi.update_buffer_coordinate()
    navi.update_frame_coordinate()
    x,y = navi.get_coordinate()
    z = navi.get_level()
    # print(img_A_paths)
    img_p.refresh_buffer_A(z, img_A_paths)
    # img_p.refresh_buffer_B(z, img_B_paths)
    img_p.refresh_A(x, y)
    
    # fx, fy = navi.get_frame_coordinate()
    img_p.zoom_in_B(x, y, z)
    img_p.refresh_B(x, y)

    # rx, ry = navi.get_cg_coordinate()
    # img_p.save_cg(x, y, z)

    response = HttpResponse()
    response.status_code = 200
    return response

def super_resolution(request):
    x, y = navi.get_coordinate()
    z = navi.get_level()
    if z == 2:
        t_model = model2
        t_opt = opt2
        t_smodel = smodel2
        t_sopt = sopt2
        buffer_path = path_buffer_B_2
    else:
        t_model = model3
        t_opt = opt3
        t_smodel = smodel3
        t_sopt = sopt3
        buffer_path = path_buffer_B_3

    if os.path.exists(buffer_path):
        if navi.refresh_buffer():
            img_A_paths, img_B_paths = navi.get_imgs_index()
            navi.update_buffer_coordinate()
            navi.update_frame_coordinate()
            x,y = navi.get_coordinate()
            z = navi.get_level()
            # print(img_A_paths)
            img_p.refresh_buffer_A(z, img_A_paths)
            
            img_p.transfer(t_model, t_opt, t_smodel, t_sopt, z)
            img_p.refresh_B(x, y)
        else:
            img_p.refresh_B(x, y)
    else:
        img_p.transfer(t_model, t_opt, t_smodel, t_sopt, z)
        img_p.refresh_B(x, y)

    response = HttpResponse()
    response.status_code = 200
    return response

def zoom_out(request):
    z = navi.get_level()
    if z < 2:
        response = HttpResponse()
        response.status_code = 200
        return response

    navi.zoom_out()
    img_p.zoom_out_B(z)
    img_A_paths, img_B_paths = navi.get_imgs_index()
    navi.update_frame_coordinate()
    navi.update_buffer_coordinate()
    x,y = navi.get_coordinate()
    z = navi.get_level()
    # print(img_A_paths)
    img_p.refresh_buffer_A(z, img_A_paths)
    # img_p.refresh_buffer_B(z, img_B_paths)
    img_p.refresh_A(x, y)

    response = HttpResponse()
    response.status_code = 200
    return response

def get_z(request):
    # response = HttpResponse("get_z----------!")
    z = navi.get_level()
    response = HttpResponse("%s"%z)
    response.status_code = 200
    return response