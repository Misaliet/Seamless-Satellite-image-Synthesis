from . import image_processing
from . import navigation
from pathlib import Path
import os


def clear():
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    path_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A.png")
    path_B = os.path.join(BASE_DIR, "static", "runtime/images/B/B.png")
    path_buffer_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b.png")
    path_buffer_B = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b.png")
    path_label_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_l.png")
    path_buffer_label_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b_l.png")
    path_ins_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_i.png")
    path_buffer_ins_A = os.path.join(BASE_DIR, "static", "runtime/images/A/A_b_i.png")
    path_buffer_B_2 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_2.png")
    path_buffer_B_3 = os.path.join(BASE_DIR, "static", "runtime/images/B/B_b_3.png")
    path_cg_2 = os.path.join(BASE_DIR, "static", "runtime/images/A/cg_2.png")
    path_cg_3 = os.path.join(BASE_DIR, "static", "runtime/images/A/cg_3.png")
    if os.path.exists(path_A):
        os.remove(path_A)
    if os.path.exists(path_B):
        os.remove(path_B)
    if os.path.exists(path_buffer_A):   
        os.remove(path_buffer_A)
    if os.path.exists(path_buffer_B):
        os.remove(path_buffer_B)
    if os.path.exists(path_label_A):   
        os.remove(path_label_A)
    if os.path.exists(path_buffer_label_A):   
        os.remove(path_buffer_label_A)
    if os.path.exists(path_ins_A):   
        os.remove(path_ins_A)
    if os.path.exists(path_buffer_ins_A):   
        os.remove(path_buffer_ins_A)
    if os.path.exists(path_buffer_B_2):
        os.remove(path_buffer_B_2)
    if os.path.exists(path_buffer_B_3):
        os.remove(path_buffer_B_3)
    if os.path.exists(path_cg_2):
        os.remove(path_cg_2)
    if os.path.exists(path_cg_3):
        os.remove(path_cg_3)

navi = navigation.navigation()
img_p = image_processing.imageProcessing()

def get_navi():
    return navi

def get_img_p():
    return img_p