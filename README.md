# Seamless Satellite-image Synthesis
by [Jialin Zhu](https://vcg.leeds.ac.uk/profiles/jialin-zhu/) and [Tom Kelly](https://vcg.leeds.ac.uk/profiles/twak/).

[**Project site**](https://vcg.leeds.ac.uk/projects/sss/). The code of our models borrows heavily from the [BicycleGAN](https://github.com/junyanz/BicycleGAN) repository and [SPADE](https://github.com/NVlabs/SPADE) repository. Some missing description can be found in the original repository.

## Results video
[![Watch the video](https://img.youtube.com/vi/-oLKVdBQQKI/hqdefault.jpg)](https://www.youtube.com/watch?v=-oLKVdBQQKI)


## Web UI system
[![Watch the video](https://img.youtube.com/vi/JVchd1pifvA/hqdefault.jpg)](https://www.youtube.com/watch?v=JVchd1pifvA)

- The UI system is developed by web framework - Django.
- Clone the code and `cd web_ui`
- Install required packages(mainly Django 3.1 and PyTorch 1.7.1)
  - These are easy to install so we do not provide a `requirements.txt` file.
  - Packages other than Django and PyTorch can be installed in sequence according to the output error logs.
- Download pre-trained weights and put them in `web_ui/sss_ui/checkpoints`.
- Run `python manage.py migrate` and `python manage.py makemigrations`.
- Run `python runserver.py`.
- Access `127.0.0.1/index` thourough a web browser.
- Start play with the UI system

### Pre-trained weights are available here: [Mega link](https://mega.nz/file/MtRhUCIQ#wBbUd4SGL2LLj5n2iQCWu5XQt5iRGicbg0p0PeQTpSA)

We provide some preset map data, if you want more extensive or other map data, you need to replace the map data yourself. There are some features that have not yet been implemented. Please report bugs as github issues.

## SSS pipeline

The SSS whole pipeline will allow users to generate a set of satellite images from map data of three different scale level.
- Clone the code and `cd SPADE`.
- Install required packages(mainly PyTorch 1.7.1)
- Run `bash scit_m.sh [level_1_dataset_dir] [raw_data_dir] [results_output_dir]`.
- The generated satellite images are in the `[results_output_path]` folder.

We provide some preset map data, if you want more extensive or other map data, you need to replace the map data yourself. 

## Training
You can also re-train the whole pipeline or train with your own data.
For copyright reasons, we will not provide download links for the data we use. But they are very easy to obtain, especially for academic institutions such as universities.
Our training data is from [Digimap](https://digimap.edina.ac.uk/). We use [OS MasterMap® Topography Layer](https://digimap.edina.ac.uk/webhelp/os/osdigimaphelp.htm#data_information/os_products/overview.htm) with [GDAL](https://gdal.org/) and [GeoPandas](https://geopandas.org/) to render map images, and we use satellite images from [Aerial](https://digimap.edina.ac.uk/webhelp/aerial/aerialdigimaphelp.htm#data_information/products/25cm_aerial_imagery.htm) via [Getmapping](https://www.getmapping.com/).

### To train *map2sat* for level 1:
- Clone the code and `cd SPADE`.
- Run `python train.py --name [z1] --dataset_mode ins --label_dir [label_dir] --image_dir [image_dir] --instance_dir [instance_dir] --label_nc 13 --load_size 256 --crop_size 256 --niter_decay 20 --use_vae --ins_edge --gpu_ids 0,1,2,3 --batchSize 16`.
- We recommend using a larger batch size so that the encoder can generate results with greater style differences.
### To train *map2sat* for level z (z > 1):
- Clone the code and `cd SPADE`.
- Run `python trainCG.py --name [z2_cg] --dataset_mode insgb --label_dir [label_dir] --image_dir [image_dir] --instance_dir [instance_dir] --label_nc 13 --load_size 256 --crop_size 256 --niter_decay 20 --ins_edge --cg --netG spadebranchn --cg_size 256 --gbk_size 8`.
### To train *seam2cont*:
- Clone the code and `cd BicycleGAN`.
- Run `python train.py --dataroot [dataset_dir] --name [z1sn] --model sn --direction AtoB --load_size 256 --save_epoch_freq 201 --lambda_ml 0 --input_nc 8 --dataset_mode sn --seams_map --batch_size 1 --ndf 32 --conD --forced_mask`.


## Citation
```
@inproceedings{zhu2021seamless,
  title={Seamless Satellite-image Synthesis},
  author={Zhu, J and Kelly, T},
  booktitle={Computer Graphics Forum},
  year={2021},
  organization={Wiley}
}
```

## Acknowledgements
We would like to thank Nvidia Corporation for hardware and Ordnance Survey Mapping for map data which made this project possible. This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK. This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1).
