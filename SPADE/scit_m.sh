# bash scit_m.sh ./datasets/scit_m/scit_1/ ./zraw_m/1/ ./scitm1/
# bash scit_m.sh ./datasets/scit_m/scit_2/ ./zraw_m/2/ ./scitm2/
# bash scit_m.sh ./datasets/scit_m/scit_3/ ./zraw_m/3/ ./scitm3/
# bash scit_m.sh ./datasets/scit_m/scit_4/ ./zraw_m/4/ ./scitm4/
# bash scit_m.sh ./datasets/scit_m/p1/ ./zraw_m/p1/ ./p1/
# bash scit_m.sh ./datasets/scit_m/p2/ ./zraw_m/p2/ ./p2/

if [ $# != 3 ] ; then 
echo "need 3 parameters (original dataset path / raw images path / saving folder path)"
echo "picture mode: 1 for BtoA and other number for AtoB)"
exit 1; 
fi

# PREPARE
rm -rf ./8k
rm -rf ./datasets/8k/
rm -rf ./scit_temp_m/
mkdir scit_temp_m
rm -rf z_samples.pt
rm -rf $3
mkdir -p $3

# ZOOM LEVEL 1
python test.py --name z1 --dataset_mode ins --label_dir $1/map/ --image_dir $1/real/ --instance_dir $1/ins/ --how_many 100000 --load_size 256 --label_nc 13 --use_vae --ins_edge --no_flip --results_dir ./scit_temp_m --random_z --save_z
# python test.py --name z1 --dataset_mode ins --label_dir $1/map/ --image_dir $1/real/ --instance_dir $1/ins/ --how_many 100000 --load_size 256 --label_nc 13 --use_vae --ins_edge --no_flip --results_dir ./scit_temp --random_z --load_z
python whole.py ./scit_temp_m/z1/test_latest/images/synthesized_random_image1 768 png ./$3/z1s.png 1
echo "create z1 large image!"

rm -rf ./8k
python scit_extract_blending.py $2/z1m.png $2/z1i.png $2/z1r.png 768
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z1/
python test.py --name z1 --dataset_mode ins --label_dir ./datasets/8k/map/ --image_dir ./datasets/8k/real/ --instance_dir ./datasets/8k/ins/ --how_many 100000 --load_size 256 --label_nc 13 --use_vae --ins_edge --no_flip --results_dir ./scit_temp_m --random_z --load_z
echo "create z1 non-seams images for blending!"

rm -rf ./8k
python scit_extract_seams.py $2/z1m.png $2/z1i.png $2/z1r.png $3/z1s.png 768
cp -rf ./datasets/gm/B/mask ./8k/
cp -ri ./scit_temp_m/z1/test_latest/images/synthesized_random_image1/* ./8k/ns/val/
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z1sn
python ../Modified_BicycleGAN/new_test.py --dataroot ./datasets/8k --name z1sn --model sn --direction AtoB --load_size 256 --dataset_mode sn --input_nc 8 --seams_map --results_dir ./scit_temp_m/z1sn --num_test 100000 --phase test --no_flip  --n_samples 0 --ndf 32 --conD --forced_mask
echo "create z1 seamless tiles!"

python scit_repalce_seams.py $3/z1s.png ./scit_temp_m/z1sn/test/images $3/z1ns.png 768 blended
python scit_temp.py get_final $3/z1ns.png $3/z1f.png
echo "create z1 final image!"


# ZOOM LEVEL 2
rm -rf ./8k
python scit_createZ.py $3/z1ns.png $2/z2i.png $2/z2m.png $2/z2r.png 0 3072
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
python testCG.py --name z2_cg_b16 --dataset_mode insgb --label_dir ./datasets/8k/map/ --image_dir ./datasets/8k/real/ --instance_dir ./datasets/8k/ins/ --how_many 100000 --load_size 256 --label_nc 13 --ins_edge --cg --netG spadebranchn --cg_size 256 --gbk_size 8 --results_dir ./scit_temp_m
python whole.py ./scit_temp_m/z2_cg_b16/test_latest/images/synthesized_image 3072 png $3/z2s.png
echo "create z2 large image!"

rm -rf ./8k
python scit_extract_blending.py $2/z2m.png $2/z2i.png $2/z2r.png 3072 $3/z1ns.png
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z2_cg_b16/
python testCG.py --name z2_cg_b16 --dataset_mode insgb --label_dir ./datasets/8k/map/ --image_dir ./datasets/8k/real/ --instance_dir ./datasets/8k/ins/ --how_many 100000 --load_size 256 --label_nc 13 --ins_edge --cg --netG spadebranchn --cg_size 256 --gbk_size 8 --results_dir ./scit_temp_m
echo "create z2 non-seams images for blending!"

rm -rf ./8k
python scit_extract_seams.py $2/z2m.png $2/z2i.png $2/z2r.png $3/z2s.png 3072
cp -rf ./datasets/gm/B/mask ./8k/
cp -ri ./scit_temp_m/z2_cg_b16/test_latest/images/synthesized_image/* ./8k/ns/val/
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z2sn
python ../Modified_BicycleGAN/new_test.py --dataroot ./datasets/8k --name z2sn --model sn --direction AtoB --load_size 256 --dataset_mode sn --input_nc 8 --seams_map --results_dir ./scit_temp_m/z2sn --num_test 100000 --phase test --no_flip  --n_samples 0 --ndf 32 --conD --forced_mask
echo "create z2 seamless tiles!"

python scit_repalce_seams.py $3/z2s.png ./scit_temp_m/z2sn/test/images $3/z2ns.png 3072 blended
python scit_temp.py get_final $3/z2ns.png $3/z2f.png
echo "create z2 final image!"


# ZOOM LEVEL 3
rm -rf ./8k
python scit_createZ.py $3/z2ns.png $2/z3i.png $2/z3m.png $2/z3r.png 0 12288
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
python testCG.py --name z3_cg_b16 --dataset_mode insgb --label_dir ./datasets/8k/map/ --image_dir ./datasets/8k/real/ --instance_dir ./datasets/8k/ins/ --how_many 100000 --load_size 256 --label_nc 13 --ins_edge --cg --netG spadebranchn --cg_size 256 --gbk_size 8 --results_dir ./scit_temp_m
python whole.py ./scit_temp_m/z3_cg_b16/test_latest/images/synthesized_image 12288 png $3/z3s.png
echo "create z3 large image!"

rm -rf ./8k
python scit_extract_blending.py $2/z3m.png $2/z3i.png $2/z3r.png 12288 $3/z2ns.png
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z3_cg_b16/
python testCG.py --name z3_cg_b16 --dataset_mode insgb --label_dir ./datasets/8k/map/ --image_dir ./datasets/8k/real/ --instance_dir ./datasets/8k/ins/ --how_many 100000 --load_size 256 --label_nc 13 --ins_edge --cg --netG spadebranchn --cg_size 256 --gbk_size 8 --results_dir ./scit_temp_m
echo "create z3 non-seams images for blending!"

rm -rf ./8k
python scit_extract_seams.py $2/z3m.png $2/z3i.png $2/z3r.png $3/z3s.png 12288
cp -rf ./datasets/gm/B/mask ./8k/
cp -ri ./scit_temp_m/z3_cg_b16/test_latest/images/synthesized_image/* ./8k/ns/val/
rm -rf ./datasets/8k/
cp -rf ./8k ./datasets/
rm -rf ./scit_temp_m/z3sn
python ../Modified_BicycleGAN/new_test.py --dataroot ./datasets/8k --name z3sn --model sn --direction AtoB --load_size 256 --dataset_mode sn --input_nc 8 --seams_map --results_dir ./scit_temp_m/z3sn --num_test 100000 --phase test --no_flip  --n_samples 0 --ndf 32 --conD --forced_mask
echo "create z3 seamless tiles!"

python scit_repalce_seams.py $3/z3s.png ./scit_temp_m/z3sn/test/images $3/z3ns.png 12288 blended
python scit_temp.py get_final $3/z3ns.png $3/z3f.png
echo "create z3 final image!"