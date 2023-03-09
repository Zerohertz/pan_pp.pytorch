# Compile Cython
cd ./models/post_processing/pa/
python setup.py build_ext --inplace
cd ../boxgen/
python setup.py build_ext --inplace
cd ../../../
echo Done!

CUDA_VISIBLE_DEVICES=1,3,4,5,7 python train.py config/pan_pp/pan_pp_test.py