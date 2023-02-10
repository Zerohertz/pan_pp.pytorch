cd outputs
rm -rf 20*
cd ..
cd time
rm -rf tmp.csv
cd ..
sh compile.sh
CUDA_VISIBLE_DEVICES=3 python test.py config/pan_pp/pan_pp_test.py
