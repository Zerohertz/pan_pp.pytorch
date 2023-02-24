cd outputs
# rm -rf 20*
rm -rf Ground_Truth
cd ..

python makeGT.py

cd outputs
# mv ./20* ./Ground_Truth
cd ..
