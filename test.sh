# Compile Cython
cd ./models/post_processing/pa/
python setup.py build_ext --inplace
cd ../boxgen/
python setup.py build_ext --inplace
cd ../../../
echo Done!

# Setup
nvidia-smi
read -p 'CUDA_NUM: ' CUDA_NUM

# Initialization
cd outputs
rm -rf 20*
rm -rf ${resize_const}_${pos_const}_${len_const}
cd ..
cd results/time
rm -rf tmp.csv
rm -rf ${resize_const}_${pos_const}_${len_const}.csv
cd ..
cd ..

# Execution
exe(){
    ## Test
    echo "${resize_const}_${pos_const}_${len_const}"
    read -p "Exp Name: " tmp
    if [ -z $tmp ]
    then
    tmp="${resize_const}_${pos_const}_${len_const}"
    fi
    echo "Exp Name: " $tmp
    CUDA_VISIBLE_DEVICES=${CUDA_NUM} python test.py config/pan_pp/pan_pp_test.py --resize_const=${resize_const} --pos_const=${pos_const} --len_const=${len_const}
    
    ## Rename tmp.csv & Make time.csv
    cd results/time
    rm -rf $tmp.csv
    mv ./tmp.csv ./$tmp.csv
    python Time_Measurement.py
    cd ..
    cd ..
    
    ## Rename outputs
    cd outputs
    rm -rf $tmp
    mv ./20* ./$tmp
    cd ..
    
    ## Move Cfg
    cp ./config/pan_pp/pan_pp_test.py ./outputs/$tmp/
    
    ## Evaluation
    cd results/evaluation
    rm -rf $tmp.csv
    cd CLEval_1024
    python prepare.py --path=$tmp
    python script.py --path=$tmp
    cd ..
    cd ..
    cd ..
}

# exe
# read -p 'Resize Constant: ' rcs
# read -p 'Position Constant: ' pcs
# read -p 'Length Constant: ' lcs

rcs=2
pcs=0.2
lcs=0.5

for rc in $rcs
do
    for pc in $pcs
    do
        for lc in $lcs
        do
            resize_const=$rc
            pos_const=$pc
            len_const=$lc
            exe
        done
    done
done