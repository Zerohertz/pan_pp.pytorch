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
cd ..
cd results/time
rm -rf tmp.csv
cd ..
cd ..

# cfg=pan_pp_TwinReader
cfg=pan_pp_test
# cfg=pan_pp_target

# Execution
exe(){
    ## Test
    read -p "Exp Name: " tmp
    echo "Exp Name: " $tmp
    CUDA_VISIBLE_DEVICES=${CUDA_NUM} python test.py config/pan_pp/$cfg.py
    
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
    cp ./config/pan_pp/$cfg.py ./outputs/$tmp/
    
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

exe