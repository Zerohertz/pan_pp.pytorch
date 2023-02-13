cd outputs
rm -rf 20*
cd ..
cd time
rm -rf tmp.csv
cd ..
sh compile.sh

exe(){
    echo "resize_const: " ${resize_const}
    echo "pos_const: " ${pos_const}
    echo "len_const: " ${len_const}

    CUDA_VISIBLE_DEVICES=3 python test.py config/pan_pp/pan_pp_test.py --resize_const=$resize_const --pos_const=$pos_const --len_const=$len_const

    cd time
    mv tmp.csv Ver3_${resize_const}_${pos_const}_${len_const}.csv
    cd ..

    cd outputs
    mv 20* Ver3_${resize_const}_${pos_const}_${len_const}
    cd ..
}

for rc in '2' '3' '4'
do
    for pc in '0' '0.1' '0.2'
    do
        for lc in '0' '0.5' '1'
        do
            echo $rc $pc $lc
            resize_const=$rc
            pos_const=$pc
            len_const=$lc
            exe ${resize_const} ${pos_const} ${len_const}
        done
    done
done

cd time
rm -rf time.csv
python Time_Measurement.py