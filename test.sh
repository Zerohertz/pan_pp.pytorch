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

    CUDA_VISIBLE_DEVICES=7 python test.py config/pan_pp/pan_pp_test.py --resize_const=$resize_const --pos_const=$pos_const --len_const=$len_const
}

read resize_const
read pos_const
read len_const
exe ${resize_const} ${pos_const} ${len_const}
