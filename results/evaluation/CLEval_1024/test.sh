rm -rf result.csv

#tmp='Ver1'
#python prepare.py --path=$tmp
#python script.py --path=$tmp

#tmp='Ver2'
#python prepare.py --path=$tmp
#python script.py --path=$tmp

tmp='Ver8_2_0.2_0.5'
python prepare.py --path=$tmp
python script.py --path=$tmp

# for rc in '2' '3' '4'
# do
#     for pc in '0.2' '0.25' '0.3'
#     do
#         for lc in '0.4' '0.5' '0.6'
#         do
#             echo $rc $pc $lc
#             resize_const=$rc
#             pos_const=$pc
#             len_const=$lc
#             python prepare.py --path=Ver6_${resize_const}_${pos_const}_${len_const}
#             python script.py --path=Ver6_${resize_const}_${pos_const}_${len_const}
#         done
#     done
# done
