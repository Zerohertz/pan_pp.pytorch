# rcs='1 2 3'
# pcs='0 0.2 0.4'
# lcs='0 0.5 1'

# for rc in $rcs
# do
#     tmp=""
#     for pc in $pcs
#     do
#         for lc in $lcs
#         do
#             tmp="$tmp${rc}_${pc}_${lc},"
#         done
#     done
#     tmp="Vanilla_PANPP,TwinReader,${tmp}"
#     python makeResults.py --mode=0 --compare=$tmp --name=$rc
# done

# tmp="2_0.2_0.5,2_0.2_0.5"
# tmp="Vanilla_PANPP,TwinReader,${tmp},"
# python makeReults.py --mode=0 --compare=$tmp --name=h

python makeResults.py --mode=1 --compare=3