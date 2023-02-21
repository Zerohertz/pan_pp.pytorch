rcs='1 2 3'
pcs='0 0.2 0.4'
lcs='0 0.5 1'

for rc in '1' '2' '3'
do
    tmp=""
    for pc in '0' '0.2' '0.4'
    do
        for lc in '0' '0.5' '1'
        do
            tmp="$tmp${rc}_${pc}_${lc},"
        done
    done
    tmp="Vanilla_PANPP,TwinReader,${tmp}"
    python makeReults.py --mode=0 --compare=$tmp --name=$rc
done