read status

if [ $status -eq "00" ]
then
    cd ..
    cd time
    python Time_Measurement.py
    cd ..
    cd Results
    rcs='1 2 3'
    pcs='0 0.2 0.4'
    lcs='0 0.5 1'

    for rc in $rcs
    do
        tmp=""
        for pc in $pcs
        do
            for lc in $lcs
            do
                tmp="$tmp${rc}_${pc}_${lc},"
            done
        done
        tmp="Vanilla_PANPP,TwinReader,${tmp}"
        python makeResults.py --mode=0 --compare=$tmp --name=$rc
    done
elif [ $status -eq "01" ]
then
    tmp="2_0.2_0.5"
    tmp="Vanilla_PANPP,TwinReader,${tmp},"
    python makeResults.py --mode=0 --compare=$tmp --name=Result
elif [ $status -eq "10" ]
then
    python makeResults.py --mode=1 --compare=3
fi