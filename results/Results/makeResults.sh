cd ../../outputs
tree -d
cd ../results/Results

read -p "Status: " status

if [ $status -eq "1" ]
then
    python makeResults.py --mode=0 --name=Result
elif [ $status -eq "2" ]
then
    python makeResults.py --mode=1 --compare=3
fi