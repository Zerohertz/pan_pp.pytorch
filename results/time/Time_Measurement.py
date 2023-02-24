import os
import csv
import pandas as pd

if 'time.csv' in os.listdir():
    os.remove('./time.csv')

for i in os.listdir():
    if '.csv' in i and not i == 'time.csv':
        data = pd.read_csv(i, header=None)
        tot = 0
        with open('./time.csv', 'a', encoding='utf8') as f:
            wr = csv.writer(f)
            wr.writerow([i[0:-4], data[0].mean()*1000, data[1].mean()*1000, data[2].mean()*1000, data[3].mean()*1000, data[4].mean()*1000, data[5].mean()*1000])