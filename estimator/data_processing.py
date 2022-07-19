from operator import mod
import pandas as pd
from datetime import datetime

df = pd.DataFrame(columns=['datetime', 'data_length'])
with open('Aug28_log.txt', mode='rb') as f:
    lines = f.readlines()
    lst_date = []
    lst_dtL_length = []
    for line in lines:
        # check if any decoding error
        try:
            x = line.decode("utf-8")
        except:
            x = str(line)
            print("error decode:", x)
        # get date _time
        # y = x.split("- -")
        # if(len(y)!=2):
        #     print()
        z = x.find("1995")
        if(z != -1):
            date_time =x[z - 7:z +13]
            try:
                xdate = datetime.strptime(date_time, '%d/%b/%Y:%H:%M:%S')
            except:
                print(date_time)
        else:
            continue

        #get data length
        yy = x.split("\"")
        yyy = yy[-1].split(" ")
        num = yyy[-1].strip("\n")
        #print(zz)
        try:
            dt_length = int(num)
        except:
            if(num == '-'):
                
            # if(num != '-'):
            #     dt_length = '-'
                #print(date_time, x)
        #lst_date.append(xdate)
        #lst_dtL_length.append(num)
    #df['datetime'] = lst_date
    #df['data_length'] = lst_dtL_length
    #print(len(lst_date), len(lst_dtL_length))
    #df.to_csv("clarknet_dataset.csv", index= False)