import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import table
import seaborn as sns
import  math
sns.set_style("white")


def write_table_offload(tdata = 1):
    mab_offloading_path ='result\MAB_{0}\MAB_0{0}403\MAB_task_offloading_s4.csv'.format(tdata) 
    dqn_path = 'result\Data{0}\DQN_0{0}403/test\DQN_task_offloading_s4_vm4.5.csv'.format(tdata)
    mab_df = pd.read_csv(mab_offloading_path)
    dqn_df = pd.read_csv(dqn_path)
    
    x = mab_df.groupby(['number_of_server']).mean()
write_table_offload(1)
# files4=d[d["somay"]==0]
# xxx.write("average_time,"+str(np.around(np.average(files1["may0"]),decimals=2))+","+str(np.around(np.average(files2["may0"]),decimals=2))+","\
#     +str(np.around(np.average(files3["may0"]),decimals=2))+","+str(np.around(np.average(files4["may0"]),decimals=2))+"\n")
# xxx.write("max_time,"+str(np.around(np.max(files1["may0"]),decimals=2))+","+str(np.around(np.max(files2["may0"]),decimals=2))+","\
#     +str(np.around(np.max(files3["may0"]),decimals=2))+","+str(np.around(np.max(files4["may0"]),decimals=2))+"\n")
# xxx.write("min_time,"+str(np.around(np.min(files1["may0"]),decimals=2))+","+str(np.around(np.min(files2["may0"]),decimals=2))+","\
#     +str(np.around(np.min(files3["may0"]),decimals=2))+","+str(np.around(np.min(files4["may0"]),decimals=2))+"\n")
# xxx.write("median_time,"+str(np.around(np.median(files1["may0"]),decimals=2))+","+str(np.around(np.median(files2["may0"]),decimals=2))+","\
#     +str(np.around(np.median(files3["may0"]),decimals=2))+","+str(np.around(np.median(files4["may0"]),decimals=2))+"\n")
# xxx.write("soluong,"+str(np.around((len(files1["may0"])/(104774*1)),decimals=2))+","+str(np.around((len(files2["may0"])/(104774*1)),decimals=2))+","\
#     +str(np.around((len(files3["may0"])/(104774)),decimals=2))+","+str(np.around((len(files4["may0"])/(104774)),decimals=2))+"\n")
# xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
#     +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
# xxx.write("var_time,"+str(np.around(np.var(files1["may0"]),decimals=2))+","+str(np.around(np.var(files2["may0"]),decimals=2))+","\
#     +str(np.around(np.var(files3["may0"]),decimals=2))+","+str(np.around(np.var(files4["may0"]),decimals=2))+"\n")
# xxx.write("may1,\n")
# files1=a[a["somay"]==1]
# files2=b[b["somay"]==1]
# files3=c[c["somay"]==1]
# files4=d[d["somay"]==1]
# xxx.write("average_time,"+str(np.around(np.average(files1["may1"]),decimals=2))+","+str(np.around(np.average(files2["may1"]),decimals=2))+","\
#     +str(np.around(np.average(files3["may1"]),decimals=2))+","+str(np.around(np.average(files4["may1"]),decimals=2))+"\n")
# xxx.write("max_time,"+str(np.around(np.max(files1["may1"]),decimals=2))+","+str(np.around(np.max(files2["may1"]),decimals=2))+","\
#     +str(np.around(np.max(files3["may1"]),decimals=2))+","+str(np.around(np.max(files4["may1"]),decimals=2))+"\n")
# xxx.write("min_time,"+str(np.around(np.min(files1["may1"]),decimals=2))+","+str(np.around(np.min(files2["may1"]),decimals=2))+","\
#     +str(np.around(np.min(files3["may1"]),decimals=2))+","+str(np.around(np.min(files4["may1"]),decimals=2))+"\n")
# xxx.write("median_time,"+str(np.around(np.median(files1["may1"]),decimals=2))+","+str(np.around(np.median(files2["may1"]),decimals=2))+","\
#     +str(np.around(np.median(files3["may1"]),decimals=2))+","+str(np.around(np.median(files4["may1"]),decimals=2))+"\n")
# xxx.write("soluong,"+str(np.around((len(files1["may1"])/(104774*1)),decimals=2))+","+str(np.around((len(files2["may1"])/(104774*1)),decimals=2))+","\
#     +str(np.around((len(files3["may1"])/(104774)),decimals=2))+","+str(np.around((len(files4["may1"])/(104774)),decimals=2))+"\n")
# xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
#     +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
# xxx.write("var_time,"+str(np.around(np.var(files1["may1"]),decimals=2))+","+str(np.around(np.var(files2["may1"]),decimals=2))+","\
#     +str(np.around(np.var(files3["may1"]),decimals=2))+","+str(np.around(np.var(files4["may1"]),decimals=2))+"\n")
# xxx.write("may2,\n")
# files1=a[a["somay"]==2]
# files2=b[b["somay"]==2]
# files3=c[c["somay"]==2]
# files4=d[d["somay"]==2]
# xxx.write("average_time,"+str(np.around(np.average(files1["may2"]),decimals=2))+","+str(np.around(np.average(files2["may2"]),decimals=2))+","\
#     +str(np.around(np.average(files3["may2"]),decimals=2))+","+str(np.around(np.average(files4["may2"]),decimals=2))+"\n")
# xxx.write("max_time,"+str(np.around(np.max(files1["may2"]),decimals=2))+","+str(np.around(np.max(files2["may2"]),decimals=2))+","\
#     +str(np.around(np.max(files3["may2"]),decimals=2))+","+str(np.around(np.max(files4["may2"]),decimals=2))+"\n")
# xxx.write("min_time,"+str(np.around(np.min(files1["may2"]),decimals=2))+","+str(np.around(np.min(files2["may2"]),decimals=2))+","\
#     +str(np.around(np.min(files3["may2"]),decimals=2))+","+str(np.around(np.min(files4["may2"]),decimals=2))+"\n")
# xxx.write("median_time,"+str(np.around(np.median(files1["may2"]),decimals=2))+","+str(np.around(np.median(files2["may2"]),decimals=2))+","\
#     +str(np.around(np.median(files3["may2"]),decimals=2))+","+str(np.around(np.median(files4["may2"]),decimals=2))+"\n")
# xxx.write("soluong,"+str(np.around((len(files1["may2"])/(104774*1)),decimals=2))+","+str(np.around((len(files2["may2"])/(104774*1)),decimals=2))+","\
#     +str(np.around((len(files3["may2"])/(104774)),decimals=2))+","+str(np.around((len(files4["may2"])/(104774)),decimals=2))+"\n")
# xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
#     +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
# xxx.write("var_time,"+str(np.around(np.var(files1["may2"]),decimals=2))+","+str(np.around(np.var(files2["may2"]),decimals=2))+","\
#     +str(np.around(np.var(files3["may2"]),decimals=2))+","+str(np.around(np.var(files4["may2"]),decimals=2))+"\n")
# xxx.write("may3,\n")
# files1=a[a["somay"]==3]
# files2=b[b["somay"]==3]
# files3=c[c["somay"]==3]
# files4=d[d["somay"]==3]
# xxx.write("average_time,"+str(np.around(np.average(files1["may3"]),decimals=2))+","+str(np.around(np.average(files2["may3"]),decimals=2))+","\
#     +str(np.around(np.average(files3["may3"]),decimals=2))+","+str(np.around(np.average(files4["may3"]),decimals=2))+"\n")
# xxx.write("max_time,"+str(np.around(np.max(files1["may3"]),decimals=1))+","+str(np.around(np.max(files2["may3"]),decimals=1))+","\
#     +str(np.around(np.max(files3["may3"]),decimals=2))+","+str(np.around(np.max(files4["may3"]),decimals=2))+"\n")
# xxx.write("min_time,"+str(np.around(np.min(files1["may2"]),decimals=2))+","+str(np.around(np.min(files2["may3"]),decimals=2))+","\
#     +str(np.around(np.min(files3["may3"]),decimals=2))+","+str(np.around(np.min(files4["may3"]),decimals=2))+"\n")
# xxx.write("median_time,"+str(np.around(np.median(files1["may3"]),decimals=2))+","+str(np.around(np.median(files2["may3"]),decimals=2))+","\
#     +str(np.around(np.median(files3["may3"]),decimals=2))+","+str(np.around(np.median(files4["may3"]),decimals=2))+"\n")
# xxx.write("soluong,"+str(np.around((len(files1["may3"])/(104774*1)),decimals=2))+","+str(np.around((len(files2["may3"])/(104774*1)),decimals=2))+","\
#     +str(np.around((len(files3["may3"])/(104774)),decimals=2))+","+str(np.around((len(files4["may3"])/(104774)),decimals=2))+"\n")
# xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
#     +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
# xxx.write("var_time,"+str(np.around(np.var(files1["may3"]),decimals=2))+","+str(np.around(np.var(files2["may3"]),decimals=2))+","\
#     +str(np.around(np.var(files3["may3"]),decimals=2))+","+str(np.around(np.var(files4["may3"]),decimals=2))+"\n")
