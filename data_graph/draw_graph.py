from cmath import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# a=[]
# b=[]
# for i in range(1,2):
#     files=pd.read_csv("result/MAB2/MAB_5phut_s4.csv")
#     x=files["mean_reward"].to_numpy()[0:100]
#     b.append(x)
#     files=pd.read_csv("result/DQN2/reward_5phut_env_s4.csv")
#     xx=files["mean_reward"].to_numpy()[0:100]
#     a.append(xx)
#     #print(a)
# files=pd.read_csv("result/DDQN2/reward_5phut_env_s4.csv")
# files1=pd.read_csv("result/DuelingDQN2/reward_5phut_env_s4.csv")
# #files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 1/ketqua_oneday.csv")
# #files1=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 2/ketqua_oneday.csv")
# m=[i for i in range(1,101)]
# #files=pd.read_excel("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/compare.xlsx")
# fig, ax = plt.subplots()
# x=[i for i in range(0,100)]
# labels=["an","an1","aN1","an3"]

# ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
# ax.plot(x, files1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="Dueling DQN",lw=1)
# ax.plot(x, files["mean_reward"][0:100],marker='o', markevery=5,label="DDQN",color="blue",lw=1)
# ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
# ax.set_ylabel('MQoE',fontsize=15)
# ax.set_xlabel('Time slots',fontsize=15)
# ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

# plt.setp(ax.get_xticklabels(), fontsize=15)
# plt.setp(ax.get_yticklabels(), fontsize=15)
# #ax.set_xticklabels(labels)
# ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
# plt.grid(alpha=0.5)
# #loc='upper center'
# plt.show()
# plt.savefig("Compare_5p.eps")
# #print(max(files1["Random"]))
# #print(min(files1["Random"]))
# #print(np.average(b,axis=0))"""
def CompareDataOffloadingType(env = 'DQN', tdata = 1 ):
    f1_path = 'result\Data{0}\DQN_0{0}103/test/test_5minute_env_s1_ts{0}_vm4.5.csv'.format(tdata)
    f2_path = 'result\Data{0}\DQN_0{0}203/test/test_5minute_env_s2_ts{0}_vm4.5.csv'.format(tdata)
    f3_path = 'result\Data{0}\DQN_0{0}303/test/test_5minute_env_s3_ts{0}_vm4.5.csv'.format(tdata)
    f4_path = 'result\Data{0}\DQN_0{0}403/test/test_5minute_env_s4_ts{0}_vm4.5.csv'.format(tdata)
    if(env == 'MAB'):
        f1_path = 'result\MAB_{0}\MAB_0{0}103\MAB_5minute_s1_vm4.5_ts{0}.csv'.format(tdata)
        f2_path = 'result\MAB_{0}\MAB_0{0}203\MAB_5minute_s2_vm4.5_ts{0}.csv'.format(tdata)
        f3_path = 'result\MAB_{0}\MAB_0{0}303\MAB_5minute_s3_vm4.5_ts{0}.csv'.format(tdata)
        f4_path = 'result\MAB_{0}\MAB_0{0}403\MAB_5minute_s4_vm4.5_ts{0}.csv'.format(tdata)
    
    file1 = pd.read_csv(f1_path)
    file2 = pd.read_csv(f2_path)
    file3 = pd.read_csv(f3_path)
    file4 = pd.read_csv(f4_path)

    # if(env == 'DQN'):
    #     file1 = pd.read_csv('result\Data{0}\DQN_0{0}103/test/test_5minute_env_s1_ts{0}_vm4.5.csv'.format(tdata))
    #     file2 = pd.read_csv('result\Data{0}\DQN_0{0}203/test/test_5minute_env_s2_ts{0}_vm4.5.csv'.format(tdata))
    #     file3 = pd.read_csv('result\Data{0}\DQN_0{0}303/test/test_5minute_env_s3_ts{0}_vm4.5.csv'.format(tdata))
    #     file4 = pd.read_csv('result\Data{0}\DQN_0{0}403/test/test_5minute_env_s4_ts{0}_vm4.5.csv'.format(tdata))
    # elif(env == 'MAB'):
    #     file1 = pd.read_csv('result\MAB_{0}\MAB_0{0}103\MAB_5minute_s1_vm4.5_ts{0}.csv'.format(tdata))
    #     file2 = pd.read_csv('result\MAB_{0}\MAB_0{0}203\MAB_5minute_s2_vm4.5_ts{0}.csv'.format(tdata))
    #     file3 = pd.read_csv('result\MAB_{0}\MAB_0{0}303\MAB_5minute_s3_vm4.5_ts{0}.csv'.format(tdata))
    #     file4 = pd.read_csv('result\MAB_{0}\MAB_0{0}403\MAB_5minute_s4_vm4.5_ts{0}.csv'.format(tdata))
    fig, ax = plt.subplots()
    x=[i for i in range(0,100)]
    print(file1.mean_reward.values)
    #ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
    ax.plot(x, file1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="Only LS",lw=1)
    ax.plot(x, file2["mean_reward"][0:100],marker='^', markevery=5,label="OPVFC 1VS",color="blue",lw=1)
    ax.plot(x, file3["mean_reward"][0:100],marker='o', markevery=5,label="OPVFC 2VS",color="green",lw=1)
    ax.plot(x, file4["mean_reward"][0:100],marker='*', markevery=5,label="OPVFC 3VS",color="orange",lw=1)
    #ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
    ax.set_ylabel('Target value',fontsize=15)
    ax.set_xlabel('Time slots',fontsize=15)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    #ax.set_xticklabels(labels)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
    plt.grid(alpha=0.5)
    plt.show()
    #plt.savefig("data_graph/fig/1_compare_offload_{0}_ts{1}.png".format(env,tdata))
    avg1 = np.average(file1["mean_reward"][0:100])
    avg2 = np.average(file2["mean_reward"][0:100])
    avg3 = np.average(file3["mean_reward"][0:100])
    avg4 = np.average(file4["mean_reward"][0:100])
    print(env,tdata,avg4/avg3, avg4/avg2, avg4/avg1)


def CompareDataMQ(tdata = 1,me = 'mae'):
    
    file1 = pd.read_csv('result\logs{0}\DQN_1{0}103.csv'.format(tdata))
    file2 = pd.read_csv('result\logs{0}\DQN_1{0}203.csv'.format(tdata))
    file3 = pd.read_csv('result\logs{0}\DQN_1{0}303.csv'.format(tdata))
    file4 = pd.read_csv('result\logs{0}\DQN_1{0}403.csv'.format(tdata))
    fig, ax = plt.subplots()
    n = 99
    x=[i for i in range(0,n)]

    metrics_name = me
    if me == 'rmse':
        file1['rmse'] = file1['mse'].apply(lambda x:sqrt(x))
        file2['rmse'] = file2['mse'].apply(lambda x:sqrt(x))
        file3['rmse'] = file3['mse'].apply(lambda x:sqrt(x))
        file4['rmse'] = file4['mse'].apply(lambda x:sqrt(x))
    #ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
    ax.plot(x, file1[me][0:n],marker="P", markevery=5,color="red",label="Only LS",lw=1)
    ax.plot(x, file2[me][0:n],marker='^', markevery=5,label="OPVFC 1VS",color="blue",lw=1)
    ax.plot(x, file3[me][0:n],marker='o', markevery=5,label="OPVFC 2VS",color="green",lw=1)
    ax.plot(x, file4[me][0:n],marker='*', markevery=5,label="OPVFC 3VS",color="orange",lw=1)

    #ax.plot(x, file1['mse'][0:n],marker="P", markevery=5,color="red",label="Only LS",lw=1)
    #ax.plot(x, file2['mse'][0:n],marker='^', markevery=5,label="OPVFC 1VS",color="blue",lw=1)
    #ax.plot(x, file3['mse'][0:n],marker='o', markevery=5,label="OPVFC 2VS",color="green",lw=1)
    #ax.plot(x, file4['mse'][0:n],marker='*', markevery=5,label="OPVFC 3VS",color="orange",lw=1)


    #ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
    ax.set_ylabel('Rmse',fontsize=15)
    ax.set_xlabel('Time slots',fontsize=15)
    x1 = [0,0.2,0.4,0.6,0.8,1]
    x2 = [0,0.4,0.8,1.2,1.6,2,2.4,2.8]
    x3 = [0, 1,2,3,4,5,6]
    ax.set_yticks(x2)

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    #ax.set_xticklabels(labels)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
    plt.grid(alpha=0.5)
    #plt.show()
    plt.savefig("data_graph/fig/2_compare_offload_{0}_ts{1}.png".format(me,tdata))
       
#CompareDataMQ(1, 'rmse')

def CompareDataOffloadingTypeAutoScale(env = 'DQN', tdata = 1 ):
    f1_path = 'result\Data_autoscale_{0}\DQN_0{0}401/test/test_5minute_env_s4_ts{0}_vm1.5.csv'.format(tdata)
    f2_path = 'result\Data_autoscale_{0}\DQN_0{0}402/test/test_5minute_env_s4_ts{0}_vm3.0.csv'.format(tdata)
    f3_path = 'result\Data_autoscale_{0}\DQN_0{0}403/test/test_5minute_env_s4_ts{0}_vm4.5.csv'.format(tdata)
    f4_path = 'result\Data_autoscale_{0}\DQN_0{0}413/test/test_5minute_env_s4_ts{0}_arima.csv'.format(tdata)
    if(env == 'MAB'):
        f1_path = 'result\MAB_{0}\MAB_0{0}401\MAB_5minute_s4_vm1.5_ts{0}.csv'.format(tdata)
        f2_path = 'result\MAB_{0}\MAB_0{0}402\MAB_5minute_s4_vm3.0_ts{0}.csv'.format(tdata)
        f3_path = 'result\MAB_{0}\MAB_0{0}403\MAB_5minute_s4_vm4.5_ts{0}.csv'.format(tdata)
        f4_path = 'result\MAB_{0}\MAB_0{0}413\MAB_5minute_s4_ts{0}_arima.csv'.format(tdata)
    
    file1 = pd.read_csv(f1_path)
    file2 = pd.read_csv(f2_path)
    file3 = pd.read_csv(f3_path)
    file4 = pd.read_csv(f4_path)

    fig, ax = plt.subplots()
    x=[i for i in range(0,100)]
    print(np.average(file1.mean_reward.values))
    print(np.average(file2.mean_reward.values))
    print(np.average(file3.mean_reward.values))
    print(np.average(file4.mean_reward.values))
    #ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
    ax.plot(x, file1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="OPVFC 1M",lw=1)
    ax.plot(x, file2["mean_reward"][0:100],marker='^', markevery=5,label="O PVFC 2M",color="blue",lw=1)
    ax.plot(x, file3["mean_reward"][0:100],marker='o', markevery=5,label="OPVFC 3M",color="green",lw=1)
    ax.plot(x, file4["mean_reward"][0:100],marker='*', markevery=5,label="OPVFC AM",color="orange",lw=1)
    #ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
    ax.set_ylabel('Target value',fontsize=15)
    ax.set_xlabel('Time slots',fontsize=15)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    #ax.set_xticklabels(labels)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
    plt.grid(alpha=0.5)
    #plt.show()
    plt.savefig("data_graph/fig/1_compare_offload_autoscale_{0}_ts{1}.pdf".format(env,tdata))
    avg1 = np.average(file1["mean_reward"][0:100])
    avg2 = np.average(file2["mean_reward"][0:100])
    avg3 = np.average(file3["mean_reward"][0:100])
    avg4 = np.average(file4["mean_reward"][0:100])
    print(env,tdata,avg4/avg3, avg4/avg2, avg4/avg1)

def export_legend(legend, filename="data_graph/fig/quality_legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def DrawTaskQuality(str_path, env, tdata,autoscale, split_legend = 0):
    x=[i for i in range(1,101)]
    labels=["Combine fuzzy and deep q 1","deep q learning 1","fuzzy","random"]
    fig, ax = plt.subplots()
    files=pd.read_csv(str_path)[0:100]
    m=files["good"]+files["medium"]+files["bad"]
    #print(m)
    #print([np.mean(files["good"]/m)]*len(x))
    ax.plot(x,[np.mean(files["good"]/m)]*len(x), label='Mean good',color="violet", linestyle='--')
    ax.plot(x,[np.mean(files["medium"]/m)]*len(x), label='Mean medium',color="y", linestyle='--')
    ax.plot(x,[np.mean(files["bad"]/m)]*len(x), label='Mean bad',color="gray", linestyle='--')
    ax.plot(x,files["good"]/m,label="Good",color="violet",marker='*',markevery=5)
    ax.plot(x,files["medium"]/m,marker='^',label="Medium",color="y",markevery=5)
    #ax1.plot(x,files["bus2"],":",label="vehicular fog 2")
    ax.plot(x,files["bad"]/m,label="Bad",color="gray",markevery=5)
    ax.set_xlabel("Time slots",fontsize=15)
    ax.set_ylabel("Ratio",fontsize=15)
    ax.set_ylim(0,1)
    legend = plt.figlegend(loc='lower center', bbox_to_anchor=(0., 1.02, 1., .102), ncol=3)

    #ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=6)
    plt.grid(alpha=0.5)
    plt.savefig("data_graph/fig/4_compare_quality_{0}_scale_{2}_ts{1}.pdf".format(env,tdata, autoscale))
    if(split_legend == 1):
        export_legend(legend)
    #plt.show()

def CompareDataOffloadingTypeTrainDQN(tdata = 1, autoscale = 0):
    f1_path = 'result\Data{0}\DQN_0{0}103/train/DQL_5_minute_s1_vm4.5_ts{0}.csv'.format(tdata)
    f2_path = 'result\Data{0}\DQN_0{0}203/train/DQL_5_minute_s2_vm4.5_ts{0}.csv'.format(tdata)
    f3_path = 'result\Data{0}\DQN_0{0}303/train/DQL_5_minute_s3_vm4.5_ts{0}.csv'.format(tdata)
    f4_path = 'result\Data{0}\DQN_0{0}403/train/DQL_5_minute_s4_vm4.5_ts{0}.csv'.format(tdata)
    
    if (autoscale == 1):
        f1_path = 'result\Data_autoscale_{0}\DQN_0{0}401/train/DQL_5_minute_s4_vm1.5_ts{0}.csv'.format(tdata)
        f2_path = 'result\Data_autoscale_{0}\DQN_0{0}402/train/DQL_5_minute_s4_vm3.0_ts{0}.csv'.format(tdata)
        f3_path = 'result\Data_autoscale_{0}\DQN_0{0}403/train/DQL_5_minute_s4_vm4.5_ts{0}.csv'.format(tdata)
        f4_path = 'result\Data_autoscale_{0}\DQN_0{0}413/train/DQL_5_minute_s4_ts{0}_arima.csv'.format(tdata)
    

    file1 = pd.read_csv(f1_path)
    file2 = pd.read_csv(f2_path)
    file3 = pd.read_csv(f3_path)
    file4 = pd.read_csv(f4_path)

    fig, ax = plt.subplots()
    l = len(file1)
    x=[i for i in range(0,l)]
    #ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
    ax.plot(x, file1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="OPVFC 1M",lw=1)
    ax.plot(x, file2["mean_reward"][0:100],marker='^', markevery=5,label="OPVFC 2M",color="blue",lw=1)
    ax.plot(x, file3["mean_reward"][0:100],marker='o', markevery=5,label="OPVFC 3M",color="green",lw=1)
    ax.plot(x, file4["mean_reward"][0:100],marker='*', markevery=5,label="OPVFC AM",color="orange",lw=1)
    #ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
    ax.set_ylabel('Target value',fontsize=15)
    ax.set_xlabel('Time slots',fontsize=15)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    #ax.set_xticklabels(labels)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
    plt.grid(alpha=0.5)
    #plt.show()
    plt.savefig("data_graph/fig/3_compare_train_dqn_scale{0}_ts{1}.pdf".format(autoscale,tdata))
    avg1 = np.average(file1["mean_reward"][0:100])
    avg2 = np.average(file2["mean_reward"][0:100])
    avg3 = np.average(file3["mean_reward"][0:100])
    avg4 = np.average(file4["mean_reward"][0:100])
    print('DQN',tdata,avg4/avg3, avg4/avg2, avg4/avg1)


    #plt.savefig("fdqoqualitywithtimeslots.eps")
DrawTaskQuality('result\Data1\DQN_01403/test\DQN_n_quality_tasks_s4_vm4.5.csv', 'dqn', 1,0,1)
#DrawTaskQuality('result\MAB_1\MAB_01403\MAB_n_quality_tasks_s4.csv','mab', 1,0)
#DrawTaskQuality('result\Data2\DQN_02403/test\DQN_n_quality_tasks_s4_vm4.5.csv', 'dqn', 2,0)
#DrawTaskQuality('result\MAB_2\MAB_02403\MAB_n_quality_tasks_s4.csv', 'mab', 2,0)

   
#DrawTaskQuality('result\Data_autoscale_1\DQN_01413/test\DQN_n_quality_tasks_s4.csv','dqn',1,1,0)
#DrawTaskQuality('result\Data_autoscale_2\DQN_02413/test\DQN_n_quality_tasks_s4.csv','dqn',2,1,0)
#DrawTaskQuality('result\Data_autoscale_1\DQN_01413/test\DQN_n_quality_tasks_s4.csv')
#DrawTaskQuality('result\Data_autoscale_1\DQN_01413/test\DQN_n_quality_tasks_s4.csv')

#CompareDataOffloadingTypeAutoScale('DQN', 2)
#CompareDataOffloadingType('DQN',1)
#CompareDataOffloadingTypeTrainDQN(2,1)