from cmath import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def CompareDataOffloadingType(env = 'DQN', tdata = 1, ):
    if(env == 'DQN'):
        file1 = pd.read_csv('result\Data{0}\DQN_0{0}401/test/test_5minute_env_s4_ts{0}_vm1.5.csv'.format(tdata))
        file2 = pd.read_csv('result\Data{0}\DQN_0{0}402/test/test_5minute_env_s4_ts{0}_vm3.0.csv'.format(tdata))
        file3 = pd.read_csv('result\Data{0}\DQN_0{0}403/test/test_5minute_env_s4_ts{0}_vm4.5.csv'.format(tdata))
        file4 = pd.read_csv('result\Data{0}\DQN_0{0}413_arima/test/test_5minute_env_s4_ts{0}_arima.csv'.format(tdata))
    elif(env == 'MAB'):
        file1 = pd.read_csv('result\MAB_{0}\MAB_0{0}103\MAB_5minute_s1_vm4.5_ts{0}.csv'.format(tdata))
        file2 = pd.read_csv('result\MAB_{0}\MAB_0{0}203\MAB_5minute_s2_vm4.5_ts{0}.csv'.format(tdata))
        file3 = pd.read_csv('result\MAB_{0}\MAB_0{0}303\MAB_5minute_s3_vm4.5_ts{0}.csv'.format(tdata))
        file4 = pd.read_csv('result\MAB_{0}\MAB_0{0}403\MAB_5minute_s4_vm4.5_ts{0}.csv'.format(tdata))
    fig, ax = plt.subplots()
    x=[i for i in range(0,100)]
    print(file1.eps,file2.eps,file3.eps,file4.eps)
    #ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
    ax.plot(x, file1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="Fixed 1 machine",lw=1)
    ax.plot(x, file2["mean_reward"][0:100],marker='^', markevery=5,label="2M",color="blue",lw=1)
    ax.plot(x, file3["mean_reward"][0:100],marker='o', markevery=5,label="3M",color="green",lw=1)
    ax.plot(x, file4["mean_reward"][0:100],marker='*', markevery=5,label="auto-scaling",color="orange",lw=1)
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
       
CompareDataOffloadingType('DQN', 1)
