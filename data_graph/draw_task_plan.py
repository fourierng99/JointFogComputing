import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def export_legend(legend, filename="data_graph/fig/quality_legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def draw_task_destination(path, env, tdata,scale, split_legend = 0):
    labels=["combine fuzzy and deep q","deep q learning","fuzzy","random"]
    fig, ax = plt.subplots()
    x=[i for i in range(0,100)]
    files=pd.read_csv(path, index_col = False)
    ax.plot(x,files["server"],label="LS",marker='^', markevery=5,color="orange",lw=1)
    ax.plot(x,files["bus1"],label="VS1",marker="o", markevery=5,color="blue",lw=1)
    ax.plot(x,files["bus2"],label="VS2",marker="P", markevery=5,color="red",lw=1)
    ax.plot(x,files["bus3"],label="VS3",marker="*", markevery=5,color="green",lw=1)

    legend = plt.figlegend(loc='lower center', bbox_to_anchor=(0., 1.02, 1., .102), ncol=6)
    
    #ax.set_title("Fuzzy-Controller in Deep Q learning")
    ax.set_xlabel("Time slots",fontsize=15)
    ax.set_ylabel("Number of tasks",fontsize=15)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.subplots_adjust(left = 0.145, right=0.945)
    ax.set_ylim(0,1500)
    #plt.show()
    plt.savefig("data_graph/fig/5_task_destination_{0}_scale_{2}_ts{1}.pdf".format(env,tdata, scale))
    if(split_legend == 1):
        export_legend(legend)

draw_task_destination('result\Data1\DQN_01403/test\DQN_config_parameter_s4_vm4.5.csv','dqn',1,0,1)
draw_task_destination('result\MAB_1\MAB_01403\MAB_config_parameter_s4.csv','mab',1,0,0)
draw_task_destination('result\Data2\DQN_02403/test\DQN_config_parameter_s4_vm4.5.csv','dqn',2,0,0)
draw_task_destination('result\MAB_2\MAB_02403\MAB_config_parameter_s4.csv','mab',2,0,0)

draw_task_destination('result\Data_autoscale_1\DQN_01413/test\DQN_config_parameter_s4.csv','dqn',1,1,0)
draw_task_destination('result\Data_autoscale_2\DQN_02413/test\DQN_config_parameter_s4.csv','dqn',2,1,0)