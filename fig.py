import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def func(x1,y1,x2):
    zy = []
    for i in x2:
        tmp = [abs(j-i)+1000*(j>i) for j in x1]
        inx1 = tmp.index(min(tmp))
        tmp = [abs(j-i)+1000*(j<i) for j in x1]
        inx2 = tmp.index(min(tmp))
        if inx1==inx2:
            zy.append(y1[inx1])
        else:
            k = (y1[inx1]-y1[inx2])/(x1[inx1]-x1[inx2])
            zy.append(y1[inx1]+k*(-x1[inx1]+i))
    return zy


def datanaly(log_dir):
    all_x = []
    all_y = []
    vert_x = []
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    for i in x:
    	all_x.append(i)
    	appended_val = x[-1]

    vert_x.append(appended_val)
    for i in y:
    	all_y.append(i)
    return x, y, vert_x


def fited_data(x,y):
    all_x = []
    all_y = []
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    for i in x:
    	all_x.append(i)
    	appended_val = x[-1]
    for i in y:
    	all_y.append(i)
    return all_x, all_y
#####################################################################
log_dir = './ddpg'
py = []
x,y,_ = datanaly(log_dir+str(2))
py.append(y)
for i in [1,3,4]:
    y = np.load('generalize_yddpg'+str(i)+'.npy')
    py.append(y)

mean_var = []
max_var = []
min_var = []
for i in range(len(x)):
    sp = []
    pinx = 0
    for j in range(3):
        try:
            sp.append(py[j][i])
            pinx +=1
        except Exception as e:
            pass
    mean_var.append(sum(sp)/pinx)
    max_var.append(max(sp))
    min_var.append(min(sp))
#fitx, fity = fited_data(x, mean_var)

plt.fill_between(x,min_var,max_var,color = 'b', alpha=0.5)
plt.plot(x, mean_var,color = 'b',label='DDPG')
#####################################################################
log_dir = './ppo'
py = []
lx = []
px = []
x,y,_ = datanaly(log_dir+str(1))
py.append(y)
for i in range(2,5):
    y = np.load('generalize_yppo'+str(i)+'.npy')
    py.append(y)


mean_var = []
max_var = []
min_var = []
for i in range(len(x)):
    sp = []
    pinx = 0
    for j in range(3):
        try:
            sp.append(py[j][i])
            pinx +=1
        except Exception as e:
            pass
    mean_var.append(sum(sp)/pinx)
    max_var.append(max(sp))
    min_var.append(min(sp))
fited_x,fited_upy = fited_data(x,max_var)
fited_x,fited_suby = fited_data(x,min_var)
fited_x,fited_y = fited_data(x,mean_var)
plt.fill_between(fited_x, fited_suby, fited_upy, color = 'r', alpha=0.5)
plt.plot(fited_x,fited_y, color = 'r',label='PPO2')
#####################################################################

#for i in vert_x:
	#plt.axvline(x=i, linestyle='--', color='#ccc5c6', label='leg increment')
plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
plt.title('robot Smoothed')
plt.legend()
#plt.savefig("figure.png")
#plt.savefig(save_name + ".eps")
plt.show()


'''for i in range(1,4):
    x,y = datanaly(log_dir+str(i))
    if len(lx)>len(x):
        lx = x

for i in range(1,4):
    x,y = datanaly(log_dir+str(i))
    if len(lx)>len(x):
        py.append(func(x,y,lx))
    else:
        py.append(y)'''
'''log_dir = './ppo'
py = []
x1,y1 = datanaly(log_dir+str(1))

x2,y2 = datanaly(log_dir+str(4))
zy = func(x2,y2,x1)
plt.plot(x2,y2,'r')
plt.plot(x1,zy,'b')
np.save('generalize_yppo4.npy',zy)
plt.show()'''
'''
log_dir = './ppo'
py3 = []
for i in range(1,3):
    all_x, all_y, vert_x = datanaly(log_dir+str(i))
    plt.plot(all_x,all_y,color = 'r',label='PPO2')
#####################################################################
#for i in vert_x:
	#plt.axvline(x=i, linestyle='--', color='#ccc5c6', label='leg increment')
plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
plt.title('hand' + " Smoothed")
plt.legend()
#plt.savefig("figure.png")
#plt.savefig(save_name + ".eps")
plt.show()
'''
