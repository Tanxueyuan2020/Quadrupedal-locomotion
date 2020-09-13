import os
import sys
import gym
import gym_real
import numpy as np
import matplotlib.pyplot as plt
import datetime
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from gym.envs.mujoco import mujoco_env
import cv2
from PIL import ImageGrab
import imageio

def image_check(inx):
	printscr = ImageGrab.grab(bbox=(30,40,2500,1300))
	tmp = np.array(printscr)
	tmp = cv2.resize(tmp,(400,200))
	#`if (inx%10==0)&(inx>10):

		#cv2.imwrite('./data/a'+str(inx)+'.png',tmp)
	return tmp

if __name__ == "__main__":
	env_name = 'Real-v0'
	file_name = 'hand'
	#gif_name = "PPO2_" + env_name
	gif_name = "DDPG_" + env_name
	save_str = "./tmp_gif/" + gif_name + '.gif'

	if file_name[:3] == "mod":
		model_name = file_name
	else:
		dirpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
		log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
		model_name = os.path.join(dirpath, file_name)

	env = gym.make(env_name)
	#model = PPO2.load(model_name)
	model = DDPG.load(model_name)
	images = []
	obs = env.reset()
	#env.render(width=200, height=200)
	for i in range(300):
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render(width=900,height=500)
		images.append(image_check(i))
		env.render()
	print("creating gif...")
	gifimages = [np.array(img) for i, img in enumerate(images[2:]) if i % 2 == 0]
	imageio.mimsave(save_str, gifimages, fps=29)
	print("gif created...")
