from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
# from generate import generate
from generate import generate
import scipy.ndimage
import scipy.io as scio



BATCH_SIZE = 18
EPOCHES = 4
LOGGING = 20

MODEL_SAVE_PATH = './models/'

def main():
	print('\nBegin to generate pictures ...\n')
	path = r'C:\data\SICE_256\test'

	T=[]
	for i in range(30):
		index = i + 1
		ue_path = os.path.join(path, 'ue', str(index)+'.png')
		oe_path = os.path.join(path, 'oe', str(index)+'.png')

		t = generate(oe_path, ue_path, MODEL_SAVE_PATH, index, output_path = './results/', format='.png')

		T.append(t)
		print("%s time: %s" % (index, t))
	scio.savemat('time.mat', {'T': T})


if __name__ == '__main__':
	main()
