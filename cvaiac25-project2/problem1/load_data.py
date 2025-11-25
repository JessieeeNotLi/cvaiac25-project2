# Computer Vision and Artificial Intelligence for Autonomous Cars
# Material for Problem 1 of Project 2

import pickle

def load_data(data_path):
	''' 
    Load data dictionary from data_path.
    '''
	with open(data_path, 'rb') as fp:
	    data = pickle.load(fp)
	return data
