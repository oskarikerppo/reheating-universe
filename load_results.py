import cPickle as pickle

with open('results.pkl', 'rb') as f:
	results = pickle.load(f)
print results