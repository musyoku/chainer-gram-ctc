from __future__ import print_function
import numpy as np
import gram_ctc as gctc
from chainer import Variable, functions

def a():
	vocab_size = 6
	seq_length = 5
	batchsize = 1
	x = np.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, 1, seq_length)).astype(np.float32)
	x = Variable(x)
	x = functions.swapaxes(x, 1, 3)
	x = functions.reshape(x, (batchsize, -1))
	x = functions.split_axis(x, seq_length, axis=1)
	t = np.asarray([
		[1, 2, 3, 4, 5]
	], dtype=np.int32)
	t = Variable(t)
	loss = gctc.bigram_connectionist_temporal_classification(x, t, 0)	

def b():
	unigram_labels = np.asarray([
		[1, 2, 3, 3, 5]
	], dtype=np.int32)
	bigram_labels = np.asarray([
		[6, -1, 7, 8]
	], dtype=np.int32)
	blank_symbol = 0
	path = gctc._label_to_path(unigram_labels, bigram_labels, blank_symbol, np)
	print(path)
	label_length = np.asarray([unigram_labels.shape[1]])
	path_length = 2 * label_length + 1
	relation_mat = gctc._create_recurrence_relation_matrix(unigram_labels, path_length, path.shape[1], np.float32, np, zero_padding=-5)
	print(relation_mat)

if __name__ == "__main__":
	a()
	b()
