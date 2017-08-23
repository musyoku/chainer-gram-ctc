from __future__ import print_function
import numpy as np
import ctc
import gram_ctc
from chainer import Variable, functions

def a():
	vocab_size = 6
	seq_length = 5
	batchsize = 2
	x = np.random.normal(0, 1, size=batchsize*vocab_size*seq_length*2).reshape((batchsize, vocab_size, 1, seq_length * 2)).astype(np.float32)
	x = Variable(x)
	x = functions.swapaxes(x, 1, 3)
	x = functions.reshape(x, (batchsize, -1))
	x = functions.split_axis(x, seq_length * 2, axis=1)
	t = np.asarray([
		[1, 2, 3, 4, 5],
		[1, 2, 3, 0, 0],
	], dtype=np.int32)
	t = Variable(t)
	x_length = Variable(np.asarray([seq_length * 2, seq_length]))
	t_length = Variable(np.asarray([5, 3]))
	loss = ctc.connectionist_temporal_classification(x, t, 0, x_length, t_length)	

def b():
	unigram_labels = np.asarray([
		[1, 2, 2, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	bigram_labels = np.asarray([
		[6, -1, 7, 8],
		[6, 9, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(unigram_labels, bigram_labels, blank_symbol, np)

	unigram_length = np.asarray([5, 3])
	bigram_length = unigram_length - 1
	path_length = unigram_length * 2 + 1 + bigram_length

	relation_mat = gram_ctc._create_recurrence_relation_matrix(unigram_labels, bigram_labels, path_length, path.shape[1], np.float32, np, zero_padding=-5)

def c():
	unigram_labels = np.asarray([
		[1, 2, 4, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	bigram_labels = np.asarray([
		[6, 7, -1, -1],
		[6, -1, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(unigram_labels, bigram_labels, blank_symbol, np)

	unigram_length = np.asarray([5, 3])
	bigram_length = unigram_length - 1
	path_length = unigram_length * 2 + 1 + bigram_length
	print("path_length", path_length)

	vocab_size = 10
	seq_length = 5
	batchsize = 2
	xs = np.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, 1, seq_length)).astype(np.float32)
	xs = Variable(xs)
	xs = functions.swapaxes(xs, 1, 3)
	xs = functions.reshape(xs, (batchsize, -1))
	xs = functions.split_axis(xs, seq_length, axis=1)
	xs = [x.data for x in xs]

	x_length = np.asarray([seq_length, seq_length // 2], dtype=np.int32)
	yseq_shape = (len(xs),) + xs[0].shape
	print(yseq_shape)
	yseq = gram_ctc._softmax(np.vstack(xs).reshape(yseq_shape), np)
	print(yseq)

	zero_padding = -100
	log_yseq = gram_ctc._log_matrix(yseq, np, zero_padding)
	prob_trans = gram_ctc._compute_transition_probability(log_yseq, x_length, unigram_labels, unigram_length, 
		bigram_labels, bigram_length, path, path_length, np, zero_padding)



if __name__ == "__main__":
	# a()
	# b()
	c()
