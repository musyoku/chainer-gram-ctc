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
	labels_unigram = np.asarray([
		[1, 1, 1, 1, 1],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	labels_bigram = np.asarray([
		[-1, 6, 6, 6, 6],
		[-1, 6, 9, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(labels_unigram, labels_bigram, blank_symbol, np)
	print(path)

	length_unigram = np.asarray([5, 3])
	length_bigram = length_unigram - 1
	path_length = length_unigram * 2 + 1 + length_bigram

	relation_mat = gram_ctc._create_forward_connection_matrix(labels_unigram, labels_bigram, path_length, path.shape[1], np.float32, np, zero_padding=-5)

def c():
	labels_unigram = np.asarray([
		[1, 2, 4, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	labels_bigram = np.asarray([
		[-1, 6, -1, 7, 8],
		[-1, 6, 9, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(labels_unigram, labels_bigram, blank_symbol, np)

	length_unigram = np.asarray([5, 3])
	length_bigram = length_unigram - 1
	path_length = length_unigram * 2 + 1 + length_bigram
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
	prob_trans = gram_ctc._compute_transition_probability(log_yseq, x_length, labels_unigram, length_unigram, 
		labels_bigram, length_bigram, path, path_length, np, zero_padding)

def d():
	np.set_printoptions(linewidth=200, precision=1)
	np.random.seed(0)
	labels_unigram = np.asarray([
		[1, 2, 2, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	labels_bigram = np.asarray([
		[-1, 6, -1, 7, 8],
		[-1, 6, 9, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0

	length_unigram = np.asarray([5, 3], dtype=np.int32)
	length_bigram = length_unigram - 1
	path_length = length_unigram * 2 + 1 + length_bigram

	vocab_size = 10
	vocab_size_ctc = 6
	seq_length = 20
	batchsize = 2
	x = np.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, 1, seq_length)).astype(np.float32)
	x[:, vocab_size_ctc:] = -10000000000.0

	# if True:
	# 	xs = Variable(x)
	# 	xs = functions.swapaxes(xs, 1, 3)
	# 	xs = functions.reshape(xs, (batchsize, -1))
	# 	xs = functions.split_axis(xs, seq_length, axis=1)

	# 	x_length = Variable(np.asarray([seq_length, seq_length // 2], dtype=np.int32))

	# 	print("Gram-CTC:")
	# 	loss_gram_ctc = gram_ctc.gram_ctc(xs, labels_unigram, labels_bigram, blank_symbol, x_length, Variable(length_unigram), reduce="no")

	if True:
		xs = Variable(x[:, :vocab_size_ctc])
		xs = functions.swapaxes(xs, 1, 3)
		xs = functions.reshape(xs, (batchsize, -1))
		xs = functions.split_axis(xs, seq_length, axis=1)

		x_length = Variable(np.asarray([seq_length, seq_length // 2], dtype=np.int32))

		print("CTC:")
		loss_ctc = ctc.connectionist_temporal_classification(xs, labels_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="no")

	# print(loss_ctc)
	# print(loss_gram_ctc)


if __name__ == "__main__":
	# a()
	b()
	# c()
	# d()
