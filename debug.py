from __future__ import print_function
import cupy
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
	np.set_printoptions(linewidth=200, precision=1)
	xp = cupy
	label_unigram = xp.asarray([
		[1, 2, 5, 3, 4, 0],
		[2, 4, 3, 0, 0, 0],
	], dtype=xp.int32)
	label_bigram = xp.asarray([
		[-1, 7, 8, 9, -1, 0],
		[-1, 6, 9, 0, 0, 0],
	], dtype=xp.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(label_unigram, label_bigram, blank_symbol, xp)
	print(path)

	length_unigram = xp.asarray([5, 3])
	length_bigram = length_unigram
	path_length = length_unigram * 3 + 1

	relation_mat = gram_ctc._create_forward_connection_matrix(label_unigram, label_bigram, path_length, path.shape[1], xp.float32, xp, zero_padding=-5)
	relation_mat = gram_ctc._create_backward_connection_matrix(gram_ctc._reverse_path(label_unigram, length_unigram, xp), gram_ctc._reverse_path(label_bigram, length_bigram, xp), path_length, path.shape[1], xp.float32, xp, zero_padding=-5)

def c():
	label_unigram = np.asarray([
		[1, 2, 4, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=np.int32)
	label_bigram = np.asarray([
		[-1, 6, -1, 7, 8],
		[-1, 6, 9, 0, 0],
	], dtype=np.int32)
	blank_symbol = 0
	path = gram_ctc._label_to_path(label_unigram, label_bigram, blank_symbol, np)

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
	prob_trans = gram_ctc._compute_transition_probability(log_yseq, x_length, label_unigram, length_unigram, 
		label_bigram, length_bigram, path, path_length, np, zero_padding)

def d():
	np.set_printoptions(linewidth=200, precision=1)
	xp = cupy
	# np.random.seed(0)
	label_unigram = xp.asarray([
		[1, 2, 2, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=xp.int32)
	label_bigram = xp.asarray([
		[-1, -1, -1, -1, -1],
		[-1, -1, -1, 0, 0],
	], dtype=xp.int32)
	blank_symbol = 0

	length_unigram = xp.asarray([5, 3], dtype=xp.int32)
	length_bigram = length_unigram
	path_length = length_unigram * 2 + 1 + length_bigram

	vocab_size = 10
	vocab_size_ctc = 6
	seq_length = 20
	batchsize = 2
	x = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, 1, seq_length)).astype(xp.float32)
	x[:, vocab_size_ctc:] = -10000000000.0		# bigramノードの確率が0になるように

	if True:
		xs = Variable(x)
		in_data = functions.swapaxes(xs, 1, 3)
		in_data = functions.reshape(in_data, (batchsize, -1))
		in_data = functions.split_axis(in_data, seq_length, axis=1)

		x_length = Variable(xp.asarray([seq_length, seq_length // 2], dtype=xp.int32))

		print("Gram-CTC:")
		loss_gram_ctc = gram_ctc.gram_ctc(in_data, label_unigram, label_bigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
		loss_gram_ctc.backward()
		grad_ctc = xs.grad

	if True:
		xs = Variable(x)
		in_data = xs[:, :vocab_size_ctc]
		in_data = functions.swapaxes(xs, 1, 3)
		in_data = functions.reshape(in_data, (batchsize, -1))
		in_data = functions.split_axis(in_data, seq_length, axis=1)

		x_length = Variable(xp.asarray([seq_length, seq_length // 2], dtype=xp.int32))

		print("CTC:")
		loss_ctc = ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
		loss_ctc.backward()
		grad_gram_ctc = xs.grad


	print(loss_ctc)
	print(loss_gram_ctc)

	print(xp.sum(xp.abs(grad_ctc - grad_gram_ctc)))




if __name__ == "__main__":
	# a()
	# b()
	# c()
	d()
