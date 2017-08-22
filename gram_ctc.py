# coding: utf-8
import collections
import numpy as np
import six

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable

def _logsumexp(a, xp, axis=None):
	vmax = xp.amax(a, axis=axis, keepdims=True)
	vmax += xp.log(xp.sum(xp.exp(a - vmax), axis=axis, keepdims=True, dtype=a.dtype))
	return xp.squeeze(vmax, axis=axis)

def _softmax(x, xp):
	val = xp.exp(x - xp.amax(x, axis=2, keepdims=True))
	val /= xp.sum(val, axis=2, keepdims=True)
	return val

def _label_to_path(unigram_labels, bigram_labels, blank_symbol, xp):
	batchsize = len(unigram_labels)
	unigram_length = unigram_labels.shape[1]
	bigram_length = bigram_labels.shape[1]
	path = xp.full((batchsize, unigram_length * 2 + 1 + bigram_length), blank_symbol, dtype=np.int32)
	# unigram
	path[:, 3::3] = unigram_labels[:, 1:]
	path[:, 1] = unigram_labels[:, 0]
	# bigram
	path[:, 4::3] = bigram_labels
	return path

def _log_dot(prob, rr, xp):
	return _logsumexp(prob + xp.swapaxes(rr, 1, 2), xp, axis=2)

def _move_label_to_back(path, path_length, xp):
	s1 = path.shape[1]  # TODO(okuta): Change name
	index = (xp.arange(0, path.size, s1, dtype=np.int32)[:, None] +
			 (xp.arange(s1) + path_length[:, None])[:, ::-1] % s1)
	return xp.take(path, index)

def _move_inputs(prob, input_length, xp):
	seq, batch, ch = prob.shape
	rotate = (xp.arange(seq)[:, None] + input_length) % seq
	index = rotate * batch + xp.arange(batch)
	return xp.take(prob.reshape(seq * batch, ch), index, axis=0)

def _log_matrix(x, xp, zero_padding):
	if xp == np:
		res = np.ma.log(x).filled(fill_value=zero_padding)
	else:
		filled = cuda.cupy.ElementwiseKernel(
			'T x, T e', 'T y',
			'y = x == 0 ? e : log(x)',
			'_log_matrix')
		res = filled(x, zero_padding)
	return res.astype(np.float32)

def _eye(N, k, xp, dtype):
	ret = xp.eye(N, k=k[-1], dtype=dtype)
	for diagonal in k[:-1]:
		ret += xp.eye(N, k=diagonal, dtype=dtype)
	return ret

# ノードの接続関係を表す行列を作る
def _create_connection_matrix(unigram_label, bigram_label, path_length, max_length, dtype, xp, zero_padding=-10000000000.0):
	batchsize, length_u = unigram_label.shape
	length_b = bigram_label.shape[1]
	N = max_length

	repeat_mask = xp.ones((batchsize, N))
	repeat_mask[:, 0::3] = unigram_label != xp.roll(unigram_label, 1, axis=1)
	repeat_mask[:, 0] = 1
	repeat_mask = repeat_mask[:, None]

	relation_mat = (
		_eye(N, (0,),	xp, dtype)[None, :] +
		_eye(N, (1,), 	xp, dtype)[None, :] * (xp.arange(-1, N - 1, dtype=dtype) % dtype(3)).astype(np.bool) +
		_eye(N, (2,),	xp, dtype)[None, :] * (xp.arange(-1, N - 1, dtype=dtype) % dtype(3)).astype(np.bool) * repeat_mask +
		_eye(N, (3,), 	xp, dtype)[None, :] * (xp.arange(N, dtype=dtype) % dtype(3) == 0) * repeat_mask +
		_eye(N, (5,), 	xp, dtype)[None, :] * (xp.arange(N, dtype=dtype) % dtype(3) == 1) + 
		_eye(N, (6, 7), xp, dtype)[None, :] * (xp.arange(N, dtype=dtype) % dtype(3) == 1) * xp.roll(repeat_mask, 4, axis=2)
	)

	relation_mat[:, 0, 1] = 1
	relation_mat[:, 0, 2] = 0
	relation_mat[:, 0, 3] = 0
	relation_mat[:, 0, 4] = 1
	relation_mat[:, 0, 7] = 0

	# パスの長さを超える部分は0埋め
	relation_mat = relation_mat * (path_length[:, None] > xp.arange(max_length))[..., None]
	relation_mat = relation_mat * (path_length[:, None] > xp.arange(max_length))[:, None, :]

	# bigramが存在しない場合、そのノードへの接続を全て切る
	ignore_mask = xp.ones((batchsize, N))
	ignore_mask[:, 4::3] = bigram_label != -1
	relation_mat *= ignore_mask[:, None, :]

	print(relation_mat)
	print(relation_mat.swapaxes(1, 2))

	return _log_matrix(relation_mat, xp, zero_padding=zero_padding)

def _compute_transition_probability(self, yseq, input_length, label, label_length, path, path_length, xp):
	dtype = np.float32
	forward_prob = _log_matrix(xp.eye(path.shape[1], dtype=dtype)[0], xp)[None, :]
	backward_prob = forward_prob
	offset = xp.arange(0, yseq[0].size, yseq[0].shape[1], dtype=path.dtype)[:, None]

	# prob[i] := forward[i] + backward[-i-1]
	index = offset + path
	forward_connection = _create_connection_matrix(label, path_length, path.shape[1], dtype, xp)
	prob = xp.empty((len(yseq),) + index.shape, dtype=dtype)

	# forward computation.
	for i, y in enumerate(yseq):
		# calc forward probability in log scale
		forward_prob = xp.take(y, index) + _log_dot(forward_prob[:, None, :], forward_connection, xp)
		prob[i] = forward_prob
	r_index = offset + _move_label_to_back(path, path_length, xp)

	# rotate yseq with path_length
	yseq_inv = _move_inputs(yseq, input_length, xp)[::-1]
	brr = _create_connection_matrix(_move_label_to_back(label, label_length, xp), path_length, path.shape[1], dtype, xp)
	# move to back.
	prob = _move_inputs(prob, input_length, xp)

	# backward computation.
	ps1 = path.shape[1]
	backward_prob_index = (
		xp.arange(0, path.size, ps1, dtype=np.int32)[:, None] +
		(xp.arange(ps1) - path_length[:, None]) % ps1)
	for i, y_inv in enumerate(yseq_inv):
		# calc backward probability
		backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
		prob[-i - 1] += xp.take(
			backward_prob[:, ::-1], backward_prob_index)
		backward_prob = xp.take(y_inv, r_index) + backward_prob

	# move to front.
	return _move_inputs(prob, -self.input_length, xp)

class GramCTC(function.Function):

	def __init__(self, blank_symbol, reduce='mean'):
		self.blank_symbol = blank_symbol
		self.zero_padding = -10000000000.0

		if reduce not in ('mean', 'no'):
			raise ValueError(
				"only 'mean' and 'no' are valid "
				"for 'reduce', but '%s' is given" % reduce)
		self.reduce = reduce

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() > 3)  # TODO(okuta): > 3?
		l_type = in_types[2]
		type_check.expect(l_type.dtype == np.int32)

		x_basetype = in_types[3]  # TODO(oktua): Check x_basetype size

		for i in six.moves.range(3, len(in_types)):
			x_type = in_types[i]
			type_check.expect(
				x_type.dtype == np.float32,
				x_type.shape == x_basetype.shape,
			)

	def label_probability(self, label_size, path, path_length, multiply_seq, xp):
		labels_prob = _log_matrix(xp.zeros((len(path), label_size),
											   dtype=multiply_seq.dtype), xp)
		ret = xp.empty(
			(len(multiply_seq),) + labels_prob.shape, dtype=labels_prob.dtype)
		ret[...] = labels_prob
		if xp == np:
			for b in six.moves.range(len(path)):
				target_path = path[b][0:path_length[b]]
				chars = {c for c in target_path}
				for c in chars:
					ret[:, b, c] = _logsumexp(
						multiply_seq[:, b, 0:path_length[b]]
						[:, target_path == c], np, axis=1)
		else:
			for i, multiply in enumerate(multiply_seq):
				cuda.cupy.ElementwiseKernel(
					'raw T x, raw I y, raw I l, I b_max, I c_max',
					'T z',
					'''
					T value = z;
					I c = i % b_max, b = i / b_max;
					int ind[2] = {b, -1};
					for (int index = 0; index < c_max; ++index) {
						ind[1] = index;
						if (ind[1] < l[ind[0]] && y[ind] == c) {
							T xvalue = x[ind];
							T at = xvalue, bt = value;
							if (value > xvalue) {
								at = value;
								bt = xvalue;
							}
							value = at + log(1 + exp(bt - at));
						}
					}
					z = value;
					''',
					'reduce_probability')(multiply, path, path_length,
										  labels_prob.shape[1],
										  path.shape[1], ret[i])
		return ret

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		self.input_length = inputs[0]
		label_length = inputs[1]
		t = inputs[2]
		xs = inputs[3:]

		if chainer.is_debug():
			# batch size check.
			assert len(xs[0]) == len(t)
			assert len(xs[0]) == len(self.input_length)
			assert len(xs[0]) == len(label_length)

			# length check.
			assert len(xs) >= xp.max(self.input_length)
			assert len(t[0]) >= xp.max(label_length)

		self.path_length = 2 * label_length + 1

		yseq_shape = (len(xs),) + xs[0].shape
		self.yseq = _softmax(xp.vstack(xs).reshape(yseq_shape), xp)
		log_yseq = _log_matrix(self.yseq, xp)
		self.path = _label_to_path(t, self.blank_symbol, xp)
		self.prob_trans = _compute_transition_probability(log_yseq, self.input_length, t, label_length, self.path, self.path_length, xp)

		loss = -_logsumexp(self.prob_trans[0], xp, axis=1)
		if self.reduce == 'mean':
			loss = utils.force_array(xp.mean(loss))
		return loss,

	def backward(self, inputs, grad_output):
		xp = cuda.get_array_module(inputs[0])
		batch_size = len(inputs[2])

		total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
		label_prob = self.label_probability(
			self.yseq.shape[2], self.path, self.path_length,
			self.prob_trans, xp)
		self.yseq -= xp.exp(label_prob - total_probability[:, None])
		if self.reduce == 'mean':
			self.yseq *= grad_output[0] / batch_size
		else:
			self.yseq *= grad_output[0][..., None]
		# mask
		self.yseq *= (
			xp.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
		return (None, None, None) + tuple([y for y in self.yseq])


def bigram_connectionist_temporal_classification(x, t, blank_symbol, input_length=None, label_length=None, reduce='mean'):
	if not isinstance(x, collections.Sequence):
		raise TypeError('x must be a list of Variables')
	if not isinstance(blank_symbol, int):
		raise TypeError('blank_symbol must be non-negative integer.')
	assert blank_symbol >= 0
	assert blank_symbol < x[0].shape[1]
	assert(len(x[0].shape) == 2)

	if input_length is None:
		xp = cuda.get_array_module(x[0].data)
		input_length = variable.Variable(xp.full((len(x[0].data),), len(x), dtype=np.int32))
		label_length = variable.Variable(xp.full((len(t.data),), len(t.data[0]), dtype=np.int32))

	return GramCTC(blank_symbol, reduce)(input_length, label_length, t, *x)
