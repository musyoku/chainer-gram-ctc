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

# axisに沿ってexpしながら足し最後にlogを通す操作と同等
def _logsumexp(a, xp, axis=None):
	vmax = xp.amax(a, axis=axis, keepdims=True)
	vmax += xp.log(xp.sum(xp.exp(a - vmax), axis=axis, keepdims=True, dtype=a.dtype))
	return xp.squeeze(vmax, axis=axis)

def _softmax(x, xp):
	val = xp.exp(x - xp.amax(x, axis=2, keepdims=True))
	val /= xp.sum(val, axis=2, keepdims=True)
	return val

# ラベル列からblankを挿入したパスを生成
def _label_to_path(unigram_labels, bigram_labels, blank_symbol, xp):
	batchsize = len(unigram_labels)
	length_unigram = unigram_labels.shape[1]
	path = xp.full((batchsize, length_unigram * 3 + 1), blank_symbol, dtype=np.int32)
	# unigram
	path[:, 1::3] = unigram_labels
	# bigram
	path[:, 2::3] = bigram_labels
	return path

def _log_dot(prob, connection, xp):
	return _logsumexp(prob + connection, xp, axis=2)

def _reverse_path(path, path_length, xp):
	mod = path.shape[1]
	index = xp.arange(0, path.size, mod, dtype=np.int32)[:, None] + (xp.arange(mod) + path_length[:, None])[:, ::-1] % mod
	return xp.take(path, index)

def _move_inputs(prob, input_length, xp):
	seq_length, batchsize, vocab_size = prob.shape
	rotate = (xp.arange(seq_length)[:, None] + input_length) % seq_length
	index = rotate * batchsize + xp.arange(batchsize)
	return xp.take(prob.reshape(seq_length * batchsize, vocab_size), index, axis=0)

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
	return ret[None, :]

# ノードの接続関係を表す行列を作る
def _create_forward_connection_matrix(label_unigram, label_bigram, path_length, max_length, dtype, xp, zero_padding):
	batchsize, _ = label_unigram.shape
	N = max_length

	repeat_mask_unigram = xp.ones((batchsize, N))
	repeat_mask_unigram[:, 1::3] = label_unigram != xp.roll(label_unigram, 1, axis=1)
	repeat_mask_unigram[:, 1] = 1
	repeat_mask_unigram = repeat_mask_unigram[:, None]

	repeat_mask_bigram = xp.ones((batchsize, N))
	repeat_mask_bigram[:, 2::3] = label_bigram != xp.roll(label_bigram, 2, axis=1)
	repeat_mask_bigram[:, 2] = 1
	repeat_mask_bigram = repeat_mask_bigram[:, None]

	arange_mod = xp.arange(N, dtype=dtype) % dtype(3)
	relation_mat = (
		_eye(N, (0,),	xp, dtype) +
		_eye(N, (1, 2),	xp, dtype) * (xp.arange(1, N + 1, dtype=dtype) % dtype(3)).astype(np.bool) +
		_eye(N, (3,), 	xp, dtype) * (arange_mod == 1) * repeat_mask_unigram +
		_eye(N, (6,), 	xp, dtype) * (arange_mod == 2) * repeat_mask_bigram +
		_eye(N, (5, 7), xp, dtype) * (arange_mod == 2)
	)

	# パスの長さを超える部分は0埋め
	mask = path_length[:, None] > xp.arange(max_length)
	relation_mat *= mask[..., None]
	relation_mat *= mask[:, None, :]

	# bigramが存在しない場合、そのノードへの接続を全て切る
	ignore_mask = xp.ones((batchsize, N))
	ignore_mask[:, 2::3] = label_bigram != -1
	relation_mat *= ignore_mask[:, None, :]
	relation_mat *= ignore_mask[..., None]
	return _log_matrix(relation_mat, xp, zero_padding).swapaxes(1, 2)

# ノードの接続関係を表す行列を作る
# 前向きと後ろ向きで形が異なるためので注意
def _create_backward_connection_matrix(label_unigram, label_bigram, path_length, max_length, dtype, xp, zero_padding):
	batchsize, _ = label_unigram.shape
	N = max_length

	repeat_mask_unigram = xp.ones((batchsize, N))
	repeat_mask_unigram[:, 2::3] = label_unigram != xp.roll(label_unigram, 1, axis=1)
	repeat_mask_unigram[:, 2] = 1
	repeat_mask_unigram = repeat_mask_unigram[:, None]
	
	repeat_mask_bigram = xp.ones((batchsize, N))
	repeat_mask_bigram[:, 1::3] = label_bigram != xp.roll(label_bigram, 2, axis=1)
	repeat_mask_bigram[:, 1] = 1
	repeat_mask_bigram[:, 4] = 1
	repeat_mask_bigram = repeat_mask_bigram[:, None]

	arange_mod = xp.arange(N, dtype=dtype) % dtype(3)
	relation_mat = (
		_eye(N, (0,),	xp, dtype) +
		_eye(N, (1,),	xp, dtype) * (xp.arange(1, N + 1, dtype=dtype) % dtype(3)).astype(np.bool) +
		_eye(N, (2,),	xp, dtype) * arange_mod.astype(np.bool) +
		_eye(N, (3,), 	xp, dtype) * (arange_mod == 2) * repeat_mask_unigram +
		_eye(N, (6,), 	xp, dtype) * (arange_mod == 1) * repeat_mask_bigram +
		_eye(N, (5,), 	xp, dtype) * (arange_mod == 0) +
		_eye(N, (7,), 	xp, dtype) * (arange_mod == 2)
	)

	# パスの長さを超える部分は0埋め
	mask = path_length[:, None] > xp.arange(max_length)
	relation_mat *= mask[..., None]
	relation_mat *= mask[:, None, :]

	# bigramが存在しない場合、そのノードへの接続を全て切る
	ignore_mask = xp.ones((batchsize, N))
	ignore_mask[:, 1::3] = label_bigram != -1
	relation_mat *= ignore_mask[:, None, :]
	relation_mat *= ignore_mask[..., None]

	return _log_matrix(relation_mat, xp, zero_padding).swapaxes(1, 2)

def _compute_transition_probability(yseq, input_length, label_unigram, length_unigram, label_bigram, length_bigram, path, path_length, xp, zero_padding):
	dtype = np.float32
	forward_prob = _log_matrix(xp.eye(path.shape[1], dtype=dtype)[0], xp, zero_padding)[None, :]
	backward_prob = forward_prob
	offset = xp.arange(0, yseq[0].size, yseq[0].shape[1], dtype=path.dtype)[:, None]

	index = offset + path
	forward_connection = _create_forward_connection_matrix(label_unigram, label_bigram, path_length, path.shape[1], dtype, xp, zero_padding)
	prob = xp.empty((len(yseq),) + index.shape, dtype=dtype)

	# forward computation.
	for i, y in enumerate(yseq):
		# calc forward probability in log scale
		forward_prob = xp.take(y, index) + _log_dot(forward_prob[:, None, :], forward_connection, xp)
		prob[i] = forward_prob

	index_rev = offset + _reverse_path(path, path_length, xp)

	# rotate yseq with path_length
	yseq_rev = _move_inputs(yseq, input_length, xp)[::-1]
	backward_connection = _create_backward_connection_matrix(_reverse_path(label_unigram, length_unigram, xp), 
		_reverse_path(label_bigram, length_bigram, xp), path_length, path.shape[1], dtype, xp, zero_padding)

	# move to back.
	prob = _move_inputs(prob, input_length, xp)

	# backward computation.
	mod = path.shape[1]
	backward_prob_index = xp.arange(0, path.size, mod, dtype=np.int32)[:, None] + (xp.arange(mod) - path_length[:, None]) % mod
	for i, y_rev in enumerate(yseq_rev):
		# calc backward probability
		backward_prob = _log_dot(backward_prob[:, None, :], backward_connection, xp)
		prob[-i - 1] += xp.take(backward_prob[:, ::-1], backward_prob_index)
		backward_prob = xp.take(y_rev, index_rev) + backward_prob

	# move to front.
	return _move_inputs(prob, -input_length, xp)

def _compute_label_probability(num_output_units, path, path_length, transition_prob, xp, zero_padding):
	label_prob = _log_matrix(xp.zeros((len(path), num_output_units), dtype=transition_prob.dtype), xp, zero_padding)
	ret = xp.empty((len(transition_prob),) + label_prob.shape, dtype=label_prob.dtype)
	ret[...] = label_prob
	if xp == np:
		for batch_idx in six.moves.range(len(path)):
			target_path = path[batch_idx][0:path_length[batch_idx]]
			chars = {unit_idx for unit_idx in target_path}
			for unit_idx in chars:
				if unit_idx == -1:
					continue
				ret[:, batch_idx, unit_idx] = _logsumexp(transition_prob[:, batch_idx, 0:path_length[batch_idx]][:, target_path == unit_idx], np, axis=1)
	else:
		for i, prob_t in enumerate(transition_prob):
			cuda.cupy.ElementwiseKernel(
				'raw T prob, raw I path, raw I path_length, I num_units, I max_path_length',
				'T z',
				'''
				T sum_prob = z;
				I unit_idx = i % num_units;
				I batch_idx = i / num_units;
				int ind[2] = {batch_idx, -1};
				for (int path_idx = 0; path_idx < max_path_length; ++path_idx) {
					ind[1] = path_idx;
					if (path_idx < path_length[ind[0]] && path[ind] == unit_idx) {
						T p = prob[ind];
						T a = p, b = sum_prob;
						if (sum_prob > p) {
							a = sum_prob;
							b = p;
						}
						sum_prob = a + log(1 + exp(b - a));
					}
				}
				z = sum_prob;
				''',
				'reduce_probability')(prob_t, path, path_length, label_prob.shape[1], path.shape[1], ret[i])
	return ret

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
		label_type_unigram = in_types[2]
		label_type_bigram = in_types[3]
		type_check.expect(label_type_unigram.dtype == np.int32)
		type_check.expect(label_type_bigram.dtype == np.int32)

		x_basetype = in_types[4]  # TODO(oktua): Check x_basetype size

		for i in six.moves.range(4, len(in_types)):
			x_type = in_types[i]
			type_check.expect(
				x_type.dtype == np.float32,
				x_type.shape == x_basetype.shape,
			)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		self.input_length = inputs[0]
		length_unigram = inputs[1]
		label_unigram = inputs[2]
		label_bigram = inputs[3]
		length_bigram = length_unigram
		xs = inputs[4:]

		if chainer.is_debug():
			# batch size check.
			assert len(xs[0]) == len(label_unigram)
			assert len(xs[0]) == len(self.input_length)
			assert len(xs[0]) == len(length_unigram)

			# length check.
			assert len(xs) >= xp.max(self.input_length)
			assert len(label_unigram[0]) >= xp.max(length_unigram)

			# unit check
			assert xs[0].shape[1] > xp.max(label_unigram)
			assert xs[0].shape[1] > xp.max(label_bigram)
			assert xs[0].shape[1] > self.blank_symbol

		self.path_length = length_unigram * 3 + 1

		yseq_shape = (len(xs),) + xs[0].shape
		self.yseq = _softmax(xp.vstack(xs).reshape(yseq_shape), xp)
		log_yseq = _log_matrix(self.yseq, xp, self.zero_padding)
		self.path = _label_to_path(label_unigram, label_bigram, self.blank_symbol, xp)
		self.prob_trans = _compute_transition_probability(log_yseq, self.input_length, 
			label_unigram, length_unigram, label_bigram, length_bigram, self.path, self.path_length, xp, self.zero_padding)

		loss = -_logsumexp(self.prob_trans[0], xp, axis=1)
		if self.reduce == 'mean':
			loss = utils.force_array(xp.mean(loss))
		return loss,

	def backward(self, inputs, grad_output):
		xp = cuda.get_array_module(inputs[0])
		batch_size = len(inputs[2])

		total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
		label_prob = _compute_label_probability(self.yseq.shape[2], self.path, self.path_length, self.prob_trans, xp, self.zero_padding)
		self.yseq -= xp.exp(label_prob - total_probability[:, None])
		if self.reduce == 'mean':
			self.yseq *= grad_output[0] / batch_size
		else:
			self.yseq *= grad_output[0][..., None]
		# mask
		self.yseq *= (xp.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
		return (None, None, None, None) + tuple([y for y in self.yseq])

# xsはリスト
def gram_ctc(xs, label_unigram, label_bigram, blank_symbol, input_length=None, length_unigram=None, reduce='mean'):
	if not isinstance(xs, collections.Sequence):
		raise TypeError('xs must be a list of Variables')
	if not isinstance(blank_symbol, int):
		raise TypeError('blank_symbol must be non-negative integer.')
	assert blank_symbol >= 0
	assert blank_symbol < xs[0].shape[1]
	assert len(xs[0].shape) == 2
	assert label_unigram.shape[1] == label_bigram.shape[1]

	if input_length is None:
		xp = cuda.get_array_module(xs[0].data)
		input_length = variable.Variable(xp.full((len(xs[0].data),), len(xs), dtype=np.int32))
		length_unigram = variable.Variable(xp.full((len(label_unigram.data),), len(label_unigram.data[0]), dtype=np.int32))

	return GramCTC(blank_symbol, reduce)(input_length, length_unigram, label_unigram, label_bigram, *xs)
