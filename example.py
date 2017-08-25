# coding: utf-8
from __future__ import print_function
import cupy
import numpy
import gram_ctc
from chainer import Variable, functions

# np.random.seed(0)
xp = cupy

vocab_size = 10
vocab_size_ctc = 6
seq_length = 20
batchsize = 2

# 正解ラベル列
# 例：["h", "e", "l", "l", "o"]
label_unigram = xp.asarray([
	[1, 2, 3, 3, 5],
	[2, 4, 3, 0, 0],
], dtype=xp.int32)

# 正解ラベル列のbigram
# unigram列のあるラベルとその左側のラベルとのペアのID
# 	unigram列の先頭のbigramは存在しないので-1（こうしたほうが実装しやすい）
# 全bigramを考えるのは非現実的なので無視するbigramには-1
# 例：
# 	unigram: ["h", "e", "l", "l", "o"]
# 	bigram:  [-1, "he", "el", "ll", "lo"]
label_bigram = xp.asarray([
	[-1, 6, -1, 7, -1],
	[-1, -1, 8, 0, 0],
], dtype=xp.int32)
blank_symbol = 0

length_unigram = xp.asarray([5, 3], dtype=xp.int32)
path_length = length_unigram * 3 + 1

# ダミーの入力データ
x_data = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, 1, seq_length)).astype(xp.float32)

def run_ctc():
	x = Variable(x_data)
	x = x[:, :vocab_size_ctc]
	x = functions.swapaxes(x, 1, 3)
	x = functions.reshape(x, (batchsize, -1))
	x = functions.split_axis(x, seq_length, axis=1)

	x_length = Variable(xp.asarray([seq_length, seq_length // 2], dtype=xp.int32))	# 入力系列長は適当

	loss_ctc = functions.connectionist_temporal_classification(x, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
	loss_ctc.backward()

def run_gram_ctc():
	x = Variable(x_data)
	x = functions.swapaxes(x, 1, 3)
	x = functions.reshape(x, (batchsize, -1))
	x = functions.split_axis(x, seq_length, axis=1)

	x_length = Variable(xp.asarray([seq_length, seq_length // 2], dtype=xp.int32))	# 入力系列長は適当

	loss_gram_ctc = gram_ctc.gram_ctc(x, label_unigram, label_bigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
	loss_gram_ctc.backward()

if __name__ == "__main__":
	run_ctc()
	run_gram_ctc()
