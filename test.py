import math
import unittest
import numpy
import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import gram_ctc

class GramCTCTestBase(object):

    def setUp(self):
        self.blank_symbol = 3
        self.x = numpy.random.uniform(-1, 1, (4, 2, 4)).astype(numpy.float32)
        self.t = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        self.bigram = numpy.array([[-1, 2], [-1, -1]]).astype(numpy.int32)
        self.l = gram_ctc._label_to_path(self.t, self.bigram, self.blank_symbol, numpy)

        self.x_length = numpy.full((len(self.x[0]),), len(self.x), dtype='i')
        self.l_length = numpy.full((len(self.t),), len(self.t[0]), dtype='i')
        self.use_length = True
        if self.reduce == 'mean':
            self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        else:
            self.gy = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)

    def alpha(self, x, l, t, u):
        if u < 0:
            return 0.0
        if l[u] == -1:  # 存在しないbigram
            return 0.0
        if t == 0:  # 初期化
            if u == 0:
                return x[0][self.blank_symbol]
            elif u == 1:
                return x[0][l[u]]
            elif u == 5:
                return x[0][l[u]]
            return 0.0
        if l[u] == self.blank_symbol:
            return (x[t][self.blank_symbol] *
                    (self.alpha(x, l, t - 1, u - 2) +
                     self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))
        if u % 3 == 1:  # unigram
            if u >= 3 and l[u] == l[u - 3]:
                return (x[t][l[u]] *
                        (self.alpha(x, l, t - 1, u - 2) +
                         self.alpha(x, l, t - 1, u - 1) +
                         self.alpha(x, l, t - 1, u)))
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 3) +
                     self.alpha(x, l, t - 1, u - 2) +
                     self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))
        # bigram
        if u >= 6 and l[u] == l[u - 6]:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 7) +
                     self.alpha(x, l, t - 1, u - 5) +
                     self.alpha(x, l, t - 1, u)))

        return (x[t][l[u]] *
                (self.alpha(x, l, t - 1, u - 7) +
                 self.alpha(x, l, t - 1, u - 6) +
                 self.alpha(x, l, t - 1, u - 5) +
                 self.alpha(x, l, t - 1, u)))

    # recursive forward computation.
    def _alpha(self, x, l, t, u):
        if u < 0:
            return 0.0
        if t == 0:
            if u == 0:
                return x[0][self.blank_symbol]
            elif u == 1:
                return x[0][l[1]]
            else:
                return 0.0
        elif l[u] == self.blank_symbol or l[u] == l[u - 2]:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))
        else:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 2) +
                     self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))

    def check_forward(self, unigram_data, bigram_data, xs_data, l_length, x_length):
        x = tuple(chainer.Variable(x_data) for x_data in xs_data)
        unigram = chainer.Variable(unigram_data)
        bigram = chainer.Variable(bigram_data)

        args = (x, unigram, bigram, self.blank_symbol)
        if self.use_length:
            args += (chainer.Variable(x_length), chainer.Variable(l_length))
        loss = gram_ctc.gram_ctc(*args, reduce=self.reduce).data

        # compute expected value by recursive computation.
        xp = cuda.get_array_module(self.x)
        xt = xp.swapaxes(self.x, 0, 1)
        for b in range(xt.shape[0]):
            for t in range(xt.shape[1]):
                xt[b][t] = numpy.exp(xt[b][t]) / numpy.sum(numpy.exp(xt[b][t]))
        batch_size = xt.shape[0]
        path_length = 3 * l_length + 1
        loss_expect = xp.zeros((batch_size,), dtype=xp.float32)

        for i in range(batch_size):
            xtb, lb, xlb, plb = xt[i], self.l[i], x_length[i], path_length[i]
            loss_expect[i] = -math.log(
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 1)) +
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 2)) +
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 3)))
            # print(self.alpha(xtb, lb, int(xlb - 1), int(plb - 1)))
            # print(self.alpha(xtb, lb, int(xlb - 1), int(plb - 2)))
            # print(self.alpha(xtb, lb, int(xlb - 1), int(plb - 3)))
        if self.reduce == 'mean':
            loss_expect = xp.mean(loss_expect)
        testing.assert_allclose(loss_expect, loss)

    def test_forward_cpu(self):
        self.check_forward(self.t, self.bigram, tuple(self.x),
                           self.l_length, self.x_length)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.t), cuda.to_gpu(self.bigram),
                           tuple(cuda.to_gpu(x_data) for x_data in self.x),
                           cuda.to_gpu(self.l_length),
                           cuda.to_gpu(self.x_length))

    # expected value(via numerical differentiation) from t_data
    def check_backward(self, unigram_data, bigram_data, xs_data, l_length, x_length, gy_data):
        gradient_check.check_backward(
            gram_ctc.GramCTC(
                self.blank_symbol, self.reduce),
            (x_length, l_length, unigram_data, bigram_data) + xs_data, gy_data,
            eps=1e-2, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.t, self.bigram, tuple(self.x),
                            self.l_length, self.x_length,
                            self.gy)

    @condition.retry(3)
    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.t), cuda.to_gpu(self.bigram),
                            tuple(cuda.to_gpu(x_data) for x_data in self.x),
                            cuda.to_gpu(self.l_length),
                            cuda.to_gpu(self.x_length),
                            cuda.to_gpu(self.gy))


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTC(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCWithoutLength(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.use_length = False


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCWithLabelPadding(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.l_length[0] = 1


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCWithInputPadding(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.x_length[0] = 3


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCWithAllPadding(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.x_length[...] = 3
        self.l_length[...] = 1


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCWithRepeatedLabel(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.t = numpy.array([[0, 1, 1], [0, 1, 0]]).astype(numpy.int32)
        self.bigram = numpy.array([[-1, 2, 2], [-1, -1, -1]]).astype(numpy.int32)
        self.l = gram_ctc._label_to_path(self.t, self.bigram, self.blank_symbol, numpy)
        self.l_length = numpy.full((len(self.t),), len(self.t[0]), dtype='i')


@testing.parameterize(
    {'reduce': 'mean'},
    {'reduce': 'no'}
)
class TestCTCBlankSymbol(unittest.TestCase, GramCTCTestBase):

    def setUp(self):
        GramCTCTestBase.setUp(self)
        self.x = numpy.random.uniform(-1, 1, (4, 2, 4)).astype(numpy.float32)
        self.t = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        self.bigram = numpy.array([[-1, 2], [-1, -1]]).astype(numpy.int32)
        self.blank_symbol = 3
        self.l = gram_ctc._label_to_path(self.t, self.bigram, self.blank_symbol, numpy)


class TestCTCUseNoBackpropMode(unittest.TestCase):

    def test_no_backprop_mode(self):
        xs_data = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(numpy.float32)
        t_data = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        bigram_data = numpy.array([[-1, -1], [-1, -1]]).astype(numpy.int32)
        with chainer.no_backprop_mode():
            x = [chainer.Variable(x_data) for x_data in xs_data]
            t = chainer.Variable(t_data)
            bigram = chainer.Variable(bigram_data)
            gram_ctc.gram_ctc(x, t, bigram, 2)


class TestCTCError(unittest.TestCase):

    def test_not_iterable(self):
        x = chainer.Variable(numpy.zeros((4, 2, 3), numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        with self.assertRaises(TypeError):
            gram_ctc.gram_ctc(x, t, 0)


class TestCTCInvalidReductionOption(unittest.TestCase):

    def test_not_iterable(self):
        x = chainer.Variable(numpy.zeros((4, 2, 3), numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        bigram = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        with self.assertRaises(ValueError):
            gram_ctc.gram_ctc(tuple(x), t, bigram, 0, reduce='invalid_option')

numpy.set_printoptions(linewidth=200, precision=4)
testing.run_module(__name__, __file__)