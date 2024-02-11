import unittest
import numpy as np
import mlp
from numpy.testing import assert_allclose
import pickle as pk

seed = 10417617

with open("tests.pk", "rb") as f:
    tests = pk.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestLinearMap
# to run all tests: python -m unittest tests


class TestLinearMap(unittest.TestCase):
    def test(self):
        weights, bias, result = tests[0]
        sl = mlp.LinearMap(18, 100, alpha=0)
        sl.loadparams(np.array(weights).T, np.array(bias))

        test1 = np.arange(18).reshape((1, 18))
        assert_allclose(sl.forward(test1).T, result, atol=TOLERANCE)

        test2 = np.arange(100).reshape((1, 100))
        result = tests[1][2]
        assert_allclose(sl.backward(test2).T, result, atol=TOLERANCE)
        sl.zerograd()

        test3 = np.arange(36).reshape((18, 2)).T
        result = tests[2][2]
        assert_allclose(sl.forward(test3).T, result, atol=TOLERANCE)

        test4 = np.arange(200).reshape((100, 2)).T
        assert_allclose(sl.backward(test4).T, tests[3][2], atol=TOLERANCE)


class TestReLU(unittest.TestCase):
    def test(self):
        sl = mlp.ReLU()

        test7 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.forward(test7, train=False), tests[6][0], atol=TOLERANCE)
        assert_allclose(sl.forward(test7), tests[6][1], atol=TOLERANCE)

        test8 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.backward(test8), tests[7], atol=TOLERANCE)


class TestLoss(unittest.TestCase):
    def test(self):
        sl = mlp.SoftmaxCrossEntropyLoss()

        np.random.seed(1)
        logits = np.random.uniform(-1, 1, [18, 2]).T
        labels = np.zeros(logits.shape)
        labels[0, 3], labels[1, 15] = 1, 1

        tests[8] = 3.341601237187909
        tests[9] = np.array(
            [
                [0.02616557, 0.03451958],
                [0.01136603, 0.01496244],
                [0.01523983, 0.00983114],
                [-0.48350725, 0.0163136],
                [0.02512681, 0.02401098],
                [0.02627951, 0.03217908],
                [0.01710387, 0.0473285],
                [0.01200323, 0.03124354],
                [0.02618037, 0.02498454],
                [0.01504693, 0.01214698],
                [0.05636732, 0.05667883],
                [0.02126896, 0.03263949],
                [0.06557396, 0.04891534],
                [0.01347032, 0.00883735],
                [0.01595961, 0.04733087],
                [0.01383351, -0.48102556],
                [0.07718302, 0.02374111],
                [0.04533841, 0.01536218],
            ]
        )
        assert_allclose(sl.forward(logits, labels).T, tests[8], atol=TOLERANCE)
        assert_allclose(sl.backward(), tests[9].T, atol=TOLERANCE)


class TestSingleLayerMLP(unittest.TestCase):
    def test(self):
        data = [np.arange(20).reshape((1, 20)), np.arange(20).reshape((1, 20)) - 1]
        ann = mlp.SingleLayerMLP(20, 19, hiddenlayer=100, alpha=0.1)
        weights, bias, resultWs, resultbs = tests[10]
        weights = [np.array(w.T) for w in weights]
        bias = [np.array(w) for w in bias]
        ann.loadparams(weights, bias)
        np.random.seed(seed)
        ann.forward(data[0])
        ann.backward(np.arange(19).reshape((1, 19)))
        ann.step()
        for aW, rW in zip(ann.getWs(), resultWs):
            assert_allclose(aW.T, rW, atol=TOLERANCE)
        for ab, rb in zip(ann.getbs(), resultbs):
            assert_allclose(ab, rb, atol=TOLERANCE)

        ann.zerograd()
        ann.backward(np.arange(19).reshape((1, 19)) + 1)
        ann.step()
        weights, bias, resultWs, resultbs = tests[11]
        for aW, rW in zip(ann.getWs(), resultWs):
            assert_allclose(aW.T, rW, atol=TOLERANCE)
        for ab, rb in zip(ann.getbs(), resultbs):
            assert_allclose(ab, rb, atol=TOLERANCE)


class TestTwoLayerMLP(unittest.TestCase):
    def test(self):
        data = [np.arange(20).reshape((1, 20)), np.arange(20).reshape((1, 20)) - 1]
        ann = mlp.TwoLayerMLP(20, 19, hiddenlayers=[100, 100], alpha=0.1)
        weights, bias, resultWs, resultbs = tests[12]
        weights = [np.array(w.T) for w in weights]
        bias = [np.array(w) for w in bias]
        ann.loadparams(weights, bias)
        np.random.seed(seed)
        ann.forward(data[0])
        ann.backward(np.arange(19).reshape((1, 19)))
        ann.step()
        for aW, rW in zip(ann.getWs(), resultWs):
            assert_allclose(aW.T, rW, atol=TOLERANCE)
        for ab, rb in zip(ann.getbs(), resultbs):
            assert_allclose(ab, rb, atol=TOLERANCE)

        ann.zerograd()
        ann.backward(np.arange(19).reshape((1, 19)) + 1)
        ann.step()
        weights, bias, resultWs, resultbs = tests[13]
        for aW, rW in zip(ann.getWs(), resultWs):
            assert_allclose(aW.T, rW, atol=TOLERANCE)
        for ab, rb in zip(ann.getbs(), resultbs):
            assert_allclose(ab, rb, atol=TOLERANCE)


class TestDropout(unittest.TestCase):
    def test(self):
        np.random.seed(10707)
        sl = mlp.Dropout(0.5)
        test18 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.forward(test18, train=False), tests[18][0], atol=TOLERANCE)
        assert_allclose(sl.forward(test18), tests[18][1], atol=TOLERANCE)

        test19 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.backward(test19), tests[19], atol=TOLERANCE)


class TestBatchNorm(unittest.TestCase):
    def test(self):
        # np.random.seed(10707)
        sl = mlp.BatchNorm(2)
        test20 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.forward(test20, train=False), tests[20][0], atol=TOLERANCE)
        assert_allclose(sl.forward(test20), tests[20][1], atol=TOLERANCE)

        test21 = (np.arange(36).reshape((18, 2)) - 18).astype("float32")
        assert_allclose(sl.backward(test21), tests[21], atol=TOLERANCE)
