import numpy as np
from tangles.util.entropy import entropy, joint_entropy, colsplit_mutual_information,\
                                pairwise_mutual_information, information_gain, datapointwise_information_gains

def test_entropy():
    assert entropy(np.array([1])) == 0
    assert entropy(np.empty(0)) == 0
    assert entropy(np.array([1, 2])) == np.log(2)
    assert entropy(np.array([1, 1, 2, 2])) == np.log(2)
    assert np.all(entropy(np.array([[1, 1], [1, 1], [1, 2], [1, 2]])) == np.array([0, np.log(2)]))

def test_joint_entropy():
    assert joint_entropy(np.array([[1, 1], [1, 1], [1, 2], [1, 2]])) == np.log(2)
    assert joint_entropy(np.empty(0)) == 0
    assert joint_entropy(np.empty((3, 0))) == 0

def test_colsplit_mutual_information():
    data = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 3]])
    seps = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]])
    print(colsplit_mutual_information(data, seps))

def test_pairwise_mutual_information():
    data = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 3]])
    print(pairwise_mutual_information(data))

def test_information_gain():
    data = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 3]])
    seps = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]])
    print(information_gain(data, seps))

def test_datapointwise_information_gains():
    data = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 3]])
    seps = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1]])
    print(datapointwise_information_gains(data, seps))