import numpy as np
import scipy.io as sio
import os
from util.path import get_root
import scipy.sparse
from os.path import expanduser

def get_cues(MOTOR):
    C = MOTOR['ev_idx'][0, 0]
    return C[1:, :]


def get_bold(MOTOR):
    ts = MOTOR['ts'][0, 0]
    X = np.matrix(ts)
    X = X.transpose()
    return X


def get_vitals(MOTOR):
    resp = MOTOR['resp'][0, 0][0]
    heart = MOTOR['heart'][0, 0][0]
    V = np.matrix([resp, heart])
    V = V.transpose()
    return V


def clean_bold(X, v):
    A_1 = np.linalg.inv(v.transpose() * v)
    A_2 = A_1 * v.transpose()
    A_hat = A_2 * X
    X_hat = v * A_hat
    X_bar = X - X_hat
    return X_bar


def get_dataset_single(file, session, p=148, T=284):
    filenames = file

    Np = 1
    m = 5

    mis_matched = 0

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])
    X_bar = np.zeros([Np, p, T])

    ds = sio.loadmat(file).get('ds')
    MOTOR = ds[0, 0][session]

    C_i = get_cues(MOTOR)
    X_i = get_bold(MOTOR)
    X_bar_i = X_i
    X_bar = X_bar_i.transpose()

    return [C_i, X_i, X_bar]


def get_delabeled_dataset(filedir, session, p=148, T=284):
    with open(filedir + 'filenames.txt', 'r') as f:
        filenames = [s.strip() for s in f.readlines()]

    Np = len(filenames)
    m = 5

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = filedir + s
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][session]
        X_i = get_bold(MOTOR)
        X[i, :, :] = X_i.transpose()

    return [C, None, X]


def load_subjects(list_url):

    with open(list_url, 'r') as f:
        subjects = [s.strip() for s in f.readlines()]

    return subjects



def get_dataset(subjects, data_path, post_fix, session, p=32492, T=284):

    # with open(list_url, 'r') as f:
    #     filenames = [s.strip() + post_fix for s in f.readlines()]

    filenames = [s + post_fix for s in subjects]

    Np = len(filenames)
    m = 5

    mis_matched = 0

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])
    X_bar = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = os.path.join(data_path, s)
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][session]

        C_i = get_cues(MOTOR)
        X_i = get_bold(MOTOR)

        C[i, :, :] = C_i
        X[i, :, :] = X_i.transpose()

    return [C, X, X_bar]


def encode(C, X, H, Gp, Gn):
    """
    encodes
    :param C: data labels
    :param X: data to be windowed
    :param H: window size
    :param Gp: start point guard
    :param Gn: end point guard
    :return:
    """
    _, m, _ = C.shape
    Np, p, T = X.shape
    N = T - H + 1
    num_examples = Np * N

    y = np.zeros([Np, N])
    C_temp = np.zeros(T)

    for i in range(Np):
        for j in range(m):
            temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
            cue_idx1 = [idx - Gn for idx in temp_idx]
            cue_idx2 = [idx + Gp for idx in temp_idx]
            cue_idx = list(zip(cue_idx1, cue_idx2))

            for idx in cue_idx:
                C_temp[slice(*idx)] = j + 1

        y[i, :] = C_temp[0: N]

    X_windowed = np.zeros([Np, N, p, H])

    for t in range(N):
        X_windowed[:, t, :, :] = X[:, :, t: t + H]

    y = np.reshape(y, (num_examples))
    X_windowed = np.reshape(X_windowed, (num_examples, p, H))

    return [X_windowed, y]


def load_strucutural(subjects, file_url):

    strut = sio.loadmat(file_url).get('strut')
    strut_subs = [strut[0][0][2][i][0][0] for i in range(len(strut[0][0][2]))]

    S = list()

    for subject in subjects:
        idx_subj = strut_subs.index(subject)
        Si = strut[0][0][1][idx_subj][0]
        S.append(scipy.sparse.csr_matrix(Si))

    return S


def load_hcp_example():

    list_file = 'subjects_inter.txt'
    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects_strut = load_subjects(list_url)

    list_file = 'subjects_all.txt'
    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects = load_subjects(list_url)

    structural_file = 'struct_dti.mat'
    structural_url = os.path.join(get_root(), 'load', 'hcpdata', structural_file)
    S = load_strucutural(subjects_strut, structural_url)

    # data_path = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/tefemerid/code/v1/tfsid/out/data/hcp/many_motor'
    # data_path = '~/data_hcp/'
    data_path = os.path.join(expanduser("~"), 'data_hcp')
    post_fix = '_aparc_tasks.mat'
    p = 148
    T = 284
    C, X, _ = get_dataset(subjects, data_path, post_fix, session='MOTOR_LR', p=p, T=T)

    H, Gp, Gn = 15, 4, 4
    Xw, y = encode(C, X, H, Gp, Gn)

    N0 = np.nonzero(y == 0)[0].shape[0]
    NN = int(np.nonzero(y > 0)[0].shape[0] / (np.unique(y).shape[0] - 1))
    print('Ratio of class imbalance: {}'.format(N0/NN))
    ididx = np.random.permutation(np.nonzero(y == 0)[0].shape[0])[0:N0 - NN]
    idx = np.nonzero(y == 0)[0][ididx]

    # y = np.delete(y, idx, axis=0)
    # Xw = np.delete(Xw, idx, axis=0)

    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)

    k = np.max(np.unique(y))

    yoh = one_hot(y, k+1)

    return Xw, yoh, S


if __name__ == '__main__':
    load_hcp_example()

