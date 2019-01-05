import numpy as np
import scipy.io as sio


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


def get_dataset(filedir, session, p=32492, T=284):
    with open(filedir + 'filenames.txt', 'r') as f:
        filenames = [s.strip() for s in f.readlines()]

    Np = len(filenames)
    m = 5

    mis_matched = 0

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])
    X_bar = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = filedir + s
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][session]

        C_i = get_cues(MOTOR)
        X_i = get_bold(MOTOR)

        X_i = X_i[:, :32492]

        X_bar_i = X_i

        if X_i.shape[1] == 32492:
            C[i, :, :] = C_i
            X[i, :, :] = X_i.transpose()
            X_bar[i, :, :] = X_bar_i.transpose()
        else:
            mis_matched += 1

    if mis_matched > 0:
        print('num mismatched: {}'.format(mis_matched))

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

if __name__ == '__main__':
    H, Gp, Gn = 15, 4, 4

    import os

    cwd = os.getcwd()

    print(cwd)

    res_path = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/tefemerid/code/v1/tfsid/out/data/hcp/many_motor'

    fpath = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/defri/code/tgcn/mesh/fmri/all_subjects/'
    C, _, X_bar = get_dataset(fpath, session='MOTOR_LR')
    X, y = encode(C, X_bar, H, Gp, Gn)