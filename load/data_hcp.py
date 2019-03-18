import numpy as np
import scipy.io as sio
import os
from util.path import get_root
import scipy.sparse
from os.path import expanduser
from sklearn.metrics import confusion_matrix, classification_report


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


def load_hcp_example(full=False):

    list_file = 'subjects_inter.txt'
    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects_strut = load_subjects(list_url)


    list_file = 'subjects_all.txt'

    if full:
        list_file = 'subjects_hcp_all.txt'

    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects = load_subjects(list_url)

    structural_file = 'struct_dti.mat'
    structural_url = os.path.join(get_root(), 'load', 'hcpdata', structural_file)
    S = load_strucutural(subjects_strut, structural_url)

    # data_path = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/tefemerid/code/v1/tfsid/out/data/hcp/many_motor'
    # data_path = '~/data_hcp/'

    data_path = os.path.join(expanduser("~"), 'data_hcp')
    post_fix = '_aparc_tasks.mat'

    if full:
        data_path = os.path.join(expanduser("~"), 'data_full/aparc')
        post_fix = '_aparc_tasks_aparc.mat'

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


def get_lookback_data(X, y, lookback=5):
    X_lb = np.zeros(shape=(X.shape[0] - lookback, lookback + 1, X.shape[1]))
    y_lb = np.zeros(shape=(X_lb.shape[0], y.shape[1]))

    for t in range(lookback + 2, X.shape[0]):
        X_lb[t - lookback - 2, :, :] = X[t - lookback - 2 : t - 1, :]
        y_lb[t - lookback - 2, :] = y[t - 1, :]
    return X_lb, y_lb


def decode(y_hat, length=6, offset=2):
    T = len(y_hat)
    y_decoded = [0] * T
    i = 0

    while (i < T - 5):
        a = (int(y_hat[i] == y_hat[i + 1] != 0))
        b = (int(y_hat[i] == y_hat[i + 2] != 0))
        c = (int(y_hat[i] == y_hat[i + 3] != 0))
        d = (int(y_hat[i] == y_hat[i + 4] != 0))
        e = (int(y_hat[i] == y_hat[i + 5] != 0))
        num_agree = (a + b + c + d + e)

        if (num_agree > 1):
            y_decoded[i - offset] = y_hat[i]
            i += length
        else:
            i += 1
    return np.array(y_decoded)


def assess_performance(c_actual, c_predicted, delta=3, include_rest=True):
    predictions = []
    N = c_actual.shape[0]

    cue_locations = np.where(c_actual != 0)[0]

    for cue_loc in cue_locations:
        chunk_actual = c_actual[cue_loc - delta: cue_loc + delta]
        chunk_predicted = c_predicted[cue_loc - delta: cue_loc + delta]

        locations_nz = np.where(chunk_predicted != 0)[0]
        for location_nz in locations_nz:
            actual, predicted = c_actual[cue_loc], chunk_predicted[location_nz]
            predictions.append((actual, predicted, cue_loc, cue_loc + location_nz - delta))

        if not isinstance(locations_nz, np.ndarray):
            predictions.append((c_actual[cue_loc], 0, 0, 0))

    rest_locations = np.pad(cue_locations, (1, 1), 'constant', constant_values=[-delta, N])
    for i in range(rest_locations.shape[0] - 1):
        begin = rest_locations[i] + delta
        end = rest_locations[i + 1] - delta

        chunk_predicted = c_predicted[begin: end]
        loc_nz = np.where(chunk_predicted != 0)[0]

        for i in loc_nz:
            predictions.append((0, chunk_predicted[i], 0, i + begin))

    if include_rest:
        num_left_over = N - len(predictions)
        for _ in range(num_left_over):
            predictions.append((0, 0, 0, 0))

    return np.asarray(predictions, dtype=int)


def gen_results(y_true_decoded, y_pred_decoded):
    # generate results
    y_actual_2 = []
    y_predicted_2 = []

    results = assess_performance(y_true_decoded, y_pred_decoded)
    for tup in results:
        y_actual_2.append(tup[0])
        y_predicted_2.append(tup[1])

    print(classification_report(y_true=y_actual_2, y_pred=y_predicted_2))
    print(confusion_matrix(y_true=y_actual_2, y_pred=y_predicted_2))


def extend_signal(X, length=6, offset=2):
    X_temp = np.zeros_like(X)
    for i in range(length):
        X_temp = X_temp + np.pad(X[:-i-1-offset, :], pad_width=((i+1+offset, 0), (0, 0)), mode='constant')
    return X_temp


def load_hcp_vote(lookback=10):

    list_file = 'subjects_inter.txt'
    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects_strut = load_subjects(list_url)

    list_file = 'subjects_hcp_all.txt'
    #list_file = 'subjects_all.txt'
    list_url = os.path.join(get_root(), 'conf', list_file)
    subjects = load_subjects(list_url)

    structural_file = 'struct_dti.mat'
    structural_url = os.path.join(get_root(), 'load', 'hcpdata', structural_file)
    S = load_strucutural(subjects_strut, structural_url)

    # data_path = '/Users/cassiano/Dropbox/cob/work/upenn/research/projects/tefemerid/code/v1/tfsid/out/data/hcp/many_motor'
    # data_path = '~/data_hcp/'

    data_path = os.path.join(expanduser("~"), 'data_full/aparc')
    post_fix = '_aparc_tasks_aparc.mat'

    #data_path = os.path.join(expanduser("~"), 'data_hcp')
    #post_fix = '_aparc_tasks.mat'

    p = 148
    T = 284
    C, X, _ = get_dataset(subjects, data_path, post_fix, session='MOTOR_LR', p=p, T=T)
    sh = C.shape

    C, X = np.swapaxes(C, 1, 2), np.swapaxes(X, 1, 2)
    C = C.reshape((C.shape[0] * C.shape[1], C.shape[2]))
    X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
    assert (C.shape[0] == X.shape[0])

    C = extend_signal(C)
    # NONE is 1 - any(motor_task)
    C[:, 0] = 1 - np.sum(C[:, 1:6], axis=1)

    N_TRAIN = int(0.75 * X.shape[0])

    X_train_1 = X[0:N_TRAIN, :]
    labels_train_1 = C[0:N_TRAIN, :]

    X_test_1 = X[N_TRAIN:, :]
    labels_test_1 = C[N_TRAIN:, :]

    X_train, y_train = get_lookback_data(X_train_1, labels_train_1, lookback=lookback)
    X_test, y_test = get_lookback_data(X_test_1, labels_test_1, lookback=lookback)
    X_train, X_test = np.swapaxes(X_train, 1, 2), np.swapaxes(X_test, 1, 2)

    return X_train, y_train, X_test, y_test, S


if __name__ == '__main__':
    load_hcp_example()

