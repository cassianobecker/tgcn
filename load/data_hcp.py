import numpy as np
import scipy.io as sio
import os
from util.path import get_root
import scipy.sparse
from os.path import expanduser
from sklearn.metrics import confusion_matrix, classification_report
import torch
import gcn.coarsening as coarsening
from load.create_hcp import process_subject, load_strucutural, load_subjects
import boto3


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

    return [X_windowed.astype("float32"), y]


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
        data_path = os.path.join(expanduser("~"), 'data_full')
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


def encode_perm(C, X, H, Gp, Gn, indices):
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

    X = X.astype('float32')
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

    X_windowed = [] #np.zeros([Np, N, p, H])

    if indices is None:
        for t in range(N):
            X_windowed.append(X[0, :, t: t + H])    #0 because there is always a single example in each batch

        y = np.reshape(y, (num_examples))
    else:
        M, Q = X[0].shape

        Mnew = len(indices)
        assert Mnew >= M

        if Mnew > M:
            diff = Mnew - M
            z = np.zeros((X.shape[0], diff, X.shape[2]), dtype="float32")
            X = np.concatenate((X, z), axis=1)

        for t in range(N):
            X_windowed.append(X[0, indices, t: t + H])

        y = np.reshape(y, (num_examples))

    #F = 1024 ** 2
    #print('Bytes of X: {:1.4f} MB.'.format(getsizeof(X_windowed) / F))

    return [X_windowed, y]



def perm_data_time(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M, Q = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew, Q), dtype="float32")
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[:, i, :] = x[:, j, :]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[:, i, :] = np.zeros((N, Q))
    return xnew


class Encode(object):

    def __init__(self, H, Gp, Gn, perm):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, C, X, perm):
        Xw, y = encode(C, X, self.H, self.Gp, self.Gn)
        Xw = perm_data_time(Xw, perm)

        one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
        k = np.max(np.unique(y))

        yoh = one_hot(y, k + 1)
        return Xw, yoh


class EncodePerm(object):

    def __init__(self, H, Gp, Gn):
        self.H = H
        self.Gp = Gp
        self.Gn = Gn

    def __call__(self, C, X, perm):
        Xw, y = encode_perm(C, X, self.H, self.Gp, self.Gn, perm)

        one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
        k = np.max(np.unique(y))

        yoh = one_hot(y, k + 1)
        return Xw, yoh


class StreamDataset(torch.utils.data.Dataset):

    def __init__(self):
        normalized_laplacian = True
        coarsening_levels = 4

        list_file = 'subjects_inter.txt'
        list_url = os.path.join(get_root(), 'conf', list_file)
        subjects_strut = load_subjects(list_url)

        structural_file = 'struct_dti.mat'
        structural_url = os.path.join(get_root(), 'load', 'hcpdata', structural_file)
        S = load_strucutural(subjects_strut, structural_url)
        S = S[0]

        #avg_degree = 7
        #S = scipy.sparse.random(65000, 65000, density=avg_degree/65000, format="csr")

        self.graphs, self.perm = coarsening.coarsen(S, levels=coarsening_levels, self_connections=False)

        self.list_file = 'subjects_hcp_all.txt'
        list_url = os.path.join(get_root(), 'conf', self.list_file)
        self.data_path = os.path.join(expanduser("~"), 'data_full')

        self.subjects = load_subjects(list_url)
        post_fix = '_aparc_tasks_aparc.mat'
        self.filenames = [s + post_fix for s in self.subjects]

        self.p = 148#65000
        self.T = 284
        self.session = 'MOTOR_LR'

        self.transform = EncodePerm(15, 4, 4, self.perm)

    def get_graphs(self, device):
        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(device) for graph in self.graphs]
        return self.graphs, coos, self.perm

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = os.path.join(self.data_path, self.filenames[idx])
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][self.session]

        C_i = np.expand_dims(get_cues(MOTOR), 0)
        X_i = np.expand_dims(get_bold(MOTOR).transpose(), 0)

        #X_i = np.random.rand(1, 65000, 284)

        Xw, yoh = self.transform(C_i, X_i)

        return Xw, yoh


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, perm):

        self.list_file = 'subjects_test.txt'
        list_url = os.path.join(get_root(), 'conf', self.list_file)
        self.data_path = os.path.join(expanduser("~"), 'data_full')

        self.subjects = load_subjects(list_url)
        post_fix = '_aparc_tasks_aparc.mat'
        self.filenames = [s + post_fix for s in self.subjects]

        self.p = 148
        self.T = 284
        self.session = 'MOTOR_LR'

        self.transform = Encode(15, 4, 4, perm)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = os.path.join(self.data_path, self.filenames[idx])
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][self.session]

        C_i = np.expand_dims(get_cues(MOTOR), 0)
        X_i = np.expand_dims(get_bold(MOTOR).transpose(), 0)

        Xw, yoh = self.transform(C_i, X_i)

        return Xw.astype('float32'), yoh


class FullDataset(torch.utils.data.Dataset):

    def __init__(self, device, data_type='dense', test=False):

        normalized_laplacian = True
        self.coarsening_levels = 4
        self.data_type = data_type

        s3 = boto3.resource('s3')

        bucket = s3.Bucket('hcp-openaccess')
        prefix = 'HCP_1200'

        for obj in bucket.objects.filter(Prefix=prefix):
            print('{0}:{1}'.format(bucket.name, obj.key))

        #############

        self.list_file = 'subjects.txt'
        if test:
            list_url = os.path.join(get_root(), 'conf/hcp/train/motor_lr', self.list_file)
        else:
            list_url = os.path.join(get_root(), 'conf/hcp/train/motor_lr', self.list_file)

        self.data_path = os.path.join(expanduser("~"), 'data_dense')
        #self.data_path = os.path.join(get_root(), 'load/hcpdense')

        self.subjects = load_subjects(list_url)

        self.p = 148
        self.T = 284
        self.session = 'MOTOR_LR'

        self.transform = EncodePerm(15, 4, 4)
        self.device = device

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        #file = os.path.join(self.data_path, self.subjects[idx])

        subject = self.subjects[idx]

        data = process_subject(self.data_path, self.data_type, subject, [self.session], None)

        cues = data['functional']['MOTOR_LR']['cues']
        ts = data['functional']['MOTOR_LR']['ts']
        S = data['adj']

        graphs, perm = coarsening.coarsen(S, levels=self.coarsening_levels, self_connections=False)
        coos = [torch.tensor([graph.tocoo().row, graph.tocoo().col], dtype=torch.long).to(self.device) for graph in graphs]

        C_i = np.expand_dims(cues, 0)
        X_i = np.expand_dims(ts, 0)

        Xw, yoh = self.transform(C_i, X_i, perm)

        return Xw, yoh, coos, perm


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

