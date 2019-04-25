import os
import path
import nibabel as nib
import string
import numpy as np
import scipy.signal
import scipy.sparse
import matplotlib.pyplot as plt
import pickle
import re
from util.path import get_root
import scipy.io as sio
import requests
import warnings
import configparser


class HCPDownloader:
    def __init__(self, base_path, settings):
        self.base_path = base_path
        self.settings = settings

    def load(self, path):
        if not os.path.isfile(path):
            subject = path.split('/')[3]
            key = path.split('/MNINonLinear/')[1]
            url = self.base_path.format(subject, subject, subject) + key
            r = requests.get(url, auth=(self.settings['CREDENTIALS']['Username'], self.settings['CREDENTIALS']['Password']),
                             stream=True)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(r.content)


class GitDownloader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load(self, path):
        if not os.path.isfile(path):
            subject = path.split('/')[3]
            key = path.split('/MNINonLinear/')[1]
            temp = key.split('/', 1)
            key = temp[0] + '/'  + subject + '/' + temp[1]
            url = self.base_path + key
            r = requests.get(url)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(r.content)


def load_strucutural(subjects, file_url):

    strut = sio.loadmat(file_url).get('strut')
    strut_subs = [strut[0][0][2][i][0][0] for i in range(len(strut[0][0][2]))]

    S = list()

    for subject in subjects:
        idx_subj = strut_subs.index(subject)
        Si = strut[0][0][1][idx_subj][0]
        S.append(scipy.sparse.csr_matrix(Si))

    return S


def load_subjects(list_url):

    with open(list_url, 'r') as f:
        subjects = [s.strip() for s in f.readlines()]

    return subjects


# def process_all(base_dir, parc, subjects, tasks):
#
#     subjects_data = dict()
#
#     for i, subject in enumerate(subjects):
#
#         data = dict()
#
#         print('=== Processing subject {}, {} of {} ...'.format(subject, i+1, len(subjects)))
#         task_list = dict()
#
#         for task in tasks:
#
#             print('\n--- task {} ...'.format(task))
#
#             task_dict = dict()
#
#             ts = get_ts(subject, task, parc, base_dir)
#
#             cues = get_all_cue_times(subject, task, base_dir)
#
#             heart, resp = get_vitals(subject, task, base_dir)
#
#             task_dict['ts'] = ts
#             task_dict['cues'] = cues
#             task_dict['heart'] = heart
#             task_dict['resp'] = resp
#
#             task_list[task] = task_dict
#
#         data['functional'] = task_list
#
#         data['adj'] = get_adj(subject, parc, base_dir)
#
#     subjects_data[subject] = data
#
#     return subjects_data


def process_subject(parc, subject, tasks, loaders):

    hcp_downloader = loaders[0]
    git_downloader = loaders[1]

    data = dict()

    #print('Processing subject {}'.format(subject))
    task_list = dict()

    for task in tasks:
        #print('\n--- task {} ...'.format(task))

        task_dict = dict()

        ts = get_ts(subject, task, parc, hcp_downloader)

        cues = get_all_cue_times(subject, task, hcp_downloader)

        cue_list = [cues['lf'], cues['lh'], cues['rf'], cues['rh'], cues['t']]
        cue_arr = np.zeros((5, 284), dtype=int)
        for i in range(len(cue_list)):
            limb = cue_list[i]
            for j in limb:
                cue_arr[i, j] = 1

        heart, resp = get_vitals(subject, task, hcp_downloader)

        task_dict['ts'] = ts
        task_dict['cues'] = cue_arr
        task_dict['heart'] = heart
        task_dict['resp'] = resp

        task_list[task] = task_dict

    data['functional'] = task_list

    data['adj'] = get_adj(subject, parc, loaders).tocsr()

    return data


def get_ts(subject, task, parc, settings):

    # get time series
    ts = load_ts_for_subject_task(subject, task, settings)

    # get parcellation
    parc_vector, parc_labels = get_parcellation(parc, subject, settings)

    # parcellate time series
    ts_p = parcellate(ts, parc, parc_vector, parc_labels)

    return ts_p


def load_ts_for_subject_task(subject, task, hcp_downloader):

    #print("  loading time series...", end="", flush=True)

    fname = 'tfMRI_' + task + '_Atlas.dtseries.nii'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)

    path = '/tmp/' + furl
    hcp_downloader.load(path)

    ts = np.array(nib.load(path).get_data())

    return ts


def get_cues(subject, task, hcp_downloader):
    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    path = '/tmp/' + fpath
    files = ['cue.txt', 'lf.txt', 'lh.txt', 'rf.txt', 'rh.txt', 't.txt']

    for file in files:
        new_path = os.path.join(path, file)
        hcp_downloader.load(new_path)

    files = os.listdir(path)
    cues = [file.split('.')[0] for file in files if file != 'Sync.txt']

    return cues   


def get_cue_times(cue, subject, task, settings, TR):
    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    fpath = '/tmp/' + fpath
    furl = os.path.join(fpath, cue + '.txt')
    with open(furl) as inp:
        evs = [line.strip().split('\t') for line in inp]
        evs_t = [int(float(evi[0])/TR) for evi in evs]
    return evs_t


def get_all_cue_times(subject, task, hcp_downloader):

    #print("  reading cue signals...", end="", flush=True)

    TR = 0.72
    cues = {cue: get_cue_times(cue, subject, task, hcp_downloader, TR) for cue in get_cues(subject, task, hcp_downloader)}

    #print("done.")

    return cues


def get_parcellation(parc, subject, hcp_downloader):

    #print("  reading parcellation...", end="", flush=True)

    fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k')

    suffixes = {'aparc': '.aparc.a2009s.32k_fs_LR.dlabel.nii',
                'dense': '.aparc.a2009s.32k_fs_LR.dlabel.nii'}

    parc_furl = os.path.join('/tmp/', fpath, subject + suffixes[parc])

    hcp_downloader.load(parc_furl)
    parc_obj = nib.load(parc_furl)

    if parc == 'aparc':
        parc_vector = np.array(parc_obj.get_data()[0], dtype='int')
        table = parc_obj.header.matrix[0].named_maps.__next__().label_table
        parc_labels = [(region[1].key, region[1].label) for region in table.items()]

    if parc == 'dense':
        n_regions = len(parc_obj.get_data()[0])
        parc_vector = np.array(range(n_regions))
        parc_labels = [(i, i) for i in range(n_regions)]

    #print("done.")

    return parc_vector, parc_labels


def parcellate(ts, parc, parc_vector, parc_labels):

    #print("  performing parcellation...", end="", flush=True)

    tst = ts[:, :parc_vector.shape[0]]

    parc_idx = list(np.unique(parc_vector))

    bad_regions = [label[0] for label in parc_labels if label[1] == '???']
    for bad_region in bad_regions:
        parc_idx.remove(bad_region)

    if parc == 'aparc':
        # build parcellated signal by taking the mean across voxels
        x_parc = np.array([np.mean(tst[:, (parc_vector == i).tolist()], axis=1) for i in parc_idx])

    if parc == 'dense':
        x_parc = tst.T

    #print("done.")

    return x_parc


def get_vitals(subject, task, hcp_downloader):

    #print("  reading vitals...", end="", flush=True)

    TR = 0.72

    fname = 'tfMRI_' + task + '_Physio_log.txt'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)
    path = '/tmp/' + furl

    hcp_downloader.load(path)

    with open(path) as inp:
        phy = [line.strip().split('\t') for line in inp]
        resp = [int(phy_line[1]) for phy_line in phy]
        heart = [int(phy_line[2]) for phy_line in phy]
    
    # base TR period fMRI sampling rate
    TR = 0.72
    fl = 1/TR
    # physio sampling rate
    fh = 400
    # decimation order
    n = 2
    q = int(fh/fl)
    heart_d = scipy.signal.decimate(np.array(heart[:-1]), q, n, zero_phase=True)
    resp_d = scipy.signal.decimate(np.array(resp[:-1]), q, n, zero_phase=True)

    #print("done.")
    
    return heart_d, resp_d


def read_surf_to_gray_map(hemi):

    fname = os.path.join(os.path.dirname(__file__), 'res', hemi + '_dense_map.txt')

    surf_to_gray = np.loadtxt(fname, delimiter=',', dtype=int)

    return surf_to_gray


def map_to_surf(idx, surf_to_gray):

    surf_idx = np.nonzero(surf_to_gray[:, 1] == idx)[0]

    if surf_idx.shape[0] == 0:
        to_idx = -1
    else:
        to_idx = surf_to_gray[int(surf_idx), 0]

    return to_idx


def get_row_cols(faces, hemi):

    rows = list()
    cols = list()

    surf_to_gray = read_surf_to_gray_map(hemi)

    for i in faces:

        p1 = map_to_surf(i[0], surf_to_gray)
        p2 = map_to_surf(i[1], surf_to_gray)
        p3 = map_to_surf(i[2], surf_to_gray)

        if p1 > 0 and p2 > 0:
            rows.append(p1)
            cols.append(p2)
            rows.append(p2)
            cols.append(p1)

        if p1 > 0 and p3 > 0:
            rows.append(p1)
            cols.append(p3)
            rows.append(p3)
            cols.append(p1)

        if p2 > 0 and p3 > 0:
            rows.append(p2)
            cols.append(p3)
            rows.append(p3)
            cols.append(p2)

    return rows, cols


def filter_surf_vertices(coords):

    new_coords = []

    hemi = 'L'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    hemi = 'R'
    surf_to_gray = read_surf_to_gray_map(hemi)
    for i in range(surf_to_gray.shape[0]):
        idx_old = surf_to_gray[i, 1]
        idx_new = surf_to_gray[i, 0]
        new_coords.insert(idx_new, coords[idx_old, :])

    new_coords = np.array(new_coords)

    return new_coords


def get_adj_hemi(hemi, inflation, subject, hcp_downloader, offset):

    fname = subject + '.' + hemi + '.' + inflation + '.32k_fs_LR.surf.gii'
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k', fname)
    file = '/tmp/' + furl

    hcp_downloader.load(file)
    img = nib.load(file)

    coords = img.darrays[0].data
    faces = img.darrays[1].data.astype(int) + offset
    rows, cols = get_row_cols(faces, hemi)
    new_coords = filter_surf_vertices(coords)

    # fname = subject + '.' + hemi + '.aparc.a2009s.32k_fs_LR.label.gii'
    # furl = os.path.join(base_dir, subject, 'MNINonLinear', 'fsaverage_LR32k', fname)
    # img_labels = nib.load(furl)
    # labels = img_labels.darrays[0].data
    # parc_labels = img_labels.get_labeltable().get_labels_as_dict()

    return rows, cols, new_coords


def get_adj(subject, parc, loaders):

    if parc == 'aparc':
        adj = get_adj_dti(subject, parc, loaders[1])

    elif parc == 'dense':
        adj, coords = get_adj_mesh(subject, loaders[0])

    return adj


def get_adj_dti(subject, parc, git_downloader):

    #print("\n--- dti adjacency matrix...", end="", flush=True)
    furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'dMRI_CONN')
    file = '/tmp/' + furl + '/' + subject + '.aparc.a2009s.dti.conn.mat'

    git_downloader.load(file)

    try:
        S = sio.loadmat(file)
        S = S.get('S')
    except:
        file_dir = os.path.join(get_root(), 'load/res/average1.aparc.a2009s.dti.conn.mat')
        S = sio.loadmat(file_dir).get('S')
        warnings.warn("Local DTI adjacency matrix for subject: <subject> in parcellation:<parcellation>  not available, using average adjacency matrix.")

    S_coo = scipy.sparse.coo_matrix(S)

    return S_coo


def get_adj_mesh(subject, settings):

    #print("\n--- mesh adjacency matrix...")

    inflation = 'inflated' #'white'

    #print("  processing left hemisphere edges...", end="", flush=True)
    hemi = 'L'
    rows_L, cols_L, coords_L = get_adj_hemi(hemi, inflation, subject, settings, offset=0)
    #print("done.")

    #print("  processing right hemisphere edges...", end="", flush=True)
    hemi = 'R'
    rows_R, cols_R, coords_R = get_adj_hemi(hemi, inflation, subject, settings, offset=0)
    #print("done.")

    #print("  processing coordinates...", end="", flush=True)
    data = np.ones(len(rows_L) + len(rows_R))
    A = scipy.sparse.coo_matrix((data, (rows_L+rows_R, cols_L+cols_R)))
    coords = np.vstack((coords_L, coords_R))
    new_coords = filter_surf_vertices(coords)
    #print("done.")

    return A, new_coords


# def load_parcs():
#
#     base_dir = '/home/semo/data_dense'
#     tasks = ['MOTOR_LR']
#     subjects = ['100307']
#
#     ###################
#
#     parc = 'aparc'
#
#     print('\n########### Parcellation: {:} #############\n'.format(parc))
#
#     data_aparc = process_subject(base_dir, parc, subjects, tasks)
#
#     print('\n=== Verifying data for subject: {:}\n'.format(subjects[0]))
#
#     ts_shape = data_aparc[subjects[0]]['functional'][tasks[0]]['ts'].shape
#     print('  regions: {:}, time samples: {:}'.format(ts_shape[0], ts_shape[1]))
#
#     heart_shape = data_aparc[subjects[0]]['functional'][tasks[0]]['heart'].shape
#     print('  heart time samples: {:}'.format(heart_shape[0]))
#
#     resp_shape = data_aparc[subjects[0]]['functional'][tasks[0]]['resp'].shape
#     print('  resp time samples: {:}'.format(resp_shape[0]))
#
#     adj_shape = data_aparc[subjects[0]]['adj'].shape
#     print('  adjacency regions: {:} x {:}'.format(adj_shape[0], adj_shape[1]))
#
#
#     ###################
#
#     parc = 'dense'
#
#     print('\n########### Parcellation: {:} #############\n'.format(parc))
#
#     data_dense = process_all(base_dir, parc, subjects, tasks)
#
#     print('\n=== Verifying data for subject: {:}\n'.format(subjects[0]))
#
#     ts_shape = data_dense[subjects[0]]['functional'][tasks[0]]['ts'].shape
#     print('  regions: {:}, time samples: {:}'.format(ts_shape[0], ts_shape[1]))
#
#     heart_shape = data_dense[subjects[0]]['functional'][tasks[0]]['heart'].shape
#     print('  heart time samples: {:}'.format(heart_shape[0]))
#
#     resp_shape = data_dense[subjects[0]]['functional'][tasks[0]]['resp'].shape
#     print('  resp time samples: {:}'.format(resp_shape[0]))
#
#     adj_shape = data_dense[subjects[0]]['adj'].shape
#     print('  adjacency regions: {:} x {:}'.format(adj_shape[0], adj_shape[1]))
#
#
#
#
# def save():
#
#     # parcellation
#     parc = 'aparc'
#
#     # file for persisting
#     db_file = 'saved1_hcp.pkl'
#     db_url = os.path.join(os.path.expanduser('~'), 'Downloads', db_file)
#
#     # base subjects folder
#     base_dir = '/Users/cassiano/Desktop/datasets/1subject/'
#
#     # parcellation
#     parc = 'aparc'
#     # tasks
#     tasks = ['MOTOR_LR', 'MOTOR_RL']
#
#     # gets a list of subjects from the directory
#     files = os.listdir(base_dir)
#     # filters only valid folder names as subjects
#     r = re.compile("[0-9]{6}")
#     subjects = list(filter(r.match, files))
#
#     # shortens the list
#     n_subjects = 1
#     subjects[0:n_subjects]
#     # subjects = ['100307']
#
#     # processes all subjects and tasks
#     data = process_all(base_dir, parc, subjects, tasks, db_url)
#
#     # persists file
#     pickle.dump(data, open(db_url, 'wb'))
#
#     # loads file to verify
#     del data
#     data = pickle.load(open(db_url, 'rb'))
#     data['100307']['MOTOR_RL']['cues']['rf']
#
#
# if __name__ == '__main__':
#     load_parcs()
#     # save()
