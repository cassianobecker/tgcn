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


def process_all(base_dir, parc, subjects, tasks, db_url):

    data = dict()

    for i, subject in enumerate(subjects):

        print('Processing subject {}, {} of {} ... '.format(subject, i+1, len(subjects)))
        task_list = dict()

        for task in tasks:

            print('--- task {} ... '.format(task))

            task_dict = dict()

            ts, cues = get_ts_cues(subject, task, parc, base_dir)

            heart, resp = get_vitals(subject, task, base_dir)

            task_dict['ts'] = ts
            task_dict['cues'] = cues
            task_dict['heart'] = heart
            task_dict['resp'] = resp

            task_list[task] = task_dict

        data[subject] = task_list

        data['adj'] = get_adj(subject, base_dir)

    return data


def get_ts_cues(subject, task, parc, base_dir):
    # get time series
    ts = get_ts_for_subject_task(subject, task, base_dir)
    # get event times
    cues = get_all_cue_times(subject, task, base_dir)
    # get parcellation
    parc_vector = get_parcellation(parc, subject, base_dir)
    # parcellate time series
    ts_p = parcellate(ts, parc_vector)

    return ts_p, cues


def get_ts_for_subject_task(subject, task, base_dir):
    fname = 'tfMRI_' + task + '_Atlas.dtseries.nii'
    furl = os.path.join(base_dir, subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)
    ts = np.array(nib.load(furl).get_data())
    return ts


def get_cues(subject, task, base_dir):
    fpath = os.path.join(base_dir, subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    files = os.listdir(fpath)
    cues = [file.split('.')[0] for file in files if file != 'Sync.txt'] 
    return cues   


def get_cue_times(cue, subject, task, base_dir, TR):
    fpath = os.path.join(base_dir, subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs')
    furl = os.path.join(fpath, cue + '.txt')
    with open(furl) as inp:
        evs = [line.strip().split('\t') for line in inp]
        evs_t = [int(float(evi[0])/TR) for evi in evs]
    return evs_t


def get_all_cue_times(subject, task, base_dir):
    TR = 0.72
    cues = {cue: get_cue_times(cue, subject, task, base_dir, TR) for cue in get_cues(subject, task, base_dir)}
    return cues    


def get_parcellation(parc, subject, base_dir):
    
    fpath = os.path.join(base_dir, subject, 'MNINonLinear', 'fsaverage_LR32k')
    suffixes = {'aparc': '.aparc.a2009s.32k_fs_LR.dlabel.nii'}

    parc_furl = os.path.join(fpath, subject + suffixes[parc])
    parc_obj = nib.load(parc_furl)
    parc_vector = np.array(parc_obj.get_data(), dtype='int')
    # crop last indices (sub-cortical structures)
    
    return parc_vector


def parcellate(ts, parc_vector):

    tst = ts[:, :parc_vector.shape[1]]

    # number of parcels
    num_parc = np.unique(parc_vector).shape[0]

    # build parcellated signal by taking the mean across voxels
    x_parc = np.array([np.mean(tst[:, (parc_vector == i).tolist()[0]], axis=1) for i in range(num_parc)])

    return x_parc


def get_vitals(subject, task, base_dir):

    TR = 0.72

    fname = 'tfMRI_' + task + '_Physio_log.txt'
    furl = os.path.join(base_dir, subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)    

    with open(furl) as inp:
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
    heart_d = scipy.signal.decimate(np.array(heart), q, n)
    resp_d = scipy.signal.decimate(np.array(resp), q, n)
    
    return heart_d, resp_d


def get_row_cols(faces):
    rows = list()
    cols = list()
    for i in faces:
        p1 = i[0]
        p2 = i[1]
        p3 = i[2]
        rows.append(p1)
        cols.append(p2)
        rows.append(p2)
        cols.append(p1)
        rows.append(p1)
        cols.append(p3)
        rows.append(p3)
        cols.append(p1)
        rows.append(p2)
        cols.append(p3)
        rows.append(p3)
        cols.append(p2)
    return rows, cols


def get_adj_hemi(hemi, inflation, subject, base_dir, offset):

    fname = subject + '.' + hemi + '.' + inflation + '.32k_fs_LR.surf.gii'
    furl = os.path.join(base_dir, subject, 'MNINonLinear', 'fsaverage_LR32k', fname)

    img = nib.load(furl)
    coords = img.darrays[0].data
    faces = img.darrays[1].data.astype(int) + offset
    rows, cols = get_row_cols(faces)

    return rows, cols, coords


def get_adj(subject, base_dir):

    inflation = 'white'

    hemi = 'L'
    rows_L, cols_L, coords_L = get_adj_hemi(hemi, inflation, subject, base_dir, offset=0)

    hemi = 'R'
    offset = len(coords_L)
    rows_R, cols_R, coords_R = get_adj_hemi(hemi, inflation, subject, base_dir, offset=offset)

    data = np.ones(len(rows_L) + len(rows_R))
    A = scipy.sparse.coo_matrix((data, (rows_L+rows_R, cols_L+cols_R)))
    coords = np.vstack((coords_L, coords_R))

    return A, coords


def save():

    # file for persisting
    db_file = 'saved1_hcp.pkl'
    db_url = os.path.join(os.path.expanduser('~'), 'Downloads', db_file)

    # base subjects folder
    base_dir = '/Users/cassiano/Desktop/datasets/1subject/'

    # parcellation
    parc = 'aparc'
    # tasks
    tasks = ['MOTOR_LR', 'MOTOR_RL']

    # gets a list of subjects from the directory
    files = os.listdir(base_dir)
    # filters only valid folder names as subjects
    r = re.compile("[0-9]{6}")
    subjects = list(filter(r.match, files))

    # shortens the list
    n_subjects = 1
    subjects[0:n_subjects]
    # subjects = ['100307']

    # processes all subjects and tasks
    data = process_all(base_dir, parc, subjects, tasks, db_url)

    # persists file
    pickle.dump(data, open(db_url, 'wb'))

    # loads file to verify
    del data
    data = pickle.load(open(db_url, 'rb'))
    data['100307']['MOTOR_RL']['cues']['rf']


if __name__ == '__main__':

    save()
