import multiprocessing
import numpy as np
from scipy import stats
from os.path import exists
import h5py, os

def load_filelist(filepath):
    imglist, y = [], []
    with open(filepath,'r') as f:
        for line in f:
            label, _, _ = line.strip().split('\t')
            y.append(int(label)) 
    return imglist, y


def get_preds(k):
    # TODO: Further optimize the stats.mode calculation by a running mode, this code is very slow for very large k which can hinder analysis
    for i in range(1, len(labels)):
        if i <= k:
            pred[i-1] = labels[0]  
        else:
            pred[i-1] = stats.mode(labels[i-(k):(i)])[0]
    np.save(logdir+'/shift_'+dataset+'_blind_preds_'+str(k)+'.npy', pred)
    return pred

if __name__ == '__main__':
    dataset='CGLM'
    #dataset='CLOC'

    num_processes = 16

    # Please set directory for labels in the CLDatasets folder
    path_to_cldatasets = '/media/bayesiankitten/Hanson/CLDatasets/data/'
    
    # Please set directory for logs
    global logdir, labels 
    logdir = '/media/bayesiankitten/Alexander/ACM/blind_logs/'
    os.makedirs(logdir, exist_ok=True)
    
    # Load labels file
    # Note: When actually picking the k, use pretrain_labels.hdf5 file. This is for analysis plots, shown in the paper (both have same results, but different purpose).
    with h5py.File(f'{path_to_cldatasets}/{dataset}/order_files/train_labels.hdf5', 'r') as f:
        labels = np.array(f['store_list'])

    labels = np.array(labels, dtype=np.uint16)
    pred = np.ones_like(labels)[1:]

    # Get dataset mode
    modelabel = stats.mode(labels)[0]
    pred = pred*modelabel
    np.save(logdir+'/'+dataset+'_mode.npy', pred)

    p = multiprocessing.Pool(num_processes)
    result = []
    for k in [1, 2, 3, 5, 7, 10, 20, 25, 35, 50, 75, 100, 150, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 75000]:
            print(f'Processing mode of past: {k}')
            pred = np.ones_like(labels)[1:]

            if not exists(logdir+'/shift_'+dataset+'_blind_preds_'+str(k)+'.npy'): 
                result.append(p.apply_async(get_preds, [k]))

    for r in result:
        r.wait()

    print('Extracted all blind classifier results!')
