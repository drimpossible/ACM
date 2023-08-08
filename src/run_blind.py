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
    pred = np.ones_like(labels)[1:]

    for i in range(1, len(labels)):
        if i <= k:
            pred[i-1] = labels[0]  
        else:
            pred[i-1] = stats.mode(labels[i-(k):(i)])[0]
    np.save(logdir+'/'+dataset+'_blind_preds_'+str(k)+'.npy', pred)
    return pred


if __name__ == '__main__':
    global logdir, labels 
    dataset='CLOC' # Choose one from: ['CLEAR10', 'CLEAR100', 'CGLM', 'CLOC']
    num_processes = 16
    path_to_cldatasets = '/media/bayesiankitten/Hanson/CLDatasets/data/' # Please set directory for labels in the CLDatasets folder
    logdir = '/media/bayesiankitten/Alexander/ACM/blind_logs/' # Please set directory for logs
    os.makedirs(logdir, exist_ok=True)
    
    # Load labels file
    # Note: When actually picking the k, use pretrain_labels.hdf5 file. This is for analysis plots, shown in the paper (both have same results, but different purpose).
    with h5py.File(f'{path_to_cldatasets}/{dataset}/order_files/train_labels.hdf5', 'r') as f:
        labels = np.array(f['store_list'])[1:]

    labels = np.array(labels, dtype=np.uint16)
    pred = np.ones_like(labels)[1:]

    # Get dataset mode, get performance to check dataset imbalance
    if not exists(logdir+'/'+dataset+'_mode.npy'):
        modelabel = stats.mode(labels)[0]
        pred = pred*modelabel
        np.save(logdir+'/'+dataset+'_mode.npy', pred)
    
    pred = np.load(logdir+'/'+dataset+'_mode.npy')
    gt = labels[1:]
    acc = np.equal(gt,pred)*1.0
    idx = np.arange(acc.shape[0])+1
    cumacc = np.cumsum(acc)/idx
    print(f'IID Best Classifier: {cumacc.mean()}')
    
    # Get blind classifier performance
    for k in [1, 2, 3, 5, 7, 10, 20, 25, 35, 50, 75, 100, 150, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 75000]:
        if not exists(logdir+'/'+dataset+'_blind_preds_'+str(k)+'.npy'):
            print(f'Processing mode of past: {k}')
            p = multiprocessing.Pool(num_processes)
            result = [] 
            result.append(p.apply_async(get_preds, [k]))
            for r in result:
                r.wait()
            
    # Show results, see the degree of label correlation        
    for k in [1, 2, 3, 5, 7, 10, 20, 25, 35, 50, 75, 100, 150, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 75000]:        
        pred = np.load(logdir+'/'+dataset+'_blind_preds_'+str(k)+'.npy')
        gt = labels[1:]
        acc = np.equal(gt,pred)*1.0
        idx = np.arange(acc.shape[0])+1
        cumacc = np.cumsum(acc)/idx
        print(f'Blind Classifier @ {k}: {cumacc.mean()}')

    print('Extracted all blind classifier results!')
