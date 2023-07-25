import multiprocessing
import numpy as np
from scipy import stats
from os.path import exists

def load_filelist(filepath):
    imglist, y = [], []
    with open(filepath,'r') as f:
        for line in f:
            label, _, _ = line.strip().split('\t')
            y.append(int(label)) 
    return imglist, y

def get_preds(shift, k):
    # TODO: Further optimize the stats.mode calculation by a running mode, this code is very slow for very large k which can hinder analysis
    for i in range(1, len(labels)):
        if i <= k+shift:
            pred[i-1] = labels[0]  
        else:
            pred[i-1] = stats.mode(labels[i-(k+shift):(i-shift)])[0][0]
    np.save(LOGDIR+'/shift_'+prefix+'_blind_preds_'+str(k)+'_'+str(shift)+'.npy', pred)
    return pred

if __name__ == '__main__':
    prefix='cglm'
    #prefix='cloc'

    ORDERFILEDIR='../data/'+prefix+'/' 
    LOGDIR='../data/blind/'

    _, labels = load_filelist(filepath=ORDERFILEDIR+'/train.txt')
    labels = np.array(labels, dtype='u2')
    pred = np.ones_like(labels)[1:]

    # Get dataset mode
    modelabel = stats.mode(labels)[0][0]
    pred = pred*modelabel
    np.save(LOGDIR+'/'+prefix+'_mode.npy', pred)

    p = multiprocessing.Pool(16)
    result = []
    for k in [1, 2, 5, 10, 20, 30, 40, 60, 75, 100, 250, 500, 1000, 5000, 25000, 75000]:
            print(k, shift)
            pred = np.ones_like(labels)[1:]
            if not exists(LOGDIR+'/shift_'+prefix+'_blind_preds_'+str(k)+'_'+str(shift)+'.npy'): 
                result.append(p.apply_async(get_preds, [shift,k]))

    for r in result:
        r.wait()

    print('Extracted all blind classifier results!')
