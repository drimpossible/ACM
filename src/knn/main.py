import random, os, online_clfs
import numpy as np
from opts import parse_args 


if __name__ == '__main__':
    # Parse arguments and init loggers
    opt = parse_args()
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    print('==> Params for this experiment:'+str(opt))

    feats, labels = np.load(opt.log_dir+'/'+opt.exp_name+'/full_features.npy'), np.load(opt.log_dir+'/'+opt.exp_name+'/labels.npy') 
    opt.feat_size = feats.shape[1]

    os.makedirs(os.path.join(f'{opt.log_dir}/{opt.exp_name}/', 'online'), exist_ok=True)

    normalizer = online_clfs.Normalizer(dim=opt.feat_size)
    online_clf = getattr(online_clfs, opt.online_classifier)(opt=opt)

    predarr, labelarr, acc = np.zeros(labels.shape[0], dtype='u2'), np.zeros(labels.shape[0], dtype='u2'), np.zeros(labels.shape[0], dtype='bool')

    for i in range(feats.shape[0]):
        feat = np.expand_dims(feats[i], axis=0)

        if opt.normalize_input: 
            feat = normalizer.update_and_transform(feat)
        if i > opt.num_neighbours: 
            pred = online_clf.predict_step(X=feat, num_neighbours=opt.num_neighbours)
            predarr[i] = pred
            labelarr[i] = labels[i]
            is_correct = (int(pred)==int(labels[i]))
            acc[i] = is_correct*1.0
            
        if i%opt.print_freq == 0:
            cum_acc = np.array(acc[:i]).mean()
            print('Step:\t'+str(i)+'\tCumul Acc:\t'+str(cum_acc))

        online_clf.learn_step(X=feat, y=np.array([labels[i]]))

    np.save(opt.log_dir+'/'+opt.exp_name+'/online/'+f'{opt.online_classifier}_{opt.lr}_{opt.wd}_pred_{opt.online_exp_name}.npy', predarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/online/'+f'{opt.online_classifier}_{opt.lr}_{opt.wd}_label_{opt.online_exp_name}.npy', labelarr)