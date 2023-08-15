import random, os, online_clfs, time, torch
import numpy as np
from opts import parse_args 
from sklearn.preprocessing import LabelEncoder


def load_dataset(model, dataset):
    #pretrain_X = np.load(os.path.join(opt.feature_path, f"xcit_dino_{dataset}_pretrain_features.npy"))
    pretrain_y = np.load(os.path.join(opt.feature_path, f"xcit_dino_{dataset}_pretrain_labels.npy"))
    pretrain_X = None
    #pretrain_y = None
    train_X = np.load(os.path.join(opt.feature_path, f"{model}_{dataset}_train_features.npy"))
    train_y = np.load(os.path.join(opt.feature_path, f"{model}_{dataset}_train_labels.npy"))
    test_X = np.load(os.path.join(opt.feature_path, f"{model}_{dataset}_test_features.npy"))
    test_y = np.load(os.path.join(opt.feature_path, f"{model}_{dataset}_test_labels.npy"))
    
    # Checks
    #assert(pretrain_X.shape[0] == pretrain_y.shape[0])
    assert(train_X.shape[0] == train_y.shape[0])
    assert(test_X.shape[0] == test_y.shape[0])
    #assert(pretrain_X.shape[1] == train_X.shape[1] == test_X.shape[1])

    #print("Total pretrain rows in the dataset:", pretrain_X.shape[0])
    print("Total train rows in the dataset:", train_X.shape[0])
    print("Total test rows in the dataset:", test_X.shape[0])

    # Normalize labels
    le = LabelEncoder()
    le.fit(np.concatenate((train_y, pretrain_y)))
    #le.fit(train_y)
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)
    #pretrain_y = le.transform(pretrain_y)

    return pretrain_X, pretrain_y, train_X, train_y, test_X, test_y, le.classes_.shape[0]


if __name__ == '__main__':
    # Parse arguments and init loggers
    opt = parse_args()
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    print('==> Params for this experiment:'+str(opt))

    pretrain_X, pretrain_y, train_X, train_y, test_X, test_y, num_classes = load_dataset(opt.model, dataset=opt.dataset)
    opt.feature_dim, opt.num_classes = train_X.shape[1], num_classes

    normalizer = online_clfs.Normalizer(dim=opt.feature_dim)
    online_clf = getattr(online_clfs, opt.online_classifier)(opt=opt)

    predarr, labelarr, acc = np.zeros(train_y.shape[0], dtype='u2'), np.zeros(train_y.shape[0], dtype='u2'), np.zeros(train_y.shape[0], dtype='bool')
    start_time = time.time()
    
    for i in range(train_X.shape[0]- opt.delay):
        feat_learn = train_X[i]
        feat_pred = train_X[i+opt.delay]

        if opt.normalize_input: 
            feat_learn = normalizer.update_and_transform(feat_learn)
            feat_pred = normalizer.transform(feat_pred)

        if i >= 256+opt.delay: # Slightly shifted ahead start point to warmup all classifiers, avoids weird jagged artifacts in plots
            if opt.online_classifier == 'ApproxKNearestNeighbours':
                if (i+1)%opt.update_k <= (opt.update_size-1):
                    pred = online_clf.full_predict_step(x=feat_pred, y=train_y[i+opt.delay])
                else:
                    pred = online_clf.predict_step(x=feat_pred)     
                if i>opt.update_k and (i+1)%opt.update_k == (opt.update_size-1+opt.delay):
                    online_clf.deploy_num_neighbours()
            else:
                pred = online_clf.predict_step(x=feat_pred)
                
            predarr[i+opt.delay] = int(pred)
            labelarr[i+opt.delay] = int(train_y[i+opt.delay])
            is_correct = (int(pred)==int(train_y[i+opt.delay]))
            acc[i+opt.delay] = is_correct*1.0
            
        if (i+1)%opt.print_freq == 0:
            cum_acc = np.array(acc[:i]).mean()
            print(f'Step:\t{i}\tCumul Acc:{cum_acc}')

        online_clf.learn_step(x=feat_learn, y=np.array([train_y[i]]))

    total_time = time.time() - start_time
    print(f'Total time taken: {total_time:.4f}')
    os.makedirs(opt.log_dir, exist_ok=True)
    np.save(os.path.join(opt.log_dir, f"{opt.model}_{opt.dataset}_{opt.online_classifier}_{opt.lr}_{opt.wd}_online_preds_{opt.online_exp_name}.npy"), predarr[256:])
    np.save(os.path.join(opt.log_dir, f"{opt.model}_{opt.dataset}_{opt.online_classifier}_{opt.lr}_{opt.wd}_online_labels_{opt.online_exp_name}.npy"), labelarr[256:])

    print('==> Testing..')
    start_time = time.time()

    preds = online_clf.predict_step(x=test_X)
    np.save(os.path.join(opt.log_dir, f"{opt.model}_{opt.dataset}_{opt.online_classifier}_{opt.lr}_{opt.wd}_test_preds_{opt.online_exp_name}.npy"), preds)
    np.save(os.path.join(opt.log_dir, f"{opt.model}_{opt.dataset}_{opt.online_classifier}_{opt.lr}_{opt.wd}_test_labels_{opt.online_exp_name}.npy"), test_y)
