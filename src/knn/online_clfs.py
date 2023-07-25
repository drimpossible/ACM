from vowpalwabbit import Workspace
from vowpalwabbit.dftovw import DFtoVW, SimpleLabel, Feature
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import hnswlib
import numpy as np
import threading
import pickle, math

class Normalizer():
    def __init__(self, size=None, dim=None, mean=None, unnormalized_var=None):
        assert(dim is not None)
        self.dim = dim
        if mean is not None: 
            assert(size is not None and unnormalized_var is not None)
            self.size = size
            self.mean = mean
            self.var_unnormalized = unnormalized_var
        else:
            self.size = 0
            self.mean = np.zeros(dim)
            self.var_unnormalized = np.zeros(dim)

    def update_and_transform(self, X):
        assert(X.shape[0] >= 1 and X.shape[1] == self.dim and len(X.shape)==2)

        for idx in range(X.shape[0]):
            self.size += 1
            new_mean = self.mean + (X[idx] - self.mean)/self.size
            self.var_unnormalized = self.var_unnormalized + (X[idx] - self.mean)*(X[idx] - new_mean)
            self.mean = new_mean
            std = np.sqrt(self.var_unnormalized/(self.size-1))
            X[idx] = X[idx] - self.mean/std 
        return X

    def transform(self, X):
        assert(X.shape[0] >= 1 and X.shape[1] == self.dim and len(X.shape)==2)
        if self.size > 2:
            std = np.sqrt(self.var_unnormalized/(self.size-1))
            return (X - self.mean)/std 
        else:
            return X


class OLR():
    def __init__(self, opt):
        self.model = Workspace(oaa=opt.num_classes, loss_function='logistic', b=30, l=opt.lr)
        self.cols = ['label']
        for i in range(opt.feat_size): self.cols.append('f_'+str(i))

    def learn_step(self, X, y):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, X, num_neighbours):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class SVM():
    def __init__(self, opt):
        self.model = Workspace(oaa=opt.num_classes, loss_function='hinge', b=30, l=opt.lr, l2=opt.wd)
        self.cols = ['label']
        for i in range(opt.feat_size): self.cols.append('f_'+str(i))

    def learn_step(self, X, y):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, X, num_neighbours):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class ContextualMemoryTree():
    def __init__(self, opt):
        num_nodes = opt.num_classes/(np.log(opt.num_classes)/np.log(2)*10) 
        self.model = Workspace("--memory_tree "+str(num_nodes)+" --max_number_of_labels "+str(opt.num_classes)+' --online --dream_at_update 1 --leaf_example_multiplier 10  --dream_repeats 12 --learn_at_leaf --alpha 0.1 -l '+str(opt.lr)+' -b 30 -c --loss_function squared --sort_features')
        self.cols = ['label']
        for i in range(opt.feat_size): self.cols.append('f_'+str(i))

    def learn_step(self, X, y):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, X, num_neighbours):
        df = pd.DataFrame(X, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class KNN():
    def __init__(self, opt):
        self.num_neighbours = opt.num_neighbours
        self.train_X, self.train_y = None, None

    def learn_step(self, X, y):
        if self.train_X is not None:
            self.train_y = np.concatenate((self.train_y, y), axis=0)
            self.train_X = np.concatenate((self.train_X, X), axis=0)
        else:
            self.train_y = y
            self.train_X = X

    def predict_step(self, X):
        model = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        model.fit(self.train_X, self.train_y)
        out = model.predict(X)
        return out


class HNSW_KNN():
    # https://raw.githubusercontent.com/nmslib/nmslib/master/manual/latex/manual.pdf
    def __init__(self, opt):
        self.index = hnswlib.Index(space=opt.search_metric, dim=opt.feat_size)
        self.lock = threading.Lock()
        self.idx2label = {}
        self.cur_idx = 0
        self.dset_size = 1024
        self.index.init_index(max_elements=self.dset_size, ef_construction=opt.HNSW_ef, M=opt.HNSW_M)
        self.num_neighbours = opt.num_neighbours 
        #self.set_num_threads(num_threads=opt.num_online_workers)

    def learn_step(self, X, y):
#        print(X.shape, y.shape)
        assert(X.shape[0]==y.shape[0])

        num_added = X.shape[0]
        start_idx = self.cur_idx
        self.cur_idx += num_added
        
        if self.cur_idx >= self.dset_size - 2:
            with self.lock:
                self.dset_size = pow(2, math.ceil(math.log2(self.cur_idx)))
                self.dset_size *= 4
                self.index.resize_index(self.dset_size)
        
        idx = []
        for label in range(y.shape[0]):
            idx.append(start_idx)
            self.idx2label[start_idx] = y[label]
            start_idx += 1
        
        self.index.add_items(data=X, ids=np.asarray(idx))

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def load_index(self, path):
        self.index.load_index(path)

        with open(path + ".pkl", "rb") as f:
            self.cur_idx, self.idx2label = pickle.load(f)

    def save_index(self, path):
        self.index.save_index(path)
        with open(path + ".pkl", "wb") as f:
            pickle.dump((self.cur_idx, self.idx2label), f)

    def set_num_threads(self, num_threads):
        self.index.set_num_threads(num_threads)

    def predict_step(self, X, num_neighbours, mode='multiclass_clf'):
        assert(mode in ['multiclass_clf','retrieval'])
        
        idx_pred_list, distances = self.index.knn_query(data=X, k=num_neighbours)
        labels = []
        
        for idx_pred in idx_pred_list:
            possible_labels = np.array([self.idx2label[idx] for idx in idx_pred]).astype(int)

            if mode == 'multiclass_clf':
                counts = np.bincount(possible_labels)
                label = np.argmax(counts)
                labels.append(label if counts[label] > 1 else possible_labels[0])
            elif mode == 'retrieval':
                labels.append(possible_labels)

        return np.array(labels)
