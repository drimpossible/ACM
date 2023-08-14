import numpy as np
import pandas as pd
import hnswlib, threading, pickle, math, torch, scipy
from sklearn.linear_model import SGDClassifier
from vowpalwabbit import Workspace
from vowpalwabbit.dftovw import DFtoVW, SimpleLabel, Feature


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

    def update_and_transform(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        for idx in range(x.shape[0]):
            self.size += 1
            new_mean = self.mean + (x[idx] - self.mean)/self.size
            self.var_unnormalized = self.var_unnormalized + (x[idx] - self.mean)*(x[idx] - new_mean)
            self.mean = new_mean
            std = np.sqrt(self.var_unnormalized/(self.size-1))
            x[idx] = x[idx] - self.mean/std 
        return x

    def transform(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        if self.size > 2:
            std = np.sqrt(self.var_unnormalized/(self.size-1))
            return (x - self.mean)/std 
        else:
            return x


class OnlineLogisticClassification_VowpalWabbit():
    def __init__(self, opt):
        self.model = Workspace(oaa=opt.num_classes, loss_function='logistic', b=30, l=opt.lr)
        self.cols = ['label']
        for i in range(opt.feature_dim): self.cols.append('f_'+str(i))

    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class OnlineSVM_VowpalWabbit():
    def __init__(self, opt):
        self.model = Workspace(oaa=opt.num_classes, loss_function='hinge', b=30, l=opt.lr, l2=opt.wd)
        self.cols = ['label']
        for i in range(opt.feature_dim): self.cols.append('f_'+str(i))

    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class OnlineSVM_Scikit():
    def __init__(self, opt):
        self.clf = SGDClassifier(loss='hinge', penalty='l2', alpha=opt.wd, fit_intercept=True, learning_rate='optimal', warm_start=True)
    
    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self.clf.partial_fit(x, y, classes=np.arange(self.num_classes))

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self.clf.predict(x)


class OnlineLogisticClassification_Scikit():
    def __init__(self, opt):
        self.clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=opt.wd, fit_intercept=True, learning_rate='optimal', warm_start=True)
        self.num_classes = opt.num_classes

    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self.clf.partial_fit(x, y, classes=np.arange(self.num_classes))

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self.clf.predict(x)
    

class HuberLossClassifier_Scikit():
    def __init__(self, opt):
        self.clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=opt.wd, fit_intercept=True, learning_rate='optimal', warm_start=True)
        self.num_classes = opt.num_classes

    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self.clf.partial_fit(x, y, classes=np.arange(self.num_classes))

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self.clf.predict(x)
    

class ContextualMemoryTree():
    def __init__(self, opt):
        num_nodes = opt.num_classes/(np.log(opt.num_classes)/np.log(2)*10) 
        self.model = Workspace("--memory_tree "+str(num_nodes)+" --max_number_of_labels "+str(opt.num_classes)+' --online --dream_at_update 1 --leaf_example_multiplier 10  --dream_repeats 12 --learn_at_leaf --alpha 0.1 -l '+str(opt.lr)+' -b 30 -c --loss_function squared --sort_features')
        self.cols = ['label']
        for i in range(opt.feature_dim): self.cols.append('f_'+str(i))

    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = y[0].tolist()
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        self.model.learn(example)

    def predict_step(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        df = pd.DataFrame(x, columns=self.cols[1:])
        df['label'] = [-1]
        feat = DFtoVW(df=df, label=SimpleLabel('label'), features=[Feature(col) for col in self.cols[1:]])
        example = feat.convert_df()[0]
        return self.model.predict(example)


class KNearestNeighbours():
    def __init__(self, opt):
        self.num_neighbours = opt.num_neighbours
        self.train_x, self.train_y = None, None
        self.num_neighbours = 1

        # Set distance function
        if opt.search_metric == 'cosine':
            self.dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        elif opt.search_metric == 'l2':
            self.dist = torch.nn.PairwiseDistance(p=2)
        assert(opt.search_metric in ['cosine', 'l2'])


    def learn_step(self, x, y):
        with torch.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)

        if self.train_x is not None:
            self.train_y = torch.cat((self.train_y, y), dim=0)
            self.train_x = torch.cat((self.train_x, x), dim=0)
        else:
            self.train_y = y
            self.train_x = x


    def predict_step(self, x):
        with torch.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)

        _, idxes = torch.topk(self.dist(x, self.train_x), 1, largest=False)
        labels, _ = torch.mode(self.train_y[idxes], dim=1)

        return labels


class ApproxKNearestNeighbours():
    # https://raw.githubusercontent.com/nmslib/nmslib/master/manual/latex/manual.pdf
    def __init__(self, opt):
        self.index = hnswlib.Index(space=opt.search_metric, dim=opt.feature_dim)
        self.lock = threading.Lock()
        self.cur_idx = 0
        self.dset_size = 65536
        self.idx2label = np.zeros(self.dset_size, dtype=np.int16)
        self.index.init_index(max_elements=self.dset_size, ef_construction=opt.HNSW_ef, M=opt.HNSW_M)
        self.num_neighbours = opt.num_neighbours


    def learn_step(self, x, y):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        assert(x.shape[0]==y.shape[0])
    
        num_added = x.shape[0]
        start_idx = self.cur_idx
        self.cur_idx += num_added
        
        if self.cur_idx >= self.dset_size - 2:
            with self.lock:
                self.dset_size = pow(2, math.ceil(math.log2(self.cur_idx)))
                self.dset_size *= 4
                self.index.resize_index(self.dset_size)

                new_idx2label = np.zeros(self.dset_size, dtype=np.int16)
                new_idx2label[:start_idx] = self.idx2label[:start_idx]
                self.idx2label = new_idx2label
        
        idx = np.arange(start_idx, start_idx + num_added)
        self.idx2label[start_idx:start_idx+num_added] = y
        
        self.index.add_items(data=x, ids=np.asarray(idx))

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

    def predict_step(self, x):
        # Note: y is used only for selecting k for the next step
        # Ideally this should be done in learn_step but to avoid computing neighbours twice, we do it here.
        if x.ndim == 1:
            x = x[np.newaxis, :]

        idxes, _ = self.index.knn_query(data=x, k=self.num_neighbours)
        neighbour_labels = self.idx2label[idxes]
        pred_labels, _ = scipy.stats.mode(neighbour_labels, axis=1)
        return pred_labels


class NearestClassMeanCosine():
    def __init__(self, opt):
        with torch.no_grad():
            # Class means is class sums, divided by number of samples
            self.class_sums = torch.zeros((1, opt.feature_dim, opt.num_classes))
            self.num_samples = torch.zeros((1,1,opt.num_classes))
            self.dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            
            if opt.gpu:
                self.class_sums = self.class_sums.cuda()
                self.num_samples = self.num_samples.cuda()
                self.dist = self.dist.cuda()
            

    def learn_step(self, x, y):
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(2)

        with torch.no_grad():
            # Update class mean and number of samples
            if self.num_samples.shape[0] <= y.shape[0]:
                for index in range(self.num_samples.shape[0]):
                    if (y==index).sum() == 0:
                        continue
                    self.class_sums[:,:,index] += x[y==index].sum(dim=0).squeeze()
                    self.num_samples[index] += (y==index).sum()
            else:
                for index in range(y.shape[0]):
                    self.class_sums[0,:,y[index]] += x[index].squeeze()
                    self.num_samples[y[index]] += 1
    

    def predict_step(self, x):
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(2)

        with torch.no_grad():
            class_means = self.class_sums / (self.num_samples+1e-6)
            distances = self.dist(x, class_means)
            distances = torch.where(distances!=0, distances, 1e5)
            return torch.argmin(distances, dim=1)
        

class NearestClassMeanL2():
    def __init__(self, opt):
        with torch.no_grad():
            # Class means is class sums, divided by number of samples
            self.class_sums = torch.zeros((opt.num_classes, opt.feature_dim))
            self.num_samples = torch.zeros((opt.num_classes,1))
                        
            if opt.gpu:
                self.class_sums = self.class_sums.cuda()
                self.num_samples = self.num_samples.cuda()
                self.dist = self.dist.cuda()
            

    def learn_step(self, x, y):
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            # Update class mean and number of samples
            if self.num_samples.shape[0] <= y.shape[0]:
                for index in range(self.num_samples.shape[0]):
                    if (y==index).sum() == 0:
                        continue
                    self.class_sums[index,:] += x[y==index].sum(dim=0).squeeze()
                    self.num_samples[index] += (y==index).sum()
            else:
                for index in range(y.shape[0]):
                    self.class_sums[y[index],:] += x[index].squeeze()
                    self.num_samples[y[index]] += 1
    

    def predict_step(self, x):
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            class_means = (self.class_sums / (self.num_samples+1e-6)).unsqueeze(0)
            x = x.unsqueeze(0)
            distances = torch.cdist(x, class_means, p=2.0).squeeze(dim=0)
            distances = torch.where(distances!=0, distances, 1e5)
            return torch.argmin(distances, dim=1)    
        

class StreamingLinearDiscriminantAnalysis():
    def __init__(self, opt):
        with torch.no_grad():
            self.feature_dim = opt.feature_dim
            self.num_classes = opt.num_classes
            self.shrinkage_param = 1e-4
            self.streaming_update_sigma = True

            # setup weights for SLDA
            self.muK = torch.zeros((opt.num_classes, opt.feature_dim))
            self.cK = torch.zeros(opt.num_classes)
            self.Sigma = torch.ones((opt.feature_dim, opt.feature_dim))
            self.Lambda = torch.zeros_like(self.Sigma)
            self.num_updates = 0
            self.prev_num_updates = -1

            if opt.gpu:
                self.muK = self.muK.cuda()
                self.cK = self.cK.cuda()
                self.Sigma = self.Sigma.cuda()
                self.Lambda = self.Lambda.cuda()


    def learn_step(self, x, y):
        # make sure things are the right shape
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1


    def predict_step(self, x):
        x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = x.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = num_samples

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                # print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.feature_dim)).to(self.Lambda.device) 
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                X = x[start:end]
                scores[start:end, :] = torch.matmul(X, W) - c
            
            # return predictions or probabilities
            return torch.argmax(scores, dim=1)


    def fit_base(self, x, y):
        print('\nFitting Base...')
        x = torch.from_numpy(x)
        # update class means
        for k in torch.unique(y):
            self.muK[k] = x[y == k].mean(0)
            self.cK[k] = x[y == k].shape[0]
        self.num_updates = x.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((x - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.Sigma.device) 