import numpy as np
from PIL import Image
import torch, os, logging, random
from os.path import isfile, isdir, join, exists


def make_data(fname):
    """
    """
    # read data
    fval = open(fname, 'r')
    lines_val = fval.readlines()
    labels = [None] * len(lines_val)
    time = [None] * len(lines_val)
    user = [None] * len(lines_val)
    store_loc = [None] * len(lines_val)
    for i in range(len(lines_val)):
        line_splitted = lines_val[i].split(',')
        labels[i] = int(line_splitted[0])
        time[i] = int(line_splitted[2])
        user[i] = line_splitted[3]
        store_loc[i] = line_splitted[-1][:-1]
    return store_loc, labels, time, user


def load_images(image_paths):
    """
    """
    imgs = []
    for i in range(len(image_paths)):
        img = pil_loader(image_paths[i])
        imgs.append(img)
    return imgs


def convert_imgfolder_to_imglist(data_dir, dataset, split):
    """
    Takes an input datset directory, and converts 
    """
    classes = [f for f in os.listdir(data_dir+'/'+dataset+'/'+split+'/') if isdir(join(data_dir+'/'+dataset+'/'+split,f))]
    img_paths, labels = [], []
    count = 0

    for cls in classes:
        folder = join(data_dir+'/'+dataset+'/'+split+'/', cls)
        imgs = ['/'+split+'/'+cls+'/'+f for f in os.listdir(folder) if (isfile(join(folder,f)))]
        lbls = [count for img in imgs]
        assert(len(imgs)==len(lbls))
        count+=1
        img_paths.extend(imgs)
        labels.extend(lbls)

    assert(len(img_paths)==len(labels))
    return img_paths, labels


def load_filelist(filepath, root_dir, check=False):
    """
    Checks whether images exist, if yes then load it into the filelist. Do check=True for the first run, then turn it off
    """
    imglist, y = [], []
    with open(filepath,'r') as f:
        for line in f:
            label, image, _ = line.strip().split('\t')
            if check: 
                assert(exists(root_dir+'/'+image))
            imglist.append(image.strip())
            y.append(int(label)) 
    return imglist, y


def get_logger(folder):
    """
    Initializes the logger for logging opts and intermediate checks in runs.
    Note: One logger per experiment where reruns get appended to the log.
    """
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fh = logging.FileHandler(os.path.join(folder, 'checkpoint.log'), mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def seed_everything(seed):
    """
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    """
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # An exemption for speed :P


def save_model(opt, model):
    """
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    """
    state = {'opt': opt,
        'state_dict': model.state_dict()}
    filename = opt.log_dir+'/'+opt.exp_name+'/model.pt'
    torch.save(state, filename)