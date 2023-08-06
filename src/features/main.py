import os, torch, random, warnings
import numpy as np
from os.path import exists
from opts import parse_args
from torch.optim import SGD
import torchvision.models as models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader
from torchutils import ListFolder, EmbedLinearClassifier, LinearLR
from trainutils import get_logger, load_filelist, make_data,  seed_everything, save_model
import torch.nn as nn
import copy
warnings.filterwarnings('ignore')

# Global variables 
test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def AdaptedACM(opt, model, ftmodel, logger):
    # Load pretraining data
    os.makedirs(opt.log_dir+'/'+opt.exp_name+'/', exist_ok=True)
    image_paths = opt.order_file_dir+'/pretrain_image_paths.hdf5'
    y = opt.order_file_dir+'/pretrain_labels.hdf5'
    train_routing_index_path = f'{opt.order_file_dir}/AdaptedACM.hdf5'
    train_dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, transform=train_transforms)
    trainloader = DataLoader(train_dataset, shuffle=True, drop_last=False, num_workers=opt.num_workers, batch_size=opt.train_batch_size, pin_memory=True)   
    logger.debug('Length of dataset: '+str(len(train_dataset)))
    logger.debug('Length of dataloader: '+str(len(trainloader)))
    model.cuda()
    model.eval()
    ftmodel.cuda()
    ftmodel.train()
    optimizer = SGD(ftmodel.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)  
    scheduler = LinearLR(optimizer, T=opt.total_iterations)

    total_epochs = opt.tora

    # Training loop
    for epoch in range(total_epochs):
        for (images, targets, _, _, _) in trainloader:
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            with torch.no_grad():
                feats = model(images)
            outputs = ftmodel(feats)  
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    torch.save(ftmodel.state_dict(), opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt')


def ACM(opt, model, logger, ftmodel=None):
    # Load pretraining data
    os.makedirs(opt.log_dir+'/'+opt.exp_name+'/', exist_ok=True)
    image_paths = opt.order_file_dir+'/train_image_paths.hdf5'
    y = opt.order_file_dir+'/train_labels.hdf5'

    offset = opt.chunk_idx*opt.num_per_chunk
    dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, offset=offset, is_train=False, transform=test_transforms)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.test_batch_size, pin_memory=True)
    print('Length of dataloader: '+str(len(loader)))

    labelarr, featarr = np.zeros(len(y),dtype='u2'), np.zeros((opt.num_per_chunk, opt.embed_size),dtype=np.float32)
    if ftmodel is not None: predarr = np.zeros(len(y),dtype='u2')

    logger.info('==> Extracting features from idx '+str(offset)+'..')
    model.cuda()
    model.eval()

    if ftmodel is not None: 
        ftmodel.cuda()
        ftmodel.eval()

    # We will collect predictions, labels and features in corresponding numpy arrays
    with torch.inference_mode():
        for (image, label, _,  sel_idx, _) in loader:
            image = image.cuda(non_blocking=True)
            feat = model(image)
            if ftmodel is not None:
                feat = ftmodel.embed(feat)
                predprobs = ftmodel.fc(ftmodel.norm(feat))
                pred = torch.argmax(predprobs, dim=1)
                predarr[sel_idx%num_per_chunk] = pred.cpu().numpy() 
            
            labelarr[sel_idx%num_per_chunk] = label.cpu().numpy()
            featarr[sel_idx%num_per_chunk] = feat.cpu().numpy()
        
            if ((sel_test_idx.max()+1)//opt.num_per_chunk) > opt.chunk_idx:
                np.save(opt.log_dir+'/'+opt.exp_name+f'/features_{chunk_idx}.npy', featarr)
                if ftmodel is not None: np.save(opt.log_dir+'/'+opt.exp_name+f'/preds_{chunk_idx}.npy', predarr)
                np.save(opt.log_dir+'/'+opt.exp_name+f'/labels_{chunk_idx}.npy', labelarr)
    return

    
if __name__ == '__main__':
    # Parse arguments and init loggers
    torch.multiprocessing.set_sharing_strategy('file_system')
    opt = parse_args()

    opt.exp_name = f'{opt.dataset}_{opt.model}_{opt.embed_size}'

    console_logger = get_logger(folder=opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info('==> Params for this experiment:'+str(opt))
    seed_everything(opt.seed)

    if opt.model == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif opt.model == 'resnet50_I1B':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    elif opt.model == 'xcit_dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
    if 'xcit' not in opt.model: model.fc = torch.nn.Identity()

    if opt.fc_only:
        for param in model.parameters():
            param.requires_grad = False

    dim = 512 if opt.model=='xcit_dino' else 2048
    ftmodel = EmbedLinearClassifier(dim=dim, embed_size=opt.embed_size, num_classes=opt.num_classes, cosfc=opt.cosine)
    
    if opt.mode=='AdaptedACM':
        assert(ftmodel is not None)
        if exists(opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt'):
            ftmodel.load_state_dict(torch.load(opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt'))
        else:
            AdaptedACM(opt=opt, model=model, ftmodel=ftmodel, logger=console_logger)
        ACM(opt=opt, model=model, ftmodel=ftmodel, logger=console_logger)
    elif opt.mode=='ACM':
        ACM(opt=opt, model=model, logger=console_logger)