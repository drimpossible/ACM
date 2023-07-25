import argparse

def parse_args():
   parser = argparse.ArgumentParser(description='main.py')
   # Changing options -- Apart from these arguments, we do not mess with other arguments

   ## Paths
   parser.add_argument('--data_dir', type=str, default='TYPE HERE', help='Full path to directory where all datasets are stored')
   parser.add_argument('--log_dir', type=str, default='TYPE HERE', help='Full path to the directory where all logs are stored')
   parser.add_argument('--order_file_dir', type=str, default='TYPE HERE', help='Full path to the order file')
   
   ## Dataset
   parser.add_argument('--dataset', type=str, default='cglm', help='Name of Dataset', choices=['cglm', 'imagenet', 'cyfcc'])
   parser.add_argument('--num_classes', type=int, default=713, choices=[10788, 1000, 713], help='Number of number of classes')
   parser.add_argument('--train_batch_size', type=int, default=896, help='Batch size to be used in training')
   parser.add_argument('--test_batch_size', type=int, default=4608, help='Batch size to be used in training')
   parser.add_argument('--total_iterations', type=int, default=1800, help='Fraction of total dataset used for ptraining')
   
   ## Model
   parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50','resnet50_random','resnet50_I1B','resnet50_dino','xcit_dino','resnet50_V2'], help='Model architecture')
   parser.add_argument('--mode', type=str, default='ACM', choices=['AdaptedACM', 'ACM', 'ER'], help='Training type')
   parser.add_argument('--embed_size', type=int, default=512, help='Embedding dimensions for retrieval tasks')
   
   ## Experiment Deets
   parser.add_argument('--exp_name', type=str, default='test', help='Full path to the order file')
   parser.add_argument('--maxlr', type=float, default=0.2, help='Starting Learning rate')
   parser.add_argument('--num_gpus', type=int, default=8, help="Number of GPUs used in training")
   
   # Default options
   parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
   parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
   parser.add_argument('--clip', type=float, default=2.0, help='Gradient Clipped if val >= clip')
   parser.add_argument('--num_workers', type=int, default=8, help='Starting Learning rate')
   parser.add_argument('--print_freq', type=int, default=1000, help='Printing utils')
   parser.add_argument('--prefrac', type=float, default=0.2, help='Fraction of total dataset used for pretraining (in CGLM)')
   parser.add_argument('--chunk_idx', type=int, default=0, help='Parallelize ACM running by sending multiple jobs which compute it on parts')
   parser.add_argument('--num_per_chunk', type=int, default=524288, help='Number of Chunks to Subdivide Data to')
   parser.add_argument('--num_gdsteps', type=int, default=1, help='Number of gradient descent steps')
   parser.add_argument('--extract_feats',action='store_true')

   parser.add_argument('--fc_only',action='store_true')
   parser.add_argument('--fc', type=str, default=None, help='Full path to the order file')

   parser.add_argument('--sampler', type=str, default='mixed', help='Full path to the order file')

   parser.add_argument('--delay', type=int, default=0, help='Sets delay in terms of training samples.')
   parser.add_argument('--cosine',action='store_true')



   opt = parser.parse_args()
   return opt
