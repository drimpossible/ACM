import argparse 

def parse_args():
   parser = argparse.ArgumentParser(description='main.py')
   parser.add_argument('--log_dir', type=str, default='../logs/', help='Full path to the directory where all logs are stored')
   parser.add_argument('--num_classes', type=int, default=10788, choices=[10788, 713], help='Number of number of classes')
   parser.add_argument('--online_classifier', type=str, default='HNSW_KNN', help='Name of online classifier', choices=['OLR', 'HNSW_KNN', 'SVM', 'ContextualMemoryTree'])
   parser.add_argument('--search_metric', type=str, default='cosine', choices=['cosine', 'l2'], help='Types of search')
   parser.add_argument('--num_neighbours', type=int, default=1, help='Types of search')
   parser.add_argument('--feat_size', type=int, default=512, help='Types of search')
   parser.add_argument('--HNSW_ef', type=int, default=200, help='Types of search')
   parser.add_argument('--HNSW_M', type=int, default=25, help='Types of search')
   parser.add_argument('--lr', type=float, default=0.5, help='Types of search')
   parser.add_argument('--wd', type=float, default=1e-5, help='Types of search')
   parser.add_argument('--normalize_input', action="store_true", help='Normalize the input to the search')
   parser.add_argument('--online_exp_name', type=str, default='test', help='Full path to the order file')
   parser.add_argument('--num_online_workers', type=int, default=12, help='Starting Learning rate')
   parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
   parser.add_argument('--print_freq', type=int, default=25000, help='Printing utils')
   opt = parser.parse_args()
   return opt
    
