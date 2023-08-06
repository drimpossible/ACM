import argparse 

def parse_args():
   parser = argparse.ArgumentParser(description='main.py')
   parser.add_argument('--feature_path', type=str, default='../../../improved_knn/saved_features/', help='Full path to the directory where all logs are stored')
   parser.add_argument('--log_dir', type=str, default='../../logs/', help='Full path to the directory where all logs are stored')
   parser.add_argument('--model', type=str, default='xcit_dino', choices=['xcit_dino','resnet50','rsnet50_i1b'], help='Model used for classification')
   parser.add_argument('--online_classifier', type=str, default='ApproxKNearestNeighbours', help='Name of online classifier', choices=['OnlineLogisticClassification_VowpalWabbit', 
                                                                'OnlineSVM_VowpalWabbit', 'OnlineSVM_Scikit', 'OnlineLogisticClassification_Scikit', 
                                                                'HuberLossClassifier_Scikit', 'ContextualMemoryTree',  'KNearestNeighbours', 'ApproxKNearestNeighbours', 
                                                                'NearestClassMean', 'StreamingLinearDiscriminantAnalysis'])
   parser.add_argument('--dataset', type=str, default='cglm', help='Name of dataset', choices=['clear10', 'clear100', 'cglm', 'cloc'])
   parser.add_argument('--search_metric', type=str, default='cosine', choices=['cosine', 'l2'], help='Types of search')
   parser.add_argument('--HNSW_ef', type=int, default=200, help='Types of search')
   parser.add_argument('--HNSW_M', type=int, default=25, help='Types of search')
   parser.add_argument('--lr', type=float, default=0.5, help='Types of search')
   parser.add_argument('--wd', type=float, default=1e-4, help='Types of search')
   parser.add_argument('--normalize_input', action="store_true", help='Normalize the input to the search')
   parser.add_argument('--gpu', action="store_true", help='Peform online learning on GPU')
   parser.add_argument('--online_exp_name', type=str, default='test', help='Full path to the order file')
   parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
   parser.add_argument('--print_freq', type=int, default=2500, help='Printing utils')
   opt = parser.parse_args()
   return opt
    
