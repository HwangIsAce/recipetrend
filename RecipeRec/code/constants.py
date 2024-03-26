import torch

class CONSTANTS:
    Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_test_neg = 100
    dataset_folder = '../data'
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'