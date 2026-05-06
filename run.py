import os

from utility import parser, set_seed

from Contrast_combination1 import train_SUBGAD1

import torch


if __name__ == '__main__':

    args = parser()
    set_seed(args)
    #books,disney,enron,reddit,weibo
    #for dataset in ['Flickr', 'BlogCatalog', 'pubmed','ACM','cora', 'citeseer']:
    #for dataset in [ 'ACM','disney','books', 'reddit','enron','weibo', 'cora','citeseer','Flickr', 'BlogCatalog', 'pubmed',]:

    datasets = ['cora','citeseer','Flickr', 'BlogCatalog', 'pubmed']
    for dataset in datasets:
        args.dataset = dataset
        if args.dataset in ['BlogCatalog','Flickr','books','disney']:
            args.lamda = 0.0
            args.psi = 2
            args.h = 1
        if args.dataset in [ 'ACM','weibo','enron','reddit']:
            args.lamda = 0.0625   #0.0625
            args.psi = 2
            args.h = 1 #1
        if args.dataset in ['cora','citeseer', 'pubmed',]:
            args.lamda = 0.125  #0.125
            args.psi = 2
            args.h = 1  #2

        if args.lr is None:
            if args.dataset in ['cora', 'citeseer', 'pubmed','books','disney','enron']:
                args.lr = 2e-3
            elif args.dataset in ['BlogCatalog','Flickr']:
                args.lr = 1e-2
            elif args.dataset == 'ACM':
                args.lr = 5e-3

        if args.K_1 is None:
            if args.dataset in ['cora','disney','enron']:
                args.K_1 = 2  #2
            elif args.dataset in ['BlogCatalog','Flickr', 'pubmed']: 
                args.K_1 = 4  #4
            elif args.dataset in ['citeseer', 'ACM' ,'books']:
                args.K_1 = 6  #6

        if args.K_2 is None:
            if args.dataset in ['disney', 'citeseer']:
                args.K_2 = 4  #4
            elif args.dataset in ['cora', 'ACM', 'BlogCatalog', 'Flickr','pubmed','books','enron']:
                args.K_2 = 8  #8


        train_GADMCLG1(args)
        

