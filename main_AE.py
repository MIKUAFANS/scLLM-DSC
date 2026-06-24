# -*- encoding: utf-8 -*-

import os
import argparse
import random
import subprocess
from loguru import logger
import numpy as np
import pickle
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.preprocessing import LabelEncoder

from model import AE_GAT, FULL, AE_NN, FULL_NN, ClusterAssignment
from model import MultiModalContrastiveModel
from model import sinkhorn, evaluation, get_laplace_matrix
import torch.nn as nn
import warnings
import torch.nn.functional as F
import scanpy as sc
from model import prepro
import h5py

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


from time import time
warnings.filterwarnings('ignore')



# Default paths relative to project root
PROJECT_ROOT = os.path.dirname(__file__)
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets')
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output')
DEFAULT_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'embeddings')
DEFAULT_RESULT_PATH = os.path.join(PROJECT_ROOT, 'result')
DEFAULT_LOG_PATH = os.path.join(PROJECT_ROOT, 'log')

label_dataset_1 = ['Xiaoping_mouse_bladder_cell','Junyue_worm_neuron_cell',
                   'Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='scLLM-DSC: LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering')
    parser.add_argument('--dataname', default='Sonya_HumanLiver_counts_top5000', type=str, help='Dataset name (without extension)')
    parser.add_argument('--num_class', default=11, type=int, help='Number of cell types/clusters')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device ID')

    # Path configuration
    parser.add_argument('--dataset_path', default=DEFAULT_DATASET_PATH, type=str, help='Path to datasets directory')
    parser.add_argument('--output_path', default=DEFAULT_OUTPUT_PATH, type=str, help='Path to output directory')
    parser.add_argument('--embedding_path', default=DEFAULT_EMBEDDING_PATH, type=str, help='Path to embeddings directory')
    parser.add_argument('--result_path', default=DEFAULT_RESULT_PATH, type=str, help='Path to results directory')
    parser.add_argument('--log_path', default=DEFAULT_LOG_PATH, type=str, help='Path to logs directory')

    embedding_num = 16
    parser.add_argument('--dims_encoder', default=[256, embedding_num], type=list)
    parser.add_argument('--dims_decoder', default=[embedding_num, 256], type=list)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lambdas', default=5, type=float)
    parser.add_argument('--balancer', default=0.5, type=float) #权衡两个图

    parser.add_argument('--factor_ort', default=1, type=float) #权衡NCUT的正交项
    parser.add_argument('--factor_ncut', default=0.05, type=float) #权衡NCUT
    parser.add_argument('--factor_KL', default=0.5, type=float) #权衡最优传输
    parser.add_argument('--factor_cl', default=0.5, type=float) #权衡对比学习
    parser.add_argument('--factor_mse', default=0.5, type=float) #权衡MSE重构损失
    parser.add_argument('--proj_dim', default=128, type=int)
    parser.add_argument('--lambda_reg', default=1e-2, type=float)
    parser.add_argument('--omega', default=0.5, type=float)
    

    parser.add_argument('--pretrain_model_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_save_path', default='pkl', type=str)
    parser.add_argument('--pretrain_model_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_centers_load_path', default='pkl', type=str)
    parser.add_argument('--pretrain_pseudo_labels_load_path', default='pkl', type=str)

    parser.add_argument('--foldername', default='MAIN_modified', type=str, help='Subfolder name for results/logs')
    parser.add_argument('--noramlize_flag', default=False, type=bool, help='Whether to normalize data')
    parser.add_argument('--species', default="human", type=str, help='Species: "human" or "mouse"')

    args = parser.parse_args()

    # Use dataname as foldername by default
    args.foldername = args.dataname

    # Configure paths based on arguments
    result_folder = os.path.join(args.result_path, args.foldername)
    log_folder = os.path.join(args.log_path, args.foldername)

    args.pretrain_model_save_path = os.path.join(result_folder, f'{args.dataname}_model.pkl')
    args.pretrain_centers_save_path = os.path.join(result_folder, f'{args.dataname}_centers.pkl')
    args.pretrain_pseudo_labels_save_path = os.path.join(result_folder, f'{args.dataname}_pseudo_labels.pkl')
    args.pretrain_model_load_path = os.path.join(result_folder, f'{args.dataname}_model.pkl')
    args.pretrain_centers_load_path = os.path.join(result_folder, f'{args.dataname}_centers.pkl')
    args.pretrain_pseudo_labels_load_path = os.path.join(result_folder, f'{args.dataname}_pseudo_labels.pkl')

    # Create directories if they don't exist
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    args.learning_rate = 1e-3
    args.weight_decay = 1e-4
    args.balancer = 0.21
    args.factor_ort = 0.32
    args.factor_ncut = 0.15
    args.factor_KL = 0.32
    args.factor_mse = 0.3
    args.factor_cl = 0.3

    # Setup logging
    log_file = os.path.join(log_folder, f'{args.dataname}.log')
    logger.add(log_file, rotation="500 MB", level="INFO")
    logger.info(args)

    torch.cuda.set_device(args.gpu)

    # Configure data paths
    datapath = os.path.join(args.dataset_path, args.dataname)
    embd_file_path = os.path.join(args.embedding_path, f'{args.dataname}.h5')
    os.makedirs(args.embedding_path, exist_ok=True)

    weighted_cell_embeddings_path = os.path.join(args.output_path, args.dataname, 'weighted_cell_embeddings.npz')
    cell2sentence_path = os.path.join(args.output_path, args.dataname, 'cell_top_genes_embeddings.npz')

    # Generate gene embeddings if not exist
    if not os.path.exists(weighted_cell_embeddings_path):
        logger.info(f"Gene embeddings not found. Running main_Gene.py...")
        try:
            subprocess.run(['python', 'main_Gene.py',
                          '--dataset', args.dataname+'.h5',
                          '--reference_file', args.species+'.csv',
                          '--dataset_path', args.dataset_path,
                          '--save_path', args.output_path],
                          check=True, shell=False)
        except:
            subprocess.run(['python', 'main_Gene.py',
                          '--dataset', args.dataname+'.h5ad',
                          '--reference_file', args.species+'.csv',
                          '--dataset_path', args.dataset_path,
                          '--save_path', args.output_path],
                          check=True, shell=False)

    weighted_cell_embeddings = np.load(weighted_cell_embeddings_path)['embeddings'] * args.omega
    cell2sentence = np.load(cell2sentence_path)['embeddings']
    weighted_cell_embeddings += cell2sentence * (1 - args.omega)
    # weighted_cell_embeddings = np.load(weighted_cell_embeddings_path)['embeddings']
        
    if args.dataname == 'Meuro_human_Pancreas_cell':
        x, y = prepro(datapath+'.h5')
    elif args.dataname.startswith("Tabula_"):
        adata = sc.read_h5ad(datapath+'.h5ad')
        x = adata.X.toarray()
        cell_type = adata.obs['cell_type'].values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(cell_type)
    else:
        data = h5py.File(datapath+'.h5','r')
        x = data['X'][:]
        y = data['Y'][:]

    if args.dataname == 'Meuro_human_Pancreas_cell':
        x = np.round(x).astype(int)
    if args.dataname in label_dataset_1:
        y = y-1
        
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    args.num_class = len(np.unique(y))

    x_ = torch.nn.functional.normalize(x, p=2, dim=1) 
    args.dim_f = x.shape[1]
    args.dim_t = weighted_cell_embeddings.shape[1]
    
    adj_self_loop = torch.mm(x_, x_.T)
    adj_f = np.abs(pairwise_cosine_similarity(x_, x_))
    adj_f = torch.mm(adj_f, adj_f.T)
    
    L_1 = get_laplace_matrix(adj_self_loop)
    L_2 = get_laplace_matrix(adj_f)

    results = pd.DataFrame(columns=['Seed','ACC','NMI','ARI','F1','FMI','V_Measure','HOM','COM','Silhouette'])

    for seed in [3047,3041,2021,2022,2050]:
        logger.info('Seed {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        pre_start_time = time()

        # ################################### PRE-TRAIN ########################################
        Model = AE_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder).cuda()
        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        acc_max = 0
        for epoch in range(1, args.epochs+1):

            h, x_hat = Model.forward(x.cuda())
            z = torch.nn.functional.normalize(h, p=2, dim=0)

            loss_mse = torch.nn.functional.mse_loss(x_hat, x.cuda())

            loss_corvariates = -torch.mm(torch.mm(z.T, (args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z).trace()/len(z.T) #ncut_1
            loss_ort =  torch.nn.functional.mse_loss(torch.mm(z.T,z).view(-1).cuda(),torch.eye(len(z.T)).view(-1).cuda()) #ncut_ort
            loss_ncut = loss_corvariates + args.factor_ort * loss_ort

            loss = args.factor_mse * loss_mse + args.factor_ncut * loss_ncut

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z.cpu().numpy())
                # acc, nmi, ari, f1_macro = evaluation(y, kmeans.labels_)
                acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, _ = evaluation(y, kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)

                logger.info('Epoch {}/{} Pre-Train ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro))
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_mse: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss_mse.cpu().item(), loss.cpu().item()))

                if acc > acc_max:
                    acc_max = acc
                    torch.save(Model.state_dict(), args.pretrain_model_save_path)
                    with open(args.pretrain_centers_save_path,'wb') as save1:
                        pickle.dump(centers, save1, protocol=pickle.HIGHEST_PROTOCOL)
                    pseudo_labels = torch.LongTensor(kmeans.labels_)
                    with open(args.pretrain_pseudo_labels_save_path,'wb') as save2:
                        pickle.dump(pseudo_labels, save2, protocol=pickle.HIGHEST_PROTOCOL)
        pre_time = time() - pre_start_time
        train_start_time = time()

    
        ####################################### TRAIN ########################################
        Model = FULL_NN(dim_input=x.shape[1], dims_encoder=args.dims_encoder, dims_decoder=args.dims_decoder, num_class=args.num_class, \
                    pretrain_model_load_path=args.pretrain_model_load_path).cuda()
        cl_model = MultiModalContrastiveModel(
            # dim_f=args.dim_f,     # feature embedding
            dim_f=embedding_num,     # feature embedding
            dim_t=args.dim_t,     # text embedding
            # proj_dim=args.proj_dim,
            proj_dim=embedding_num,
            lambda_reg=args.lambda_reg
        ).cuda()

        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate)
        with open(args.pretrain_centers_load_path,'rb') as load1:
            centers = pickle.load(load1).cuda()
        with open(args.pretrain_pseudo_labels_load_path,'rb') as load2:
            pseudo_labels = pickle.load(load2).cuda()

        acc_max, nmi_max, ari_max, f1_macro_max, fmi_max, v_measure_max, hom_max, com_max, silhouette_max = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for epoch in range(1, args.epochs+1):
            z_f, x_hat = Model(x.cuda())
            z_t = torch.tensor(weighted_cell_embeddings, dtype=torch.float).cuda()
            centers = centers.detach()

            loss_mse = torch.nn.functional.mse_loss(x_hat, x.cuda())

            loss_corvariates = -torch.mm(torch.mm(z_f.T, ( args.balancer * L_1.cuda() + (1-args.balancer) * L_2.cuda())),z_f).trace()/len(z_f.T)
            loss_ort = torch.nn.functional.mse_loss(torch.mm(z_f.T,z_f).view(-1).cuda(),torch.eye(len(z_f.T)).view(-1).cuda())
            loss_ncut = loss_corvariates + args.factor_ort * loss_ort

            loss_cl, Z = cl_model(z_f, z_t)
       
            #### DEC 
            class_assign_model = ClusterAssignment(args.num_class, len(Z.T), 1, centers).cuda()
            temp_class = class_assign_model(Z.cuda())
            if epoch == 1:
                p_distribution = torch.tensor(sinkhorn(temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)
            elif epoch // 10 == 0:
                p_distribution = torch.tensor(sinkhorn(temp_class.cpu().detach().numpy(), args.lambdas, torch.ones(x.shape[0]).numpy(), torch.tensor([torch.sum(pseudo_labels==i) for i in range(args.num_class)]).numpy())).float().cuda().detach()
                p_distribution = p_distribution.detach()
                q_max, q_max_index = torch.max(p_distribution, dim=1)

            KL_loss_function = nn.KLDivLoss(reduction='sum') 
            loss_KL = KL_loss_function(temp_class.cuda(), p_distribution.cuda())

            loss = args.factor_mse * loss_mse + args.factor_ncut * loss_ncut + args.factor_KL * loss_KL + args.factor_cl * loss_cl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                kmeans = KMeans(n_clusters=args.num_class, random_state=2021, n_init=20).fit(z_f.cpu().numpy())
                acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, y_pred_ = evaluation(y, kmeans.labels_)
                silhouette = silhouette_score(z.cpu().numpy(), y_pred_)
                if acc_max < acc:
                    acc_max, nmi_max, ari_max, f1_macro_max, fmi_max, v_measure_max, hom_max, com_max, silhouette_max = acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, silhouette
                    with h5py.File(embd_file_path, 'w') as file:
                        file.create_dataset('X', data=Z.cpu().numpy())
                        file.create_dataset('Y', data = y_pred_)
                pseudo_labels = torch.LongTensor(kmeans.labels_)
                centers = torch.tensor(kmeans.cluster_centers_)
                #### logger
                logger.info('Epoch {}/{} | loss_corvariate: {:.6f} | loss_ort: {:.6f} | loss_ncut: {:.6f} | loss_mse: {:.6f} | loss_KL: {:.6f} | loss_total: {:.6f}'.format(epoch, args.epochs, loss_corvariates.cpu().item(), loss_ort.cpu().item(), loss_ncut.cpu().item(), loss_mse.cpu().item(), loss_KL.cpu().item(), loss.cpu().item()))
                logger.info('Epoch {}/{} ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, FMI: {:4f}, V_Measure: {:.4f}, HOM: {:.4f}, COM: {:.4f}, silhouette: {:.4f}'.format(epoch, args.epochs, acc, nmi, ari, f1_macro, fmi, v_measure, hom, com, silhouette))
        logger.info('MAX ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, FMI: {:4f}, V_Measure: {:.4f}, HOM: {:.4f}, COM: {:.4f}, silhouette: {:.4f}'.format(acc_max, nmi_max, ari_max, f1_macro_max, fmi_max, v_measure_max, hom_max, com_max, silhouette_max))
        results.loc[len(results)] = [seed, acc_max, nmi_max, ari_max, f1_macro_max, fmi_max, v_measure_max, hom_max, com_max, silhouette_max]
        

        train_time = time() - train_start_time
        all_time = time() - pre_start_time
        logger.info('dataset_name:{},pre_time:{:.7f},train_time:{:.7f},all_time:{:.7f}'.format(args.dataname,pre_time,train_time,all_time))
        logger.info('Average_time,pre_time:{:.7f},train_time:{:.7f},all_time:{:.7f}'.format(pre_time/args.epochs,train_time/args.epochs,all_time/args.epochs))
        logger.info('seed:{}, dataset_name:{}, learning_rate:{}, weight_decay:{}, balancer:{}, factor_ort:{}, factor_ncut:{}, factor_KL:{}, factor_mse:{}'.format(seed, args.dataname, args.learning_rate, args.weight_decay, args.balancer, args.factor_ort, args.factor_ncut, args.factor_KL, args.factor_mse))

    # Save results
    result_save_path = os.path.join(result_folder, f'{args.dataname}_results.csv')
    results.to_csv(result_save_path, index=False)
    logger.info(f'Results saved to {result_save_path}')