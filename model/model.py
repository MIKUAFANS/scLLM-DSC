# -*- encoding: utf-8 -*
import torch
from .utils import variance_regularization, contrastive_loss
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
import torch.nn.functional as F

######################################## GAT Layer ########################################

# dim_in: dim_input , 256 , embedding_num
# dim_out: embedding_num , 256 , dim_imput
class GAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, negative_slope=0.2):
        super(GAT_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x, adj):
        # x (num_nodes, dim_in)
        x_ = torch.mm(x, self.w) # (num_nodes, dim_out)
        scores_target = torch.mm(x_, self.a_target) # (num_nodes, )
        scores_neighbor = torch.mm(x_, self.a_neighbor) # (num_nodes, )
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1) # (num_nodes, num_nodes)
        
        scores = torch.mul(adj, scores)
        scores = self.leakyrelu(scores)
        scores = torch.where(adj>0, scores, -9e15*torch.ones_like(scores)) # score of non-negihbor is -9e15
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        return x_



######################################## GAT Auto_Encoder ########################################


# dims_en: dim_input , 256 , embedding_num
# dims_de: embedding_num , 256 , dim_imput
# num_layer: 2

class AE_GAT(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_GAT, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        # print("self_numlayer", self.num_layer)
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        for index in range(self.num_layer):
            # print("layer:", index, self.dims_en[index], self.dims_en[index+1] )
            # print("layer:", index, self.dims_de[index], self.dims_de[index+1] )
            self.Encoder.append(GAT_Layer(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(GAT_Layer(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x, adj):
        # print("x,adj:", x.shape, adj.shape)
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x, adj)
        h = x
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x, adj)
        x_hat = x      
        return h, x_hat


######################################## NN Auto_Encoder ########################################

# dims_en [1870, 256, 16] [16, 256, 1870]
# self_numlayer 2
# layer: 0 1870 256
# layer: 0 16 256
# layer: 1 256 16
# layer: 1 256 1870

# dims_en: dim_input , 256 , embedding_num
# dims_de: embedding_num , 256 , dim_imput
# num_layer: 2

class AE_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en)-1
        # print("self_numlayer", self.num_layer)
        
        self.Encoder = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        for index in range(self.num_layer):
            self.Encoder.append(torch.nn.Linear(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(torch.nn.Linear(self.dims_de[index], self.dims_de[index+1]))

    def forward(self, x):
        for index in range(self.num_layer):
            # print("x,adj:", x.shape, adj.shape)
            x = self.Encoder[index].forward(x)
            # x = self.leakyrelu(x)
        h = x
        
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x)
            # x = self.leakyrelu(x)
        x_hat = x    
  
        return h, x_hat



####################################### FULL ########################################
class FULL(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_GAT(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu')) # initialization with pretrain auto_encoder
    
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1) 
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T)) # (num_nodes, num_class)
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p


####################################### FULL NN ########################################
class FULL_NN(torch.nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_model_load_path):
        super(FULL_NN, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        self.AE.load_state_dict(torch.load(pretrain_model_load_path, map_location='cpu')) # initialization with pretrain auto_encoder
  
    def forward(self, x):
        h, x_hat = self.AE.forward(x)
        self.z = torch.nn.functional.normalize(h, p=2, dim=1)# 保留；不然 embedding 方向性不稳定，会严重影响对比学习效果
        # print('x_hat.shape:', x_hat.shape)
        # Dual Self-supervised Module
 
        return self.z, x_hat
 

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T)) # (num_nodes, num_class)
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = torch.nn.functional.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p


####################################### ClusterAssignment ########################################
class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)



# =============================
# 1. 投影头（Projection Heads）
# =============================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)


# 4. 总模块：输入 z, z_t 输出对比损失和融合 embedding Z
# =============================
class MultiModalContrastiveModel(nn.Module):
    def __init__(self, dim_f, dim_t, proj_dim=128, lambda_reg=1e-2):
        super().__init__()
        self.proj_f = ProjectionHead(dim_f, proj_dim)
        self.proj_t = ProjectionHead(dim_t, proj_dim)
        self.lambda_reg = lambda_reg

    def forward(self, z_f, z_t):
        """
        z   : cell feature embedding  (N, dim_f)
        z_t : cell text embedding     (N, dim_t)
        """

        # ---- Step 1: Projection ----
        z_f_proj = self.proj_f(z_f)
        z_t_proj = self.proj_t(z_t)

        # ---- Step 2: L2 normalization ----
        z_f_norm = F.normalize(z_f_proj, dim=1)   # (N, d)
        z_t_norm = F.normalize(z_t_proj, dim=1)   # (N, d)

        # ---- Step 3: contrastive loss ----
        loss_cl = contrastive_loss(z_t_norm, z_f_norm)

        # ---- Step 4: regularization ----
        loss_var = variance_regularization(z_f_norm) + variance_regularization(z_t_norm)

        # ---- Total loss ----
        loss = loss_cl + self.lambda_reg * loss_var

        # ---- Step 5: fused embedding ----
        # Z = 0.5 * (z_f + z_t)   # 最终融合 embedding
        Z = 0.5 * (z_f_norm + z_t_norm)   # 最终融合 embedding

        return loss, Z