import numpy as np
# import pyxnat
# import nibabel as nb
# import hcp_utils as hcp
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import csv
# import seaborn as sns
# import pingouin as pg
import logging
import igraph as ig
import networkx as nx
import numpy as np
from utils_golem import is_dag 
import re
from numpy import inf
import os 
import time
from tqdm.auto import tqdm , trange
# import statistics
from scipy import stats
# import matlab
# from scipy.io import savemat, loadmat
from numpy.linalg import inv



def extract_conn(x) :
    pre_conn = np.zeros(x.shape)
    for la in range(x.shape[-1]):
        for sub in range(x.shape[-2]) :
            data = x.copy()[:,:,sub , la]
            con = np.zeros(data.shape)
            mask = abs(data) > abs(data.T)
            con = data.copy()*mask
            con = con + np.transpose(con)
            pre_conn[:,:,sub,la] = con           
    return pre_conn

        
def directed_mrtx(data , edges) :   
    mrtx_cn = np.zeros(data.shape)   
    for la in range(data.shape[-1]):
        for sub in range(data.shape[-2]):     
            arr = data.copy()[:,:,sub , la]
            ind_sort = np.dstack(np.unravel_index(np.argsort(arr.ravel()), (arr.shape[0], arr.shape[1]))
                                ).squeeze()[::-1]
            ther = arr[ind_sort[edges][0] , ind_sort[edges][1]]
            conn = (arr > ther)*1
            mrtx_cn[:,:,sub , la] = conn 
    return mrtx_cn


def post_likl_comp(add_ns ,add_s , add_strc ,num_ed  , num_div):
    rg_s = 0
    rg_d = 164
    sc = np.load(add_strc)
    sc = sc[rg_s:rg_d,rg_s:rg_d]
    list_dif = []
    B_ps = np.expand_dims(np.load(add_ns+ ".npy"),3)[rg_s:rg_d,rg_s:rg_d,:,:]
    B_lk = np.expand_dims(np.load(add_s+ ".npy"),3)[rg_s:rg_d,rg_s:rg_d,:,:]
    B_ps2 = extract_conn(B_ps)
    B_lk2 = extract_conn(B_lk)
    for i in range(num_ed) :
        num_div += 1
        edges = int(sc.sum()/num_div)
        print("num_div is : " ,num_div)
        print("conn is : " ,sc.sum()/num_div)
        B_ps3 = directed_mrtx(B_ps2 , edges)
        B_lk3 = directed_mrtx(B_lk2 , edges)
        diff_pos_lk = -((B_ps3[:,:,0,0]-sc)>0).sum() + ((B_lk3[:,:,0,0]-sc)>0).sum()
        print(f"different is : {diff_pos_lk/edges *100}")
        list_dif.append(B_ps3[82: , :82].sum() - B_lk3[82: , :82].sum())
        print('*' * 10)
        print(B_ps3[82: , :82].sum() - B_lk3[82: , :82].sum())
    return B_ps3 , B_lk3 ,list_dif


def pfdr_(add_ns , add_strc ,num_ed  , num_div , th):
    rg_s = 0
    rg_d = 164
    sc = np.load(add_strc)
    sc = sc[rg_s:rg_d,rg_s:rg_d]
    B_ps = np.expand_dims(np.load(add_ns+ ".npy"),3)[rg_s:rg_d,rg_s:rg_d,:,:]
    print("estimated matrix edges are  : " ,(B_ps!=0).sum())
    B_ps2 = extract_conn(B_ps)
    print("estimated matrix edges are  : " ,(B_ps2!=0).sum())
    for i in range(num_ed) :
        num_div += 1
        edges = int(sc.sum()/num_div)
        print("num_div is : " ,num_div)
        print("expected num edges is : " ,edges)
        B_ps3 = directed_mrtx(B_ps2 , edges)
        B_ps3 = np.sum(abs(B_ps3) , 2)
        B_ps3 = (B_ps3 > th) *1
        print("real num edges is : " ,B_ps3.sum())
        
        diff_est_strc = ((B_ps3[:,:,0]-sc)>0).sum() 
        print("num of diff_est_strc  " , diff_est_strc)
        print(f"pfdr percentage is : {diff_est_strc/B_ps3[:,:,0].sum() *100}")
        print('*' * 10)
    return B_ps3 


def connectivity_map(add1 , add2 ,edges , name1 , name2 , th):
    #load structural or estimated matricies and make them ready
    rg_s = 0
    rg_d = 164
    sc = np.load(add1)
    sc = sc[rg_s:rg_d,rg_s:rg_d]
    B_ps = np.expand_dims(np.load(add2+ ".npy"),3)[rg_s:rg_d,rg_s:rg_d,:,:]
    # B_ps = np.transpose(B_ps,(1,2,0,3))
    # print("bps shape " , B_ps.shape)
    B_ps2 = extract_conn(B_ps)
    B_ps3 = directed_mrtx(B_ps2 , edges)
    B_ps3 = np.sum(abs(B_ps3) , 2)
    B_ps3 = (B_ps3 > th) *1
    print("wrong edges count is : " ,  B_ps3[82: , :82].sum())
    print('*' * 10)
    #prepare the prerequisites of the function
    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(16)
    fig.set_figwidth(8)
    imshow_kwargs = {}
    #plot results
    im = axes[0].imshow(sc , **imshow_kwargs)
    plt.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.5)
    im = axes[1].imshow(B_ps3[:,:,0] , **imshow_kwargs)
    plt.colorbar(im, ax=axes[1], orientation="horizontal", fraction=0.5)
    axes[0].set_title(name1)
    axes[1].set_title(name2 + ' for ' + str(B_ps3.sum()) + ' Edges')
    plt.show()









class SyntheticDataset:
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, graph_type, degree, noise_type, B_scale, seed=1 , n_ns = 5):

        self.n = n
        self.n_ns = n_ns
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.noise_type = noise_type
        self.B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                         (B_scale * 0.5, B_scale * 2.0))
        self.rs = [np.random.RandomState(seed + i) for i in range(n_ns)]    # Reproducibility
        self.ix = 1

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):
        """Generate B_bin, B and X."""
        self.B_bin = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                          self.graph_type, self.rs[0])
        self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs[0])
        self.X = SyntheticDataset.simulate_linear_sem(self.B, self.n, self.noise_type, self.ix ,self.rs[0])
        self.X = np.expand_dims(self.X , 2)
        print(self.X.shape)
        for i in range(1,self.n_ns) :
            # self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs[i])
            self.ix = self.ix + int(i) * 1.5
            tmp= np.expand_dims(SyntheticDataset.simulate_linear_sem(self.B,
                                                                     self.n, self.noise_type,
                                                                     self.ix ,self.rs[i] ) , 2)
            self.X = np.concatenate((self.X , tmp) , 2)

        assert is_dag(self.B)

    @staticmethod
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, rs=np.random.RandomState(1)):

        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree, rs)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges, rs=np.random.RandomState(1)):
        B = np.zeros(B_bin.shape)
        S = rs.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = rs.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, ix , rs=np.random.RandomState(1) ):

        def _simulate_single_equation(X, B_i , ix):

            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(loc=ix , scale=1.0 * ix**2 , size=n) 
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=4.0)
                loc = rs.uniform(low=1.0+ix, high=2.0 +ix )
                N_i = rs.normal(scale=scale * ix**2, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0* ix**2 + 0.2*ix, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0* ix**2 + 0.2*ix, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i], ix)

        return X



     
                                             
