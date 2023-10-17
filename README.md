# Bayesian Structure learning using DAGMA where you have some prior knowledge about the existence of some edges powers in the Graph

  Here the Hadamard product of the inverse of each component in the prior knowledge Matrix $B$ ($B^{*}$) and adjacency matrix $W$ is replaced with the original sparsity term in 
  DAGMA[1].
  the new sparsity term would be :

   $$Loss_{sparsity} = |B^{*}\circ W|$$
  
This repository is based on the original implementation codes of DAGMA:
the link to the original implementation: https://github.com/kevinsbello/dagma#readme


# Citation
[1] Bello K., Aragam B., Ravikumar P. (2022). [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization][dagma]. [NeurIPS'22](https://nips.cc/Conferences/2022/). 
