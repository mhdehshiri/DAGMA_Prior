# Bayesian Structure learning using DAGMA where you have some prior knowledge about the existence of some edges powers in the Graph

  Here the Hadamard product of the inverse of each component in the prior knowledge Matrix $B$ ($B^{*}$) and adjacency matrix $W$ is replaced with the original sparsity term in 
  DAGMA(https://proceedings.neurips.cc/paper_files/paper/2022/hash/36e2967f87c3362e37cf988781a887ad-Abstract-Conference.html).
  the new sparsity term would be :

   $$Loss_{sparsity} = |B^{*}\circ W|$$
  
This repository is based on the original implementation codes of DAGMA:
the link to the original implementation: https://github.com/kevinsbello/dagma#readme

