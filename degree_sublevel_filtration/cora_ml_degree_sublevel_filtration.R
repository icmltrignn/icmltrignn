library(igraph)
library(TDA)
library(rlist)
setwd(".../")
cora_ml_adj = as.matrix(read.csv("cora_ml_data.csv",header = FALSE))
cora_ml_net = graph_from_adjacency_matrix(cora_ml_adj, mode = "undirected") # 2708 5278
V(cora_ml_net)$name=c(1:2708)
PDs = list()
maxDimension = 2
maxscale = 10
for (n_id in c(1:2708)) {
  print(n_id)
  # 3-hop neighborhood #
  tmp_induced_vertices = ego(cora_ml_net, order = 3, nodes = n_id)#neighbors(cora_ml_net, v = n_id)
  # corresponding induced graph #
  tmp_induced_subgraph = induced_subgraph(cora_ml_net, vids = tmp_induced_vertices[[1]])
  # node features - degree #
  F.values = apply(as_adjacency_matrix(tmp_induced_subgraph), 1, sum)
  # cut point as 10 #
  F.values[which(F.values>10)] = maxscale
  
  # for maxDimension=3 below means we are adding 0,1,2 simplices (nodes,edges,triangles) to our complex
  cmplx = cliques(as.undirected(tmp_induced_subgraph), min = 0, max = maxDimension)
  # use sublevel=T for sublevel, sublevel=F for superlevel filtration
  FltRips = funFiltration(FUNvalues = F.values,
                  cmplx = cmplx,
                  sublevel = T) # Construct filtration using F.values
  
  persistenceDiagram = filtrationDiag(filtration = FltRips, maxdimension = maxDimension)$diagram
  persistenceDiagram[persistenceDiagram[,3]==Inf,3]=maxscale
  #PD1data = persistenceDiagram[persistenceDiagram[,1]==0,2:3]
  #PD1data[PD1data[,2]==Inf,2]=maxscale
  
  PDs[[n_id]] = persistenceDiagram
}

# save list #
list.save(PDs, 'PDs_list.rdata')
#list.load('PDs_list.rdata')


# wasserstein distance matrix #
wasserstein_distance_mat = matrix(data = 0, nrow = 2708, ncol = 2708)
for (i in c(1:2708)) {
  print(i)
  tmp_induced_vertices = as.numeric(ego(cora_ml_net, order = 3, nodes = i)[[1]])
  for (j in tmp_induced_vertices) {
    wasserstein_distance_mat[i,j] = wasserstein(PDs[[i]], PDs[[j]], dimension = 0)
  }
}

write.csv(wasserstein_distance_mat, file = "cora_ml_wasserstein_distance_mat.csv")
