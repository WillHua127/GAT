from utils import *
import torch


adj, features, labels = load_data('cora')
dense_adj = adj.to_dense()
torch.save(dense_adj, "cora_dense_adj.pt")
torch.save(features, "cora_features.pt")
torch.save(labels, "cora_labels.pt")

adj, features, labels = load_data('citeseer')
dense_adj = adj.to_dense()
torch.save(dense_adj, "citeseer_dense_adj.pt")
torch.save(features, "citeseer_features.pt")
torch.save(labels, "citeseer_labels.pt")

adj, features, labels = load_data('pubmed')
dense_adj = adj.to_dense()
torch.save(dense_adj, "pubmed_dense_adj.pt")
torch.save(features, "pubmed_features.pt")
torch.save(labels, "pubmed_labels.pt")

#adj, features, labels = load_data('nell.0.1')
#dense_adj = adj.to_dense()
#torch.save(dense_adj, "nell.0.1_dense_adj.pt")
#torch.save(features, "nell.0.1_features.pt")
#torch.save(labels, "nell.0.1_labels.pt")

#adj, features, labels = load_data('nell.0.01')
#dense_adj = adj.to_dense()
#torch.save(dense_adj, "nell.0.01_dense_adj.pt")
#torch.save(features, "nell.0.01_features.pt")
#torch.save(labels, "nell.0.01_labels.pt")

#adj, features, labels = load_data('nell.0.001')
#dense_adj = adj.to_dense()
#torch.save(dense_adj, "nell.0.001_dense_adj.pt")
#torch.save(features, "nell.0.001_features.pt")
#torch.save(labels, "nell.0.001_labels.pt")



del adj, dense_adj, features, labels
