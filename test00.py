import torch
from torch_geometric.data import Data, HeteroData


x=torch.tensor(
    [[-1, 1],
    [0, 1],
    [2, 3]]
)
edge_index=torch.tensor(
    [[0, 1],
    [1, 2]]
)
y=torch.tensor([0, 0, 1])

data = Data (x=x, 
             edge_index=edge_index,
             y=y,
             )

#print(data)

data.validate(raise_on_error=True)

print(data.x)
print(data.x.shape)
print(data.edge_index)
print(data.edge_index.shape)
print(data.y)
print(data.y.shape)
print(data.num_nodes)

data.node_id=torch.tensor([100, 200, 300])
data.node_names=['Alice', 'Bob', 'Charlie']
data.edge_id=torch.tensor([1000, 2000])
data.edge_names=['Edge1', 'Edge2']

print(data.node_id)
print(data.node_names)
print(data.edge_id)
print(data.edge_names)