# TransGCN: Transformer based Spatial-Temporal Graph Convolutional Networks for Metro Ridership Forecasting
This is a PyTorch implementation of **TransGCN: Transformer based Spatial-Temporal Graph Convolutional Networks for Metro Ridership Forecasting**. 

With the rapid development of urbanization, the accurate forecasting of metro ridership is helpful for people to travel efficiently and avoid contacting with high concentrations of people closely, while this is a challenging task in intelligent transportation systems. Although existing deep learning methods can be modeled in terms of spatial-temporal dimensions and get a roughly satisfying forecasting result, there are still limitations on capturing highly nonlinear spatial dependencies and the periodicity of metro ridership with complex temporal patterns. In this paper, we propose a novel Transformer based Spatial-Temporal Graph Convolutional Networks (TransGCN) for metro ridership forecasting. More specifically, we exploit the graph convolutional networks (GCN) module to deal with the multiple graphs in which we treat each metro station as a node and construct multiple graphs based on station connectivity, regional similarity, and ridership similarity, so the spatial dependence can be captured dynamically in many different aspects. And we also develop a history metro ridership information-based Transformer module with multi-head self-attention mechanisms and a one-dimensional convolution layer that replaces the fully connected layer in the Transformer to capture the continuity and periodicity of time series. Finally, we feed the historical relevant data into the decoder of the Transformer to get the forecasting data by one step that can reduce the cumulative error. Furthermore, evaluation of our TransGCN on two real-world datasets from the Shanghai Metro (SHMetro) and Hangzhou Metro (HZMetro) demonstrates that the proposed network is competitive, and ablation studies confirm the importance of each component of the architecture. 


### Requirements
- python3
- numpy
- yaml
- pytorch
- torch_geometric
### Extract dataset
```
cd data && tar xvf data.tar.gz
```
## Train
- SHMetro
```
python trans_gcn_train.py --config data/model/trans_sh_multi_graph.yaml
```

- HZMetro
```
python trans_gcn_train.py --config data/model/trans_hz_multi_graph.yaml
```




