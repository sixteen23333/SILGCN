# The code for paper "SILGCN: An Efficient Threat Detection Method Based on Semantic Inductive Learning".
# 1、 Document Description
Main. py: Main file, complete training and evaluation process
Train.by: Train the improved GCN
Model. py: Improved GCN Model
Evaluate.py: Testing and Evaluation
Build_graph.py: Building a semantic association graph
Build_dataset.cy: Data loading and sampling
Preprocess. py: Data Preprocessing and Label Encoding
# 2、 How to operate
(1) Data preparation: Download the original PCAP, split it into quintuples, convert it into hexadecimal sequence data, prepare the data file and label file, and refer to the formats data.txt and label.txt.
(2) Replace your own dataset address, adjust parameter settings, and run the main file
