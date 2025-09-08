# Lets First build Out Extractor and Load a GCN!! I will use DinoV3 and Arma


# Imports!!

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import utils
import Tools

# Clear CUDA cache before starting
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vits14.to(device)
dinov2_vits14.eval()

# Tranfomrando imagens como no arquiovo original
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


import os

# Directory containing all images
images_dir = "imgs"

# List all image files in the directory (filtering for common image extensions)
image_paths = [
    os.path.join(images_dir, fname)
    for fname in os.listdir(images_dir)
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
]

input_batch = []
for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    input_batch.append(img_tensor)

input_batch = torch.stack(input_batch)
input_batch = input_batch[:20]
input_batch.requires_grad_(True)
input_batch = input_batch.to(device)
features = dinov2_vits14(input_batch)

print("Here!!!")
print(features.shape)

#----------------------------------------------------------

"""
    Here we start the Actual Science!! I will leave the old BallTree Implementation
    as legacy, but what we want is to find a way to cauculate rankings so that we can
    backprop from the GCN though the Ranskings and into the extractor!!

    One way is to make the rankings based on something else rather than euclidean distance.
    cosine similarity is a operation that can be backproped though. 

    But to keep legacy lets attempt to keep the more "rigid" euclidean distance. This is due to 
    cos similarity being a more nuanced metric than E_Distance!!

    Lets first compare various diferent ranking methods (with mAP). (see Rankings.py)


"""



#----------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np


def gumbel_topk_ranking(features: torch.Tensor, k: int = 100,
                        temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    
    # Compute similarity scores
    features_norm = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(features_norm, features_norm.t())
    
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_matrix) + 1e-20) + 1e-20)
    gumbel_scores = (similarity_matrix + gumbel_noise) / temperature
    
    # Apply softmax
    gumbel_weights = F.softmax(gumbel_scores, dim=1)
    
    if hard:
        # Straight-through estimator: hard selection in forward, soft in backward
        _, hard_indices = torch.topk(gumbel_scores, k=k, dim=1, largest=True)
        hard_weights = torch.zeros_like(gumbel_weights)
        hard_weights.scatter_(1, hard_indices, 1.0)
        
        # Straight-through: use hard weights in forward, soft weights in backward
        gumbel_weights = hard_weights + gumbel_weights - gumbel_weights.detach()
    
    return gumbel_weights




# Euclidean topk
k = 2
rks = gumbel_topk_ranking(features, k)



try:
    rks.sum().backward()
except Exception as e:
    print(":(")
    raise e








#----------------

import gc
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import BallTree
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import ARMAConv


#---------------------------------------



# Definicao das Classes(nn.module) das GCNs

class ARMA(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ARMA, self).__init__()

        self.conv1 = ARMAConv(num_features, 16, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(16, num_classes, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)




#------// Classe Geral //---------------------------------


class GCNClassifier():
    def __init__(self, gcn_type, rks, pN, k, pNNeurons, graph_type="knn"):
        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = 0.0001
        self.pNNeurons = pNNeurons
        self.pNEpochs = 200
        self.gcn_type = gcn_type
        self.graph_type = graph_type


    def fit(self, test_index, train_index, features, labels):
        # masks
        print('Creating masks ...')
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.train_size = len(train_index)
        self.test_size = len(test_index)
        self.train_mask = [False for i in range(self.pN)]
        self.val_mask = [False for i in range(self.pN)]
        self.test_mask = [False for i in range(self.pN)]
        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True
        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)
        # labels
        print('Set labels ...')
        y = labels
        self.numberOfClasses = max(y)+1
        self.y = torch.tensor(y).to(self.device)
        # features
        self.x = torch.tensor(features).to(self.device)
        self.pNFeatures = len(features[0])
        # build graph
        self.create_graph()

    def read_ranked_lists_file(self, top_k, file_path):
        print("\tReading file", file_path)
        with open(file_path, 'r') as f:
            return [[int(y) for y in x.strip().split(' ')][:top_k] for x in f.readlines()]

    def create_graph(self):
        print('Making edge list ...')
        self.top_k = self.pK
        # Create simple KNN graph that works with tensor rankings
        edge_index = []
        for img1 in range(self.pN):
            for pos in range(min(self.top_k, self.rks.shape[1])):
                # Use .item() to get integer from tensor element
                img2 = int(self.rks[img1][pos].item())
                edge_index.append([img1, img2])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # convert to torch format
        self.edge_index = edge_index.t().contiguous().to(self.device)

    def predict(self):
        # data object
        print('Loading data object...')
        data = Data(x=self.x.float(),
                    edge_index=self.edge_index,
                    y=self.y,
                    test_mask=self.test_mask,
                    train_mask=self.train_mask,
                    val_mask=self.val_mask)
        # TRAIN MODEL #
        if self.gcn_type == "gcn_arma":
            model = ARMA(self.pNFeatures, self.numberOfClasses).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        print('Training...')
        model.train()
        for epoch in range(self.pNEpochs):
            print("Training epoch: ", epoch)
            optimizer.zero_grad()
            out = model(data)
            data.y = torch.tensor(data.y, dtype=torch.long)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        # MODEL EVAL #
        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.masked_select(pred, data.test_mask.to(self.device))

        return pred.tolist()

# Use the batch features and create dummy labels for testing
labels = [0] * len(features)  # Dummy labels for testing
extrator = "dinov2_vits14"  # Set extractor name
gcn_type = "gcn_arma"  # Use ARMA GCN for testing



n_executions = 1   # Just 1 execution for testing
n_folds = 2     # Just 2 folds for testing

# Split data in folds
folds = utils.fold_split(features, labels, n_folds=n_folds)

pNNeurons = 32 # Get From FrontEnd

# Import the run function  
from Tools import run
    
#----// inferencia do GCN //--------------------------------

# Define number of neighbors (k) for the graph
number_neighbors = 10  # You can adjust this value as needed

# List to store results from all runs
all_results = []

print(f"\nRunning {n_executions} executions with {gcn_type}...\n")

# Run multiple executions and collect results
for i in range(n_executions):
    print(f"Execution {i+1}/{n_executions}")
    results = run(
        features,
        labels,
        folds,
        rks,
        gcn_type,
        pNNeurons,
        GCNClassifier,  # Pass the GCNClassifier class
        number_neighbors  # Pass k parameter
    )
    all_results.append(results)
    
    # Calculate and print accuracy for this execution
    execution_accuracies = [fold_result[2] for fold_result in results]  # Extract accuracies
    avg_accuracy = sum(execution_accuracies) / len(execution_accuracies)
    print(f"Extractor: {extrator} and GCN: {gcn_type}")
    print(f"Execution {i+1} average accuracy: {avg_accuracy:.4f}")

# Calculate overall average accuracy across all executions and folds
all_accuracies = []
for execution_results in all_results:
    for fold_result in execution_results:
        all_accuracies.append(fold_result[2])  # Extract accuracy value

overall_avg_accuracy = sum(all_accuracies) / len(all_accuracies)
print(f"\n{'-'*50}")
print(f"Overall average accuracy across {n_executions} executions and {n_folds} folds: {overall_avg_accuracy:.4f}")
print(f"{'-'*50}\n")

# Print detailed fold results in a table
print(f"{'='*80}")
print(f"{'Detailed Results by Fold':^80}")
print(f"{'='*80}")
print(f"{'Execution':<10}{'Fold':<10}{'Accuracy':<15}")
print(f"{'-'*80}")

# Create a table with all fold results
for exec_idx, execution_results in enumerate(all_results):
    for fold_idx, fold_result in enumerate(execution_results):
        accuracy = fold_result[2]  # Extract accuracy value
        print(f"{exec_idx+1:<10}{fold_idx+1:<10}{accuracy:.4f}{'':15}")
    # Add a separator between executions
    if exec_idx < len(all_results) - 1:
        print(f"{'-'*80}")

print(f"{'='*80}\n")




