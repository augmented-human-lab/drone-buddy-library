import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F


class SiameseModel(nn.Module):

    # potential configurable arguments
    # channels: int, n_classes: int, dim_sizes: List[int], kernel_size: int, stride: int, padding: int, **kwargs
    # pretrained model and weights
    def __init__(self, base_model, base_model_weights):
        super(SiameseModel, self).__init__()
        self.emb_size = 20
        self.weights = base_model_weights
        self.siamese = base_model(weights=self.weights)
        train_nodes, eval_nodes = get_graph_node_names(self.siamese)
        self.classifier = nn.Linear(self.siamese.get_submodule(train_nodes[-1]).out_features, 1)


    def get_embedding(self, x):
        return self.siamese(x)

    def forward(self, img1, img2):
        preprocess = self.weights.transforms()
        x1 = preprocess(img1)
        x2 = preprocess(img2)
        out1 = self.siamese(x1)
        out2 = self.siamese(x2)

        diff = torch.abs(torch.sigmoid(out1) - torch.sigmoid(out2))

        output = self.classifier(diff)
        print("Similarity score: ", output)
        return output

    def forward_difference(self, img1, img2):
        preprocess = self.weights.transforms()
        x1 = preprocess(img1)
        x2 = preprocess(img2)
        out1 = self.siamese(x1)
        out2 = self.siamese(x2)
        diff = torch.abs(torch.sigmoid(out1) - torch.sigmoid(out2))
        return diff

    def forward_difference_tsne(self, img1, img2):
        preprocess = self.weights.transforms()
        x1 = preprocess(img1)
        x2 = preprocess(img2)
        out1 = self.siamese(x1)
        out2 = self.siamese(x2)

        # get tsne difference
        tensor_1_np = out1.detach().cpu().numpy()
        tensor_2_np = out1.detach().cpu().numpy()

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_1_values = tsne.fit_transform(tensor_1_np)
        tsne_2_values = tsne.fit_transform(tensor_2_np)

        # tsne_values now contains the 2D t-SNE representation of the tensor
        print(tsne_1_values)
        print(tsne_2_values)
        diff = torch.abs(torch.sigmoid(out1) - torch.sigmoid(out2))
        sigmoid_abs_diff = torch.sigmoid(torch.abs(out1 - out2))
        abs_diff = torch.abs(tensor_1_np - tensor_2_np)
        return abs_diff
