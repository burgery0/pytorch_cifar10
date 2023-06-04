import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import cm
import numpy as np
from sklearn.cluster import KMeans
import random



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self._initialize_weights()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.num_classes = num_classes

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.outputs = {'conv1': [], 'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}

        self.cam = None
        self.register_hooks()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def register_hooks(self):
        self.conv1.register_forward_hook(self.hook_fn('conv1'))
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], 1):
            layer[0].conv1.register_forward_hook(self.hook_fn(f'layer{i}'))

    def hook_fn(self, layer):
        def hook(module, input, output):
            if len(self.outputs[layer]) < 1:
                self.outputs[layer].append(output)

        return hook

    def plot_3d_outputs(self, dataloader, epoch, sample_percentage):
        device = next(self.parameters()).device
        max_values = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                self.forward(inputs)
                output = self.cam.detach()

                max_value, _ = torch.max(output.view(output.shape[0], output.shape[1], -1), dim=2)
                max_values.append(max_value.cpu())

                labels_list.append(labels)

        max_values = torch.cat(max_values)
        labels = torch.cat(labels_list)

        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if epoch % 3 == 0:
            kmeans = KMeans(n_clusters=self.num_classes)
            clusters = kmeans.fit_predict(max_values)

            pca = PCA(n_components=3)
            max_values_pca = pca.fit_transform(max_values)
            cluster_centroids_pca = pca.transform(kmeans.cluster_centers_)

            colormap = cm.get_cmap('tab10', self.num_classes)

            angles = [0,60,90]
            for angle in angles:
                fig = plt.figure(figsize=(13, 13))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=0, azim=angle)
                ax.dist = 6
                for i in range(self.num_classes):
                    indices = np.where(clusters == i)[0]
                    subset_size = int(len(indices) * sample_percentage)
                    sampled_indices = random.sample(list(indices), subset_size)
                    color = colormap(i)
                    ax.scatter(max_values_pca[sampled_indices, 0], max_values_pca[sampled_indices, 1],
                               max_values_pca[sampled_indices, 2], color=color, label=classes[i])
                    ax.scatter(cluster_centroids_pca[i, 0], cluster_centroids_pca[i, 1], cluster_centroids_pca[i, 2],
                               color=color, s=250, edgecolor='k', marker='*')

                ax.legend(loc="upper right", title="Classes")
                plt.title(f'PCA at epoch {epoch}, azim={angle}')
                plt.show()

    def plot_outputs(self, labels):
        fig, axs = plt.subplots(1, 5, figsize=(7, 7))
        axs = axs.flatten()
        for i, (layer_name, outputs) in enumerate(self.outputs.items()):
            output = outputs[-1]
            output = output.detach()  
            if torch.cuda.is_available():
                output = output.cpu() 

            if output.dim() == 4:  # output of Conv layers
                axs[i].imshow(output[0, 0, :, :])
                axs[i].set_title(f'{layer_name}')

        plt.tight_layout()
        plt.show()

        for layer_name in self.outputs:
          self.outputs[layer_name].clear()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self.layer4_output = out.detach().cpu()
        self.cam = out.clone()
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def build_model(**kwargs):
    return ResNet18(**kwargs)
