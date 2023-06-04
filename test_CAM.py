import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(torch.load('/home/model18_epoch_152.pth', map_location=device)) 
model.to(device)
model.eval()

class_count = [0]*10
fig, axs = plt.subplots(10, 4, figsize=(10, 40)) 



with torch.no_grad():
    correct = 0
    total = 0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    mean = np.array([0.4914, 0.4822, 0.4465])    
    std = np.array([0.2023, 0.1994, 0.2010])
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        loss += criterion(outputs, labels).item()
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        for i in range(images.size(0)):
            single_image = images[i]
            single_label = labels[i]
            single_pred = preds[i]
            
        # Calculating CAM
        weight_softmax_params = list(model.linear.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        weights_map = weight_softmax[preds[0]].dot(model.cam[0].cpu().data.numpy().reshape((model.cam.shape[1], -1)))
        weights_map = weights_map.reshape(model.cam[0].shape[1], model.cam[0].shape[2])
        weights_map -= weights_map.min()
        weights_map /= weights_map.max()
        weights_map_resized = cv2.resize(weights_map, (images.shape[2], images.shape[3]))
        weights_map_color = cv2.applyColorMap(np.uint8(255 * weights_map_resized), cv2.COLORMAP_JET)

        
        # Denormalize the image
        image_np = single_image.cpu().numpy().transpose((1, 2, 0))
        image_np = std * image_np + mean  # apply denormalization
        image_np = np.clip(image_np, 0, 1)  # ensure the values are within [0,1]
        image_np = np.uint8(255 * image_np)
    
        cam_img = cv2.addWeighted(image_np, 0.5, weights_map_color, 0.3, 0)


        # Plot
        if class_count[single_label.item()] < 1:
            axs[single_label.item(), 0].imshow(transforms.ToPILImage()(single_image.cpu()))
            axs[single_label.item(), 0].axis('off')
            axs[single_label.item(), 0].set_title(f'Input Image - Class: {classes[single_label.item()]}')

            axs[single_label.item(), 1].imshow(weights_map_resized, cmap='jet')
            axs[single_label.item(), 1].axis('off')
            axs[single_label.item(), 1].set_title('CAM')

            axs[single_label.item(), 2].imshow(cam_img)
            axs[single_label.item(), 2].axis('off')
            axs[single_label.item(), 2].set_title('CAM Overlay')

            axs[single_label.item(), 3].text(0.5, 0.5, f'Model Predict : {classes[single_pred.item()]}', ha='center', va='center', size=20)
            axs[single_label.item(), 3].axis('off')

            class_count[single_label.item()] += 1

        if all(count == 1 for count in class_count):
            break

    print('Accuracy : %d %%' % (100 * correct / total))
    print('Loss:', loss / total)

plt.tight_layout()
plt.show()
