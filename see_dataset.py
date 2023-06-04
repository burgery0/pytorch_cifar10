import matplotlib.pyplot as plt
num_images_per_class = 5

class_counts = {class_name: 0 for class_name in classes}
for data, labels in train_loader:
    for i, label in enumerate(labels):
        class_name = classes[label]
        
        if class_counts[class_name] < num_images_per_class:
            plt.figure()
            plt.imshow(transforms.ToPILImage()(data[i]))
            plt.title(f'Class: {class_name}')
            plt.show()

            class_counts[class_name] += 1
    if all(count >= num_images_per_class for count in class_counts.values()):
        break
