def plot_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            plt.figure()
            plt.title(str(m))
            plt.hist(m.weight.data.cpu().numpy().flatten(), bins=100)
            plt.show()

model = build_model()
plot_weights(model)
