if torch.cuda.is_available():
    train_acc_list = [acc.cpu() for acc in train_acc_list]
    val_acc_list = [acc.cpu() for acc in val_acc_list]

plt.figure()
plt.plot(range(56), train_loss_list[1:], label='Train Loss')
plt.plot(range(56), val_loss_list[1:], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()


plt.figure()
plt.plot(range(57), [acc * 100 for acc in train_acc_list], label='Train Accuracy')
plt.plot(range(57), [acc * 100 for acc in val_acc_list], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
