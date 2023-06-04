import torch.optim as optim
import numpy as np
from torchvision import transforms



# 모델 정의
model = build_model()

# 학습에 사용할 장치 설정
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=2e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 50], gamma=0.2)





train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []



# 학습 시작
max_epoch = 100
for epoch in range(max_epoch):
    print(f"Train Epoch {epoch} start..")
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 순전파 (forward)
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        # 역전파 (backward)
        loss.backward()
        optimizer.step()
        

        # 훈련 중 손실과 정확도 계산
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
        

    # 에포크가 끝날 때마다 PCA를 출력
    if epoch % 3 == 0:  # 에포크 10마다 실행
        model.plot_3d_outputs(train_loader, epoch, 0.3)

        # 메모리 효율화를 위한 변수 삭제
        del outputs, loss, preds



    # 에폭별 훈련 손실과 정확도 계산
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = (train_acc) / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)


    # 검증 중 손실과 정확도 계산
    val_loss = 0.0
    val_acc = 0.0
    model.eval()  # 모델을 검증 모드로 설정

    with torch.no_grad():
        print(f"Val {epoch} start..")
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 순전파 (forward)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 검증 중 손실과 정확도 계산
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

            # 메모리 효율화를 위한 변수 삭제
            del outputs, loss, preds

    # 에폭별 검증 손실과 정확도 계산
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = (val_acc) / len(val_loader.dataset)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    
    # 에폭별 학습 후 learning rate 조절
    scheduler.step()
    
    # 에폭별 결과 출력
    print('[Epoch {}/{}] Train Loss: {:.4f} Train Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
        epoch+1, max_epoch, train_loss, train_acc, val_loss, val_acc))
    #plot
    model.plot_outputs(inputs)
    

    # 에폭별 모델의 가중치를 저장
    torch.save(model.state_dict(), f'/home/ssu36/etc/noise/model18_epoch_{epoch+1}.pth')
