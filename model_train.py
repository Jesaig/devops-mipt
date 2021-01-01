import mlflow

import matplotlib.pyplot as plt

import numpy as np

import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision 
from torchvision import datasets
from torchvision import transforms

# def plot_learning_curves(history):
#     '''
#     Функция для обучения модели и вывода лосса и метрики во время обучения.

#     :param history: (dict)
#         accuracy и loss на обучении и валидации
#     '''
#     # sns.set_style(style='whitegrid')
#     fig = plt.figure(figsize=(20, 7))
    
#     plt.rcParams['axes.facecolor'] = 'white'

#     plt.subplot(1,2,1)
#     plt.title('Loss', fontsize=15)
#     plt.plot(history['loss']['train'], label='train')
#     plt.plot(history['loss']['val'], label='val')
#     plt.ylabel('loss', fontsize=15)
#     plt.xlabel('epoch', fontsize=15)
#     plt.legend()

#     plt.subplot(1,2,2)
#     plt.title('Accuracy', fontsize=15)
#     plt.plot(history['acc']['train'], label='train')
#     plt.plot(history['acc']['val'], label='val')
#     plt.ylabel('accuracy', fontsize=15)
#     plt.xlabel('epoch', fontsize=15)
#     plt.legend()
#     plt.show()


def train(
    model, 
    criterion,
    optimizer, 
    train_batch_gen,
    val_batch_gen,
    scheduler=None,
    num_epochs=10
):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.

    :param model: обучаемая модель
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :param train_batch_gen: генератор батчей для обучения
    :param val_batch_gen: генератор батчей для валидации
    :param num_epochs: количество эпох

    :return: обученная модель
    :return: (dict) accuracy и loss на обучении и валидации ("история" обучения)
    '''

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True) 

        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in train_batch_gen:
            # Обучаемся на батче (одна "итерация" обучения нейросети)

            logits = model(X_batch)

            loss = criterion(logits, y_batch.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # Подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_gen)
        train_acc /= len(train_batch_gen)

        mlflow.log_metric('Loss Train', train_loss, step=epoch)
        mlflow.log_metric('Accuracy Train', train_acc, step=epoch)

        # Устанавливаем поведение dropout / batch_norm в режим тестирования
        model.train(False) 

        # Полный проход по валидации    
        with torch.no_grad():
            for X_batch, y_batch in val_batch_gen:
                X_batch = X_batch
                y_batch = y_batch

                logits = model(X_batch)
                loss = criterion(logits, y_batch.long())
                val_loss += np.sum(loss.detach().cpu().numpy())
                y_pred = logits.max(1)[1].detach().cpu().numpy()
                val_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # Подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_gen)
        val_acc /= len(val_batch_gen)

        mlflow.log_metric('Loss Test', val_loss, step=epoch)
        mlflow.log_metric('Accuracy Test', val_acc, step=epoch)

        if scheduler is not None:
            scheduler.step(val_loss)
        
        # clear_output()

        # # Печатаем результаты после каждой эпохи
        # print("Epoch {} of {} took {:.3f}s".format(
        #     epoch + 1, num_epochs, time.time() - start_time))
        # print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        # print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        # print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        # print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))
        
        # plot_learning_curves(history)
        
    return model


class BuiltInFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(BuiltInFolder, self).__init__(*args, **kwargs)
        self.images = []
        for i in range(len(self)):
            self.images.append(super(BuiltInFolder, self).__getitem__(i))

    def __getitem__(self, ind):
        return self.images[ind]


def batches_generator(
    size,
    augmentation=False,
    data_dir='data/',
    batch_size=64
): 

    transform_train = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    ])

    if augmentation is True:
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])

    transform_val = transforms.Compose([
        transforms.Resize(size),                                  
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),                                  
        transforms.ToTensor(),
    ])


    train_dataset = BuiltInFolder(
        os.path.join(data_dir, 'train'), 
        transform=transform_train
    )

    val_dataset = BuiltInFolder(
        os.path.join(data_dir, 'val'), 
        transform=transform_val
    )

    test_dataset = BuiltInFolder(
        os.path.join(data_dir, 'test'), 
        transform=transform_val
    )

    train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_batch_gen = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_batch_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_batch_gen, val_batch_gen, test_batch_gen)



	
mlflow.set_tag('model_type', 'convolutional_nn')

device = 'cpu'
	 
num_epochs = 10
mlflow.log_param('num_epochs', num_epochs)

model_conv = nn.Sequential(
	nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
	nn.ReLU(),

	nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
	nn.ReLU(),

	nn.Conv2d(64, 128, kernel_size=3, bias=False),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.MaxPool2d(2),
	nn.Dropout2d(0.5),

	nn.Conv2d(128, 256, kernel_size=3, bias=False),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.MaxPool2d(2),

	nn.Conv2d(256, 256, kernel_size=3, bias=False),
	nn.MaxPool2d(2),
	nn.BatchNorm2d(256),
	nn.ReLU(),

	nn.AvgPool2d(5),

	nn.Flatten(),

	nn.Dropout(0.5),
	nn.Linear(256, 700),
	nn.ReLU(),

	nn.Linear(700, 50)
)

criterion_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.3, verbose=True, patience=5)

train_batch_gen, val_batch_gen, test_batch_gen = batches_generator(
    [240, 240],
    augmentation=True
)
	 
model_conv = train(
    model_conv,
    criterion_loss,
    optimizer,
    train_batch_gen,
    val_batch_gen,
    scheduler,
    num_epochs
)
	 
mlflow.pytorch.save_model(model_conv, 'cnnmodel.cbm')
mlflow.log_artifact('cnnmodel.cbm')