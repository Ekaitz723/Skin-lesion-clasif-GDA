import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
# import seaborn as sns


# Configurar el logger
def setup_logger(rank, log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank == 0:  # Solo configurar el archivo de log en la GPU 0
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3, depth=18, input_size=512, dropout_rate=0.5):
        super(CustomResNet, self).__init__()
        self.input_size = input_size
        if depth == 18:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif depth == 34:
            base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif depth == 50:
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif depth == 101:
            base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif depth == 152:
            base_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet depth. Choose from 18, 34, 50, 101, 152.")
        
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]['image']
        label = self.data[index]['label']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=1.5, contrast=2, saturation=1.5, hue=0.2),  # Ajusta el brillo, contraste, saturación y matiz
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_dataloader(data, batch_size, prueba):
    train_dataset = CustomDataset(data['train'], transform=transform)
    test_dataset = CustomDataset(data['test'], transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_dataloader, test_dataloader

def train(model, device, dataloader, criterion, optimizer):
    model.train()
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_metrics(model, device, dataloader, exp_path, num_classes=2, epoch=-1):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    all_labels = [int(label) for label in all_labels]
    all_predictions = [int(pred) for pred in all_predictions]
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.xticks(ticks=np.arange(num_classes) + 0.5, labels=[f"Class {i}" for i in range(num_classes)])
    plt.yticks(ticks=np.arange(num_classes) + 0.5, labels=[f"Class {i}" for i in range(num_classes)])
    plt.tight_layout()

    cm_name = f"{exp_path}/conf_matrix/ep_{epoch}.jpg"
    plt.savefig(cm_name)
    plt.close()
    
    return accuracy, precision, recall, cm_name

def calculate_metrics_by_threshold(labels, scores, num_classes=3):
    thresholds = np.linspace(0, 1, num=100)
    precisions, recalls, f1_scores = [], [], []

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        if precision or recall:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return thresholds, precisions, recalls, f1_scores

def train_loop(rank, world_size, dataset, epochs=21, exp_path="compGDA", val_ep=5, batch_size=45, depth=18, input_size=512, dropout_rate=0.2, lr=0.0005, wd=0.01, prueba=False, seed=42, resume_from=None):
    print(f"entra en {rank}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Configurar el logger
    log_file = f"{exp_path}/training_log.txt"
    logger = setup_logger(rank, log_file)

    resultados_train = []

    train_dataloader, val_dataloader = get_image_dataloader(data=dataset, batch_size=batch_size, prueba=prueba)

    model = CustomResNet(num_classes=3, depth=depth, input_size=input_size, dropout_rate=dropout_rate).to(device)
    model = DDP(model, device_ids=[rank])
    print(f"model en {model.device}")

    # Cargar el modelo desde un checkpoint si resume_from no es None
    if resume_from is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(resume_from, map_location=map_location))
        logger.info(f"Modelo cargado desde {resume_from}")
        
    
    logger.info(f"Model is on {next(model.parameters()).device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    def calc_f1(p, r):
        return (2 * p * r) / (p + r) if (p != 0 and r != 0) else 0

    best_f1 = -1
    best_datadict = {}

    for epoch in range(epochs):
        logger.info(f"epoch {epoch}")
        train(model, device, train_dataloader, criterion, optimizer)
        if epoch % val_ep == 0 and epoch!=0:
            accuracy, precision, recall, cm_name = evaluate_metrics(model, device, val_dataloader, exp_path, num_classes=2, epoch=epoch)
            logger.info(f"\tAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
            data_dict = {'epoch': epoch, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'conf_matrix_path': cm_name}
            resultados_train.append(data_dict)
            cur_f1 = calc_f1(precision, recall)
            if best_f1 <= cur_f1:
                best_f1 = cur_f1
                best_datadict = data_dict
                if rank == 0:
                    torch.save(model.state_dict(), f'{exp_path}/resnet_weights_best.pth')
                    logger.info(f"\tSaving best with f1 {cur_f1:.2f}")

    if rank == 0:
        torch.save(model.state_dict(), f'{exp_path}/resnet_weights_last.pth')
        logger.info(f"Saving last with f1 {cur_f1:.2f}")

    dist.destroy_process_group()

    return resultados_train, best_datadict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--exp_path", type=str, default="resultados_clasif/exp_default")
    parser.add_argument("--val_ep", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=75)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--prueba", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--resume_from", type=str, default=None, help="Ruta para cargar el modelo desde un checkpoint")
    args = parser.parse_args()

    with open(args.dataset_path, "r") as archivo:
        dataset_isic = json.load(archivo)

    nombre_carpeta = f"{args.exp_path}/conf_matrix"
    if not os.path.exists(nombre_carpeta) and int(os.environ["LOCAL_RANK"]) == 0:
        os.makedirs(nombre_carpeta)
        logger = setup_logger(0, f"{args.exp_path}/training_log.txt")
        logger.info(f"Carpeta '{nombre_carpeta}' creada correctamente.\nHP: \nepochs: {args.epochs} \nexp_path: {args.exp_path} \nval_ep: {args.val_ep} \nbatch_size: {args.batch_size} \ndepth: {args.depth} \ninput_size: {args.input_size} \ndropout_rate: {args.dropout_rate} \nlr: {args.lr} \nwd: {args.wd} \nprueba: {args.prueba} \nseed: {args.seed} \ndataset_path: {args.dataset_path} \nnum_gpus: {args.num_gpus} \nresume_from: {args.resume_from}")

    print(f"rank: {int(os.environ['LOCAL_RANK'])}")

    resultado = train_loop(int(os.environ["LOCAL_RANK"]),args.num_gpus, dataset_isic, args.epochs, args.exp_path, args.val_ep, args.batch_size, args.depth, args.input_size, args.dropout_rate, args.lr, args.wd, args.prueba, args.seed, args.resume_from)
    print(resultado)

