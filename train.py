import os
import pandas as pd
import albumentations as A
import torch
import argparse
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from model import faster_rcnn
from dataset import DetectDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (676, 380) # (w, h)


def train_loop(model, optimizer, train_loader):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X = [x.to(DEVICE) for x in X]
        for t in range(len(y)):
            y[t]['boxes'] = y[t]['boxes'].to(DEVICE)
            y[t]['labels'] = y[t]['labels'].to(DEVICE)
        optimizer.zero_grad()
        loss_dict = model(X, y)
        loss = sum([v for k, v in loss_dict.items()])
        loss.backward()
        optimizer.step()
        print(f"train loss --> {loss}")


def val_loop(model, val_loader):
    model.train()
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
            X = [x.to(DEVICE) for x in X]
            for t in range(len(y)):
                y[t]['boxes'] = y[t]['boxes'].to(DEVICE)
                y[t]['labels'] = y[t]['labels'].to(DEVICE)
            loss_dict = model(X, y)
            loss = sum([v for k, v in loss_dict.items()])
            print(f"val loss --> {loss}")


def test_loop(model, test_loader):
    model.train()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X = [x.to(DEVICE) for x in X]
            for t in range(len(y)):
                y[t]['boxes'] = y[t]['boxes'].to(DEVICE)
                y[t]['labels'] = y[t]['labels'].to(DEVICE)
            loss_dict = model(X, y)
            loss = sum([v for k, v in loss_dict.items()])
            print(f"test loss --> {loss}")


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main(args):
    global IMAGE_SIZE
    dataset_path = args["dataset_path"] if args["dataset_path"] else r'..\Detection\data'
    label_csv_path = args["label_csv_path"] if args["label_csv_path"] else r'..\Detection\data\label.csv'
    save_model_path = args["save_model_path"] if args["save_model_path"] else f'../Detection/models'

    image_height = args["image_height"] if args["image_height"] else IMAGE_SIZE[0]
    image_width = args["image_width"] if args["image_width"] else IMAGE_SIZE[1]

    IMAGE_SIZE = (image_width, image_height)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    transform = A.Compose([
        A.Resize(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), ToTensorV2()],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    label_data = pd.read_csv(label_csv_path)

    _, val_data = train_test_split(label_data, test_size=0.2, random_state=42)
    train_data, test_data = train_test_split(_, test_size=0.2, random_state=42)

    train_dataset = DetectDataset(dataset_path, train_data.reset_index(), transform=transform)
    val_dataset = DetectDataset(dataset_path, val_data.reset_index(), transform=transform)
    test_dataset = DetectDataset(dataset_path, test_data.reset_index(), transform=transform)

    epoch = args['epoch'] if args['epoch'] else 2
    lr = args['lr'] if args['lr'] else 0.005
    batch_size = args['batch_size'] if args['batch_size'] else 2
    num_classes = 2

    model = faster_rcnn(num_classes=num_classes).to(DEVICE)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for i in range(epoch):
        train_loop(model, optimizer, train_loader)
        val_loop(model, val_loader)
        test_loop(model, test_loader)
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{save_model_path}/frcnn_{i}_epoch.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        help='Specify path to your dataset')
    parser.add_argument("--label_csv_path", type=str,
                        help='Specify path to your labels')
    parser.add_argument("--save_model_path", type=str,
                        help='Specify path for save models, where models folder will be created')
    parser.add_argument("--epoch", type=int,
                        help='Specify epoch for model training')
    parser.add_argument("--batch_size", type=int,
                        help='Specify batch size for model training')
    parser.add_argument("--lr", type=float,
                        help='Specify learning rate')
    parser.add_argument("--image_height", type=float,
                        help='Specify image height')
    parser.add_argument("--image_width", type=float,
                        help='Specify image width')
    args = parser.parse_args()
    args = vars(args)
    main(args)
