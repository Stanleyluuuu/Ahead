import pdb
import numpy as np
import glob
import FlowCal
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 8787
N = 30000

class MyDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.data = train_x
        self.label = train_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trans = transforms.Compose([transforms.ToTensor()])
        data = trans(self.data[idx])
        label = self.label[idx]

        return data, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
                                   nn.Linear(N * 31, 200),
                                   nn.BatchNorm1d(200),
                                   nn.ReLU(),
                                   nn.Linear(200, 50),
                                   nn.BatchNorm1d(50),
                                   nn.ReLU(),
                                   nn.Linear(50, 2),
                                   )
    
    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=1))

def train_NN(train_x, test_x, train_y, test_y):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Currently Using Device {device}")
    config = {
              "BatchSize": 2,
              "Epochs": 100,
              "LR": 1e-3,
              "Early_stop": 30,
              }
    model = NeuralNetwork().to(device)

    optimizer = Adam(model.parameters(), lr=config["LR"])
    scheduler = lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=15)
    w = torch.tensor([0.3, 0.7]).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    
    trainSet, testSet = MyDataset(train_x, train_y), MyDataset(test_x, test_y)
    
    trainLoader = DataLoader(dataset=trainSet, batch_size=config["BatchSize"], shuffle=True, num_workers=4, pin_memory=True)
    testLoader = DataLoader(dataset=testSet, batch_size=config["BatchSize"], shuffle=True, num_workers=4, pin_memory=True)

    writer = SummaryWriter()
    best_performace = float("inf")
    best_result = None
    for epoch in range(config["Epochs"]):
        # training mode
        model.train()
        count, running_loss, running_acc = 0, 0, 0
        for data in trainLoader:
        # for data in tqdm(trainLoader, desc="Running train loader"):
            optimizer.zero_grad()

            input, label = data[0].to(device), data[1].to(device)
            count += input.shape[0]
            output = model(input)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, axis=1)[1]
            acc = (predict == label).sum()

            running_loss += loss
            running_acc += acc
        train_loss, train_acc = running_loss / count, running_acc / count * 100
        # evaluation mode
        scheduler.step()
        model.eval()
        count, running_loss, running_acc = 0, 0, 0
        result = []
        with torch.no_grad():
            for data in testLoader:
                input, label = data[0].to(device), data[1].to(device)
                count += input.shape[0]
                output = model(input)
                
                loss = criterion(output, label)
                predict = torch.max(output, axis=1)[1]
                acc = (predict == label).sum()
                result.append(predict)
                running_loss += loss
                running_acc += acc
        test_loss, test_acc = running_loss / count, running_acc / count * 100

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        if running_loss / count < best_performace:
            best_result = result
            best_performace = running_loss / count
            best_acc = running_acc / count * 100
            early_stop_count = 0
        else: early_stop_count += 1
        print("{} / {}: training loss: {:.3f}, training acc: {:.3f}, testing loss: {:.3f}, testing acc: {:.3f}".format(epoch + 1, config["Epochs"], train_loss, train_acc, test_loss, test_acc))
        if early_stop_count > config["Early_stop"]: break
    print(best_result)
    print(test_y)
    print("Best performace of NN is {:.3f}, best acc = {:.3f}".format(best_performace, best_acc))


def set_seed():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)

def get_feature():
    df = pd.read_excel("EU_marker_channel_mapping.xlsx")
    print(f"There are {len(df)} kind of features")
    df = df.loc[df["use"] == 1]
    print(f"We only use {len(df)} kind of features")
    features = list(df["PxN(channel)"])
    
    return features

def get_label():
    df = pd.read_excel("EU_label.xlsx")
    df = df.loc[df["label"] == "Healthy"]
    health_id = list(df["file_flow_id"])

    return health_id

def padding(array, target_x, target_y):
    h, w = array.shape[0], array.shape[1]
    pad_x = target_x - h
    pad_y = target_y - w

    return np.pad(array, pad_width=((0, pad_x), (0, pad_y)), mode='constant')

def read_data(features, health_id):
    data_dir = "raw_fcs/*/*.fcs"
    fcs_data = glob.glob(data_dir)
    procressed_data, labels = [], []
    for data in fcs_data:
        d = FlowCal.io.FCSData(data)
        label = 1 if data.split("/")[1] in health_id else 0
        df = pd.DataFrame(d)
        df.columns = d.channels
        array = df.astype('f4')[features].sample(n=min(len(df.index), N), random_state=SEED).to_numpy()
        procressed_data.append(padding(array, N, 31))
        labels.append(label)
    
    return procressed_data, labels

def do_linaer_models(train_x_2d, test_x_2d, train_y, test_y):
    weight = {0: 3, 1: 7}
    model_zoo = {
                 "LogisticRegression": LogisticRegression(class_weight=weight),
                 "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight=weight),
                 "Random Forest": RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight=weight),
                 "K Neighbors": KNeighborsClassifier(n_neighbors=9, weights="distance"),
                 "Linear SVM": SVC(kernel="linear", C=0.025, class_weight=weight),
                 "RBF SVM": SVC(gamma=2, C=1, class_weight=weight),
                 "MLP": MLPClassifier(alpha=1, max_iter=1000, learning_rate="adaptive", early_stopping=True),
                 "XGBoost": XGBClassifier(base_score=0.5,
                                          booster='gbtree',
                                          n_estimators=1000,
                                          early_stopping_rounds=20,
                                          objective='reg:linear',
                                          max_depth=5,
                                          learning_rate=0.0001,
                                          ),
                 }
    
    for k, v in model_zoo.items():
        if k == "XGBoost": v.fit(train_x_2d, train_y, eval_set=[(train_x_2d, train_y), (test_x_2d, test_y)], verbose=100)
        else: v.fit(train_x_2d, train_y)
        preds = v.predict(test_x_2d)
        print(preds)
        print("Accuracy of {} = {:.3f}".format(k, sum(preds==test_y) / 8))

if __name__ == "__main__":
    set_seed()
    features = get_feature()
    health_id = get_label()

    data, labels = read_data(features, health_id)
    train_x, test_x, train_y, test_y = train_test_split(np.array(data), np.array(labels), test_size=0.2, shuffle=True, stratify=np.array(labels))
    train_x_2d, test_x_2d = train_x.reshape((32, -1)), test_x.reshape((8, -1))
    # do_linaer_models(train_x_2d, test_x_2d, train_y, test_y)
    
    train_NN(train_x, test_x, train_y, test_y)