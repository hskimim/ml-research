import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import calibration
calib = calibration.CalibrationLoss() # calculate uniform-mean calibration error

def entropy(prob) :
    assert prob.dim() == 2
    eps = 1e-5
    return torch.sum(-(prob+eps) * torch.log(prob+eps), dim=1)

class ERL(nn.Module):

    def __init__(self, eps=1e-5, beta=0.1):
        super().__init__()
        self._eps = eps
        self._beta = beta

    def forward(self, outputs, true):
        assert outputs.shape[0] == outputs.shape[0]
        assert outputs.dim() == 2

        pred = torch.softmax(outputs, dim=1) + self._eps
        prob = torch.gather(input=pred, dim=1, index=true.unsqueeze(1)).squeeze()
        ce = -1 * torch.log(prob)
        penalty = entropy(pred)
        return torch.mean(ce) -self._beta * torch.mean(penalty)

# Copied from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module) : 
    
    def __init__(self, eps=1e-5, gamma=3) : 
        super().__init__()
        self._eps = eps
        self._gamma = gamma

    def forward(self, outputs, true) :         
        assert outputs.shape[0] == outputs.shape[0]
        assert outputs.dim() == 2
        
        pred = torch.softmax(outputs, dim=1)+self._eps
        prob = torch.gather(input=pred, dim=1, index=true.unsqueeze(1)).squeeze()
        focal_loss = -1 * torch.pow(1-prob, self._gamma) * torch.log(prob)
            
        return torch.mean(focal_loss)

class LabelSmoothingLoss(nn.Module):
    # Copied from https://programmersought.com/article/27102847986/
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
class Experiments : 
    epoch = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, writer, modelname, today_ymd) : 
        
        self._writer = writer
        self._modelname = modelname
        self._today_ymd = today_ymd
        
        self._train_loss_memory = list()
        self._valid_loss_memory = list()
        
        self._train_acc_memory = list()
        self._valid_acc_memory = list()
        
        self._train_cum_steps = 0
        self._valid_cum_steps = 0
        self._best_valid_loss = 1e10

    def load_model(self, model) : 
        self._model = model
    
    def load_data(self, train_data, valid_data) : 
        self._trainloader = train_data
        self._validloader = valid_data
        
    def load_prerequisite(self, criterion, optimizer, scheduler, mixup_alpha=0) :
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._mixup_alpha = mixup_alpha
        if mixup_alpha > 0 :
            self._use_mixup = True
        else :
            self._use_mixup = False

    def train(self) :
        self._model.train()
        loss_container = list()
        acc_container = list()

        for inputs, labels in self._trainloader :
            use_cuda = False if Experiments.device == 'cpu' else True
            inputs, labels = inputs.to(Experiments.device), labels.to(Experiments.device)

            if self._use_mixup :
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels,
                                                               self._mixup_alpha, use_cuda)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))
                inputs, targets_a, targets_b = map(lambda x : x.to(Experiments.device), (inputs, targets_a, targets_b))

            self._optimizer.zero_grad()
            outputs = self._model(inputs)

            predictions = outputs.argmax(1)
            acc = (predictions.eq(labels).sum() / outputs.shape[0]).item()

            if self._use_mixup :
                loss = mixup_criterion(self._criterion, outputs, targets_a, targets_b, lam)
            else :
                loss = self._criterion(outputs, labels)

            loss.backward()
            self._optimizer.step()
            
            loss_container.append(loss.item())            
            acc_container.append(acc)
            
            ece = calib.calculate_ce(torch.softmax(outputs, dim=1).cpu().data.numpy(), labels.cpu().data.numpy())
            
            self.write(loss.item(), acc, self._train_cum_steps, 'Training')
            self.write(ece, None, self._train_cum_steps, 'Calibration(Train)')
            self._train_cum_steps += 1
            
        epoch_loss = np.mean(loss_container)
        epoch_acc = np.mean(acc_container)
        return epoch_loss, epoch_acc

    def valid(self) :
        self._model.eval()
        loss_container = list()
        acc_container = list()
        for inputs, labels in self._validloader :

            inputs, labels = inputs.to(Experiments.device), labels.to(Experiments.device)

            outputs = self._model(inputs)

            predictions = outputs.argmax(1)
            acc = (predictions.eq(labels).sum() / outputs.shape[0]).item()

            loss = self._criterion(outputs, labels)

            loss_container.append(loss.item())
            acc_container.append(acc)

            ece = calib.calculate_ce(torch.softmax(outputs, dim=1).cpu().data.numpy(), labels.cpu().data.numpy())

            self.write(loss.item(), acc, self._valid_cum_steps, 'Validation')
            self.write(ece, None, self._valid_cum_steps, 'Calibration(Valid)')
            self._valid_cum_steps += 1

        epoch_loss = np.mean(loss_container)
        epoch_acc = np.mean(acc_container)
        return epoch_loss, epoch_acc
        
    def dump(self, valid_loss) : 
        if valid_loss < self._best_valid_loss : 
            torch.save(self._model.state_dict(), f'{self._modelname}.{self._today_ymd}.pt')
            self._best_valid_loss = valid_loss            
    
    def write(self, loss, acc, cum_step, vers='Training') : 
        if loss is not None :
            self._writer.add_scalar(f'{vers} loss',
                                loss,
                                cum_step)
        if acc is not None : 
            self._writer.add_scalar(f'{vers} accuracy',
                                acc,
                                cum_step)

    def run(self, verbose=False) :
        
        for process in range(Experiments.epoch) : 
            
            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.valid()

            if verbose :
                print("##############################################################################")
                print(f"Epoch : {process} | Train Loss : {train_loss} | Train Accuracy : {train_acc}")
                print(f"Epoch : {process} | Valid Loss : {valid_loss} | Train Accuracy : {valid_loss}")
                print("##############################################################################")

            self.dump(valid_loss)
            self._scheduler.step()