import math
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import scipy

from icecream import ic

def count_characters_in_messages(filepath, sample_numbers,char_set):

    counter = [0]*len(char_set)
    char_set_length = len(char_set)
    #ic(char_set_length)

    # assume they are all TXT files
    for sample_no in sample_numbers:
        full_filepath = filepath + str(sample_no) + ".txt"

        with open(full_filepath) as infile:
            for line in infile:

                # for each line, just count the number of times this letter appears.
                # NOTE: this is very slow and not optimized, but it'll work for now (and not be prohibitively slow)
                # if ever needed for better purposes, this should be sped up
                for i in range(char_set_length):

                    char = char_set[i]
                    #ic(char)

                    this_count = line.count(char)
                    counter[i] += this_count

    #ic(counter)
    return counter

def calculate_conditional_prob(count, total_count, alpha, total_chars):
    return (count + alpha)/(total_count + (alpha * total_chars))

def get_conditional_probabilities(filepath, sample_numbers,char_set, alpha):

    counter = count_characters_in_messages(filepath, sample_numbers, char_set)
    total_count = sum(counter)

    #ic(counter[0])
    #ic(total_count)
    theta = list(map(lambda x: calculate_conditional_prob(x, total_count, alpha, len(char_set)), counter))
    #ic(theta)
    return theta

def calculate_log_probability(xs, thetas):
    ret = 0
    for i in range(len(xs)):
        x = xs[i]
        theta = thetas[i]

        ret += x * math.log(theta)
    return ret

def classify_messges(char_set, sample_numbers, alpha, lang_char, messages_to_classify):

    for i in messages_to_classify:
        msg_x = count_characters_in_messages("hw4/languageID/" + str(lang_char), [10], char_set)
        sum_x = sum(msg_x) + (alpha*27)
        #ic(sum_x)
        #ic(msg_x)
        arr = list(map(lambda x: math.log((x + alpha) / sum_x), msg_x))
        #log_p_x = sum(arr)
        #ic(log_p_x)

        # we don't need to calculate log p(x). Just need to compare the log probabilities
        th_e = get_conditional_probabilities("hw4/languageID/e", sample_numbers, char_set, alpha)
        th_s = get_conditional_probabilities("hw4/languageID/s", sample_numbers, char_set, alpha)
        th_j = get_conditional_probabilities("hw4/languageID/j", sample_numbers, char_set, alpha)
        pr_e = calculate_log_probability(msg_x, th_e)
        pr_s = calculate_log_probability(msg_x, th_s)
        pr_j = calculate_log_probability(msg_x, th_j)

        # calculate the max
        max_pr = max(pr_e,pr_s,pr_j)

        classification = ""
        if max_pr == pr_e:
            classification="e"
        elif max_pr == pr_s:
            classification="s"
        elif max_pr == pr_j:
            classification="j"

        print("Msg: ", i, "Classification:", classification)


def nn_from_scratch():
    d0 = 784
    d1 = 300
    d2 = 200
    #d3

    # initialize
    #imagenet_data = torchvision.datasets.ImageNet('hw4/')
    # handling this taken from
    # some of this taken from https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                              download=True,
                                                              transform=transforms.ToTensor(),
                                                              train=True),
                                               batch_size=10,
                                               shuffle=True)

    count = 0
    for batch_id, (data, label) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(label)

        if count == 1:
            return
        count += 1

        print("Batch: ", batch_id)
        print("Data: ", data)
        print("Shape: ", np.shape(data))
        print("Target: ", target)

    w1 = np.zeros(())


# from https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch?irclickid=3EFTkYQsYxyIRCJ33oUxgVIDUkAW0Fzhwy252A0&irgwc=1&utm_medium=affiliate&utm_source=impact&utm_campaign=000000_1-2003851_2-mix_3-all_4-na_5-na_6-na_7-mp_8-affl-ip_9-na_10-bau_11-Bing%20Rebates%20by%20Microsoft&utm_content=BANNER&utm_term=EdgeBingFlow
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR # https://github.com/pytorch/examples/blob/main/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

# from https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main_nn_pytorch():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', # updated to 2 from 14, for testing
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # model.apply(weights_init_uniform) # uncomment for applying initial weight of uniform dist (4.3b)
    # init_all(model, torch.nn.init.constant_, 0.) # uncomment for applying initial weight of constant 0 (4.3a)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr) # optimizer used for 3.2

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

def nn_by_pytorch(self):
    X = torch.rand(1, 28, 28)
    model = NeuralNetwork()
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")



# run
# 2.2. english
char_set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w", "x", "y", "z", " "]
sample_numbers = [0,1,2,3,4,5,6,7,8,9]
alpha = 0.5

# part 1
#get_conditional_probabilities("hw4/languageID/e", sample_numbers, char_set, alpha)

# part 2
#th = get_conditional_probabilities("hw4/languageID/s", sample_numbers, char_set, alpha)
#th = get_conditional_probabilities("hw4/languageID/j", sample_numbers, char_set, alpha)

# part 3
#print(count_characters_in_messages("hw4/languageID/e", [10], char_set))

# part 5
'''
e10 = count_characters_in_messages("hw4/languageID/e", [10], char_set)
sum_e10 = sum(e10)
arr = list(map(lambda x: math.log(x/sum_e10), e10))
log_p_x = sum(arr)
ic(log_p_x)
th_e = get_conditional_probabilities("hw4/languageID/e", sample_numbers, char_set, alpha)
th_s = get_conditional_probabilities("hw4/languageID/s", sample_numbers, char_set, alpha)
th_j = get_conditional_probabilities("hw4/languageID/j", sample_numbers, char_set, alpha)
pr_e = calculate_log_probability(e10, th_e)
pr_s = calculate_log_probability(e10, th_s)
pr_j = calculate_log_probability(e10, th_j)
ic(pr_e)
ic(pr_s)
ic(pr_j)
'''

# part 6
'''
print("English")
classify_messges(char_set, sample_numbers, alpha, "e", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print("Spanish")
classify_messges(char_set, sample_numbers, alpha, "s", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print("Japanese")
classify_messges(char_set, sample_numbers, alpha, "j", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
'''

# part 3.1
main_nn_pytorch()