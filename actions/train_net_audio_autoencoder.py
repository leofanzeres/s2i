from __future__ import print_function
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import models.net_audio_autoencoder as net_audio_autoencoder
import models.net_audio as net_audio
import sys
import datetime
import values as v


TEST_NUMBER = '1'
LOAD_PATH_AUDIO_AUTOENCODER = v.AUDIO_AUTOENCODER_NN_MODEL_FILE # Define a pretrained .pth model path or an empty string '' for training the model from scratch
LOAD_PATH_AUDIO_NN = v.AUDIO_NN_MODEL_FILE # Define a pretrained .pth model path
SAVE_PATH = v.TRAINED_MODELS_AUTO_SAVE_DIR + 'net_audio_autoencoder_model_(test_' + TEST_NUMBER + '_' # Use empty string '' for not saving the model
CSV_SAVE_PATH = v.TRAINED_MODELS_AUTO_SAVE_DIR + 'train_net_audio_autoencoder_(test_' + TEST_NUMBER + ').csv'
NUM_EPOCHS = 1000 # The number of epochs to train the model
last_n_epochs_to_save = 15 # Forces to save these last n epochs model
epoch_save_jump = 50 # Forces to save the model from n to n epochs
l_rate = 0.05 # Initial learning rate
momentum = 0.9 # SGD momentum
w_decay = 0 # 0 / 5e-4 / 1e-4 / 1e-5
best_acc = 0 # Best test accuracy
start_epoch = 0 # Start from epoch 0 or last checkpoint epoch
log_interval = 10 # How many batches to wait before logging training status
ACTIVE_DROPOUT = True # Test time dropout
activation_function = 'relu' # Activation options: sigmoid, relu, l_relu, elu, celu, selu, tanh, custom
activation_alpha = 1.0 # 1.0 / 1/16
MAX_LOSS = 0.0003 # Set the maximum desired loss
min_loss = 1000. # Minimum loss will be updated automatically


# Model
print(str(datetime.datetime.now()) + ' ==> Building model..')
net = net_audio_autoencoder.Net(len(v.VEGAS_CLASSES_INDEXES), mode='auto', activation=activation_function, activation_alpha=activation_alpha)
net_a = net_audio.Net(len(v.VEGAS_CLASSES_INDEXES))

print("Test number " + TEST_NUMBER)
print("Learning rate: " + str(l_rate))
print("Momentum: " + str(momentum))
print("Weight decay: " + str(w_decay))
print("ACTIVE_DROPOUT: " + str(ACTIVE_DROPOUT))
print("Activation function: " + activation_function)
print("Activation alpha: " + str(activation_alpha))
print("Input height: " + str(net_audio_autoencoder.input_height))
print("Input width: " + str(net_audio_autoencoder.input_width))
print("Embedding dimension: " + str(v.AUDIO_EMBEDDING_DIMENSION))
if LOAD_PATH_AUDIO_AUTOENCODER == '':
    print("Loaded model: started from scratch")
else:
    print("Loaded model: " + LOAD_PATH_AUDIO_AUTOENCODER.split('/')[-1])
print("\n---------------------------------------------------------------\n")
print("Model Summary:\n")
print(net.torch_summarize(False,True))
print("")
net.summary()
print("")
print("Shape of filters:\n")
print("\n---------------------------------------------------------------\n")

device = net_audio_autoencoder.device
net = net.to(device)
net_a = net_a.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # @UndefinedVariable
    net_a = torch.nn.DataParallel(net_a)  # @UndefinedVariable
    cudnn.benchmark = True

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=momentum, weight_decay=w_decay)

csv_str = []
csv_row = ()

# Data
print(str(datetime.datetime.now()) + ' ==> Preparing data..')
trainloader = net_audio_autoencoder.get_train_loader()
testloader = net_audio_autoencoder.get_test_loader()

# Load Autoencoder NN
if (LOAD_PATH_AUDIO_AUTOENCODER != ''):
    net.load_state_dict(torch.load(LOAD_PATH_AUDIO_AUTOENCODER))

# Load Audio NN
net_a.load_state_dict(torch.load(LOAD_PATH_AUDIO_NN))
net_a.eval()


def train(epoch, print_epoch_res):
    net.train()
    train_pixel_loss = 0
    train_loss = 0
    correct = 0
    global csv_row
    sys.stdout.write('>>>Epoch ' + str(epoch) + ' ')
    sys.stdout.flush()
    for batch_idx, (data, target_class) in enumerate(trainloader):
        data, target_class = data.to(device), target_class.to(device)
        optimizer.zero_grad()
        output = net(data)[0]
        pixel_loss = criterion_MSE(output, data)

        # Classification used only to report accuracy
        classifier_output = net_a(output)
        pred = classifier_output[0].max(1, keepdim=True)[1] # get the indexes of the max log-probability
        correct += pred.eq(target_class.view_as(pred)).sum().item()

        # Other losses can be implemented here

        loss = pixel_loss

        loss.backward()
        optimizer.step()

        train_pixel_loss += pixel_loss.item()
        train_loss += loss.item()

        if ((batch_idx % log_interval == 0) & print_epoch_res):
            print('Train log: [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                batch_idx * len(data), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.item()))
        elif batch_idx % log_interval == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    train_pixel_loss /= len(trainloader.dataset)
    train_loss /= len(trainloader.dataset)
    csv_row = (round(train_pixel_loss,8), round(train_loss,8), correct, len(trainloader.dataset))
    print('\nTrain set: Average pixel-loss: {:.8f}, Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_pixel_loss, train_loss, correct, len(trainloader.dataset), 100. * correct / len(trainloader.dataset)))


def test(epoch):
    net.eval()
    if ACTIVE_DROPOUT:
        for m in net.modules():
            if (m.__class__.__name__.startswith('Dropout')) | (m.__class__.__name__.startswith('AlphaDropout')):
                m.train()
    test_pixel_loss = 0
    test_loss = 0
    correct = 0
    global csv_row
    global MAX_LOSS
    global min_loss
    with torch.no_grad():
        for batch_idx, (data, target_class) in enumerate(testloader):
            data, target_class = data.to(device), target_class.to(device)
            output = net(data)[0]
            pixel_loss = criterion_MSE(output, data)

            # Classification used only to report accuracy
            classifier_output = net_a(output)
            pred = classifier_output[0].max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target_class.view_as(pred)).sum().item()

            # Other losses can be implemented here

            loss = pixel_loss

            test_pixel_loss += pixel_loss.item()
            test_loss += loss.item()

    test_pixel_loss /= len(testloader.dataset)
    test_loss /= len(testloader.dataset)
    csv_row += (round(test_pixel_loss,8), round(test_loss,8), correct, len(testloader.dataset),)
    csv_str.append(csv_row)
    precision = 100. * correct / len(testloader.dataset)

    if ((epoch >= start_epoch + NUM_EPOCHS - last_n_epochs_to_save) | ((epoch+1) % epoch_save_jump == 0) | (test_loss <= MAX_LOSS) & (test_loss <= min_loss) ):
        save_model(epoch)
        min_loss = test_loss

    print('Test set: Average pixel-loss: {:.8f}, Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_pixel_loss, test_loss, correct, len(testloader.dataset), precision))

def save_model(epoch):
    torch.save(net.state_dict(), SAVE_PATH + 'epoch_' + str(epoch) + ').pth')

for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
    train(epoch, False)
    test(epoch)

if (SAVE_PATH != ''):
    torch.save(net.state_dict(), SAVE_PATH + 'epoch_' + str(NUM_EPOCHS - 1) + ').pth')

with open(CSV_SAVE_PATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_str)
