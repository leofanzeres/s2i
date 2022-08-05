from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import models.net_visual_interpretability as net_visual_interpret
from PIL import Image
from PIL import ImageFilter
from matplotlib import pyplot as plt
import numpy as np
import csv
import utils as ut
import values as v


SAVE_IMAGES = True
CLASSIFY_INTERPRET = True
ACTIVE_DROPOUT_G = True
blur_radius = 0
activation_function_generator = 'relu' # activation options: sigmoid, relu, l_relu, softplus, elu, celu, selu, tanh
activation_alpha_generator = 1.0
NOISE_DIM = v.NOISE_VECTOR_DIM
NOISE_MU = v.NOISE_MU # mean for generating noise vector
NOISE_SIGMA = v.NOISE_SIGMA # standard deviation for generating noise vector
# Set s2i to True only if you prefer to load spectrograms instead of audio embeddings, which implies that data will need to pass through
# the audio encoder before entering the generator network.
s2i = False
test_number = '191_R3_1-4_TS'
epochs = ['3999']

paths_list = []
for e in epochs:
    load_and_save_paths = []
    load_and_save_paths.append(v.TRAINED_MODELS_DIR + 'net_visual_generator_model_(test_'+test_number+'_epoch_'+e+').pth')
    load_and_save_paths.append('Test_'+test_number+'_(epoch_'+e+')/')
    paths_list.append(load_and_save_paths)
csv_save_path = 'interpretability_classification_results_from_'+test_number+'.csv'
csv_str = []

gen_input_dimensions = v.AUDIO_EMBEDDING_DIMENSION

if CLASSIFY_INTERPRET:
    LOAD_PATH_VISUAL_INTERPRET_NN = v.VISUAL_INTERPRET_NN_MODEL_FILE
    net_v_interpret_input_length = 32

import models.net_visual_2_generator_dense as net_visual_generator

net_v_gen = net_visual_generator.Net(
    len(v.VEGAS_CLASSES_INDEXES), gen_input_dimensions, activation = activation_function_generator, activation_alpha=activation_alpha_generator)

if CLASSIFY_INTERPRET:
    nets_v_interpret = []
    for _ in range(len(v.VEGAS_CLASSES_INDEXES)):
        nets_v_interpret.append(net_visual_interpret.Net(net_v_interpret_input_length))

# NET AUDIO VISUAL
device_av = net_visual_generator.device
net_v_gen = net_v_gen.to(device_av)

if CLASSIFY_INTERPRET:
    device_interpret = net_visual_interpret.device
    for i in range(len(nets_v_interpret)):
        nets_v_interpret[i] = nets_v_interpret[i].to(device_interpret)
if device_av == 'cuda':
    net_v_gen = torch.nn.DataParallel(net_v_gen)  # @UndefinedVariable
    if CLASSIFY_INTERPRET:
        for i in range(len(nets_v_interpret)):
            nets_v_interpret[i] = torch.nn.DataParallel(nets_v_interpret[i])  # @UndefinedVariable
    cudnn.benchmark = True

# Data
testloader_a = net_visual_generator.get_test_loader(s2i)

if CLASSIFY_INTERPRET:
    for i in range(len(nets_v_interpret)):
        nets_v_interpret[i].load_state_dict(torch.load(LOAD_PATH_VISUAL_INTERPRET_NN[i]))
        nets_v_interpret[i].eval()

def test(paths_list):
    for visual_generator_path, save_path in paths_list:
        net_v_gen.load_state_dict(torch.load(visual_generator_path))
        net_v_gen.eval()
        if ACTIVE_DROPOUT_G:
            for m in net_v_gen.modules():
                if (m.__class__.__name__.startswith('Dropout')) | (m.__class__.__name__.startswith('AlphaDropout')):
                    m.train()
        initial_image = 0
        with torch.no_grad():
            c = 0
            if CLASSIFY_INTERPRET: predictions_by_class = [0]*(len(nets_v_interpret)+1) # plus one for the total number of images
            for batch_idx, (data, target_images, labels) in enumerate(testloader_a):
                data, target_images = data.to(device_av), target_images.to(device_av)
                embeddings = data
                tensor = torch.ones(())  # @UndefinedVariable
                noise_vectors = tensor.new_empty((data.size()[0], NOISE_DIM), dtype=torch.float32, device=device_av)  # @UndefinedVariable
                for i in range(noise_vectors.size()[0]):
                    noise = torch.tensor(np.random.normal(NOISE_MU, NOISE_SIGMA, NOISE_DIM), dtype=torch.float32)  # @UndefinedVariable
                    noise = noise.to(device_av)
                    noise_vectors[i] = noise
                input_g = torch.cat((noise_vectors, embeddings), 1)  # @UndefinedVariable
                input_channels_num = int(input_g.size()[1]/4)
                input_g = input_g.view(-1, input_channels_num, 2, 2)
                output_images = net_v_gen(input_g)

                if SAVE_IMAGES:
                    output_images = output_images.to('cpu').detach()
                    target_images_cpu = target_images.to('cpu').detach()
                    target_labels = []
                    sep = np.empty((net_visual_generator.target_height,10, 3), dtype=float)
                    sep.fill(1.)
                    for i,img in enumerate(output_images):
                        if c >= initial_image:
                            target_img = target_images_cpu[i].numpy().transpose(1, 2, 0)
                            img = img.numpy().transpose(1, 2, 0)
                            label = labels[i]
                            label = v.VEGAS_CLASSES_INDEXES[int(label[:3])-1][2]
                            visual = (img - img.min()) / (img.max() - img.min())
                            columns = 2 # 2 for 2 images + sep
                            if columns > 1:
                                with np.errstate(divide='ignore',invalid='ignore'):
                                    visual_target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
                                    visual_img = (img - img.min()) / (img.max() - img.min())
                                    visual = np.append(visual_target_img, sep, axis=1)
                                    visual = np.append(visual, visual_img, axis=1)

                                fig=plt.figure()
                                rows = 1
                                imgs = (visual_target_img, visual_img)
                                for j in range(0, columns*rows):
                                    fig.add_subplot(rows, columns, j+1)
                                    plt.imshow(imgs[j])
                            else:
                                plt.imshow(visual)
                            result = Image.fromarray((visual * 255).astype(np.uint8))
                            if blur_radius > 0:
                                result = result.filter(ImageFilter.GaussianBlur(blur_radius))
                            result.save(save_path + 'out_{0:06d}'.format(c) + '.jpg')
                            plt.title(label)
                        c += 1

                if CLASSIFY_INTERPRET:
                    target_labels = []
                    for label in labels:
                        label = v.VEGAS_CLASSES_INDEXES[int(label[:3])-1][1]
                        target_labels.append(label)
                    visual_interpret_output = ut.get_visual_interpretability_classification(
                        nets_v_interpret, net_visual_interpret.non_interpretable_image_label, output_images, target_labels, device_interpret)
                    for i, res in enumerate(visual_interpret_output[1]):
                        predictions_by_class[i] = predictions_by_class[i] + res
                    l = len(visual_interpret_output[1])
                    predictions_by_class[l] = predictions_by_class[l] + output_images.size()[0]
            if CLASSIFY_INTERPRET: csv_str.append(predictions_by_class)

test(paths_list)

if CLASSIFY_INTERPRET:
    with open(csv_save_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_str)
