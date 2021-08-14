import torch
import torch.nn as nn
import torch.nn.functional as F



def get_visual_interpretability_classification(nets_v_interpret, non_interpretable_image_label, input_images, sound_class_labels, device):
    """Returns
        [0]: List of predicted class for each image of the batch. Ex: [0, 0, 1, 0, ..., 0]. Dimension equal to batch size.
        [1]: List of sum of interpretable images for each class. Ex: [4, 0, 3, 6, 5]. Dimension equal to the number of sound classes.
    """
    
    _, num_channels, height, width = input_images.size()
    one_image_batch = torch.zeros((1, num_channels, height, width), device=device)  # @UndefinedVariable
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    predictions = []
    predictions_by_class = [0]*len(nets_v_interpret)
    for i, img in enumerate(input_images):
        one_image_batch[0] = img
        output = nets_v_interpret[sound_class_labels[i]](one_image_batch)
        num_classes = output.size()[1]
        res = 0
        if num_classes==1:
            # Correct implementation. Variable target not initialized. Why use a tensor???
            #predictions = torch.zeros(target.size(), device=device)  # @UndefinedVariable
            for i,o in enumerate(output):
                if o > 0:
                    res = 1
                else:
                    res = non_interpretable_image_label
        elif num_classes==2:
            res = output.max(1)[1].item()
        predictions.append(res) # get the index of the max log-probability
        if res > 0:
            predictions_by_class[sound_class_labels[i]] += 1
    return predictions, predictions_by_class


def get_activated(x, act_type, n_slope=0.2, alph=1.0):
    if act_type == 'sigmoid':
        x = torch.sigmoid(x)  # @UndefinedVariable
    elif act_type == 'relu':
        x = F.relu(x)
    elif act_type == 'l_relu':
        l_relu = nn.LeakyReLU(negative_slope=n_slope)
        x = l_relu(x)
    elif act_type == 'softplus':
        x = F.softplus(x)
    elif act_type == 'elu':
        elu = nn.ELU(alpha=alph)
        x = elu(x)
    elif act_type == 'celu':
        celu = nn.CELU(alpha=alph)
        x = celu(x)
    elif act_type == 'selu':
        selu = nn.SELU(alpha=alph)
        x = selu(x)
    elif act_type == 'tanh':
        x = torch.tanh(x)  # @UndefinedVariable
    
    return x


