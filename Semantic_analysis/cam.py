import json
import torch
import numpy as np

from Semantic_analysis.utils import imload_tensor, imshow, array_to_cam, blend
from Semantic_analysis.models import CAM



def getCam(image: torch.tensor, network='resnet50', gpu=True, topk=1, blend_alpha=0.75,
           save_path="Semantic_analysis/results/"):
    # ImageNet class index to label
    ## ref: https://discuss.pytorch.org/t/imagenet-classes/4923/2
    idx_to_label = json.load(open('Semantic_analysis/imagenet_class_index.json'))
    idx_to_label = {int(key):value[1] for key, value in idx_to_label.items()}


    # set device
    device = torch.device('cuda:%d'%gpu if gpu else 'cpu')
    network = CAM(network).to(device)
    network.eval()  # turn spesific layers off (Dropouts Layers, BatchNorm Layers etc).
    # The common practice is using torch.no_grad() in pair with model.eval() to turn off gradients computation.
    # image = imload_8(image).to(device)  # change to divided by 98
    # image = imload(image, imsize, cropsize).to(device)
    image = imload_tensor(image)
    image = image.to(device)

    # make class activation map
    with torch.no_grad():
        prob, cls, cam = network(image, topk=topk)

        # tensor to pil image
        img_pil = imshow(image)
        img_pil.save(save_path + "input.jpg")

        for k in range(topk):
            print("Predict '%s' with %2.4f probability"%(idx_to_label[cls[0]], prob[0]))
            cam_ = cam[0].squeeze().cpu().data.numpy()  # keep in gpu?
            cam_pil = array_to_cam(cam_)
            cam_pil.save(save_path+"cam_class__%s_prob__%2.4f.jpg"%(idx_to_label[cls[k]], prob[k]))

            # overlay image and class activation map
            blended_cam = blend(img_pil, cam_pil, blend_alpha)
            blended_cam.save(save_path+"blended_class__%s_prob__%2.4f.jpg"%(idx_to_label[cls[0]], prob[0]))

    return save_path+"cam_class__%s_prob__%2.4f.jpg"%(idx_to_label[cls[0]], prob[0])
