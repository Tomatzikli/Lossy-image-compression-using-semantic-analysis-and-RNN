import json
import torch
from Semantic_analysis_trained.utils import imload_tensor, imshow, array_to_cam, blend
from Semantic_analysis_trained.models import CAM
from Semantic_analysis_trained.train import MResnet


def getCam(image: torch.tensor, gpu=True, topk=1, blend_alpha=0.75,
           save_path="Semantic_analysis/results/"):

    # ImageNet class index to label
    ## ref: https://discuss.pytorch.org/t/imagenet-classes/4923/2
    idx_to_label = json.load(open('Semantic_analysis_trained/imagenet_class_index.json'))
    idx_to_label = {int(key): value[1] for key, value in idx_to_label.items()}

    # set device
    device = torch.device('cuda:%d'%gpu if gpu else 'cpu')

    # -------------------------------------
    model = MResnet(3, 100).to(device)
    ## model with best learning rate of 8e-05
    model.load_state_dict(torch.load("Semantic_analysis_trained/cifar100-resnet_wd8e-05.pth"))
    model = model.to(device)
    network = CAM(model).to(device)
    network.eval()
    image = imload_tensor(image)
    image = image.to(device)

    # make class activation map
    with torch.no_grad():
        prob, cls, cam = network(image, topk=topk)
        img_pil = imshow(image)
        img_pil.save(save_path + "input.jpg")

        for k in range(topk):
            print("Predict '%s' with %2.4f probability" % (idx_to_label[cls[0]], prob[0]))
            cam_ = cam[0].squeeze().cpu().data.numpy()
            cam_pil = array_to_cam(cam_)
            cam_pil.save(save_path + "cam_class__%s_prob__%2.4f.jpg" % (idx_to_label[cls[k]], prob[k]))

            # overlay image and class activation map
            blended_cam = blend(img_pil, cam_pil, blend_alpha)
            blended_cam.save(save_path + "blended_class__%s_prob__%2.4f.jpg" % (idx_to_label[cls[0]], prob[0]))

    return save_path + "cam_class__%s_prob__%2.4f.jpg" % (idx_to_label[cls[0]], prob[0])