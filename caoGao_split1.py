import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(1)
'''

checkpoint = torch.load('/home/hl/hl/FSMOD/model_0070000.pth', map_location=torch.device("cpu"))
model = checkpoint['model']

# roi_heads.box.predictor.cls_score.weight

change = [('roi_heads.box.predictor.cls_score.weight', (7, 1024)), ('roi_heads.box.predictor.cls_score.bias', 7)]
t = torch.empty(change[0][1])
torch.nn.init.normal_(t, std=0.001)
model[change[0][0]] = t

t = torch.empty(change[1][1])
torch.nn.init.constant_(t, 0)
model[change[1][0]] = t

change2 = [('roi_heads.box.predictor.bbox_pred.weight', (28, 1024)), ('roi_heads.box.predictor.bbox_pred.bias', 28)]
t = torch.empty(change2[0][1])
torch.nn.init.normal_(t, std=0.001)
model[change2[0][0]] = t

t = torch.empty(change2[1][1])
torch.nn.init.constant_(t, 0)
model[change2[1][0]] = t

checkpoint = dict(model=model)
torch.save(checkpoint, 'z_base_pretrain.pth')
