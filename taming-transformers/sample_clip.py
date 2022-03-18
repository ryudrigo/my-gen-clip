from omegaconf.omegaconf import OmegaConf
from taming.models.clip_transformer import CLIPCond
from main import instantiate_from_config
import torch, glob
from omegaconf import OmegaConf
import clip
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

config = OmegaConf.load('configs/ffhq_thumbnails_transformer.yaml')
clip_cond = instantiate_from_config(config.model).cuda()

steps = 16
prompt="donald trump"
x = torch.zeros(3,64, 64).cuda()
x = x.unsqueeze(dim=0)
_, x = clip_cond.encode_to_z(x)
x = x[:,-8:]
x_cond = clip_cond.encode_text_to_c(prompt)
x_cond = x_cond.unsqueeze(dim=1)
for k in range(steps):
  logits, _ = clip_cond.transformer(x, embeddings=x_cond)
  logits = logits[:,-1,:]
  probs = F.softmax(logits, dim=-1)
  ix = torch.multinomial(probs, num_samples=1)
  x = torch.cat((x, ix), dim=1)

x = x[:,-16:]
img = clip_cond.decode_to_img(x, (1,16*16,4,4))
img = (img - img.min()) /(img.max()-img.min())
img = ToPILImage()(img[0])
img.save('abacate.png')
