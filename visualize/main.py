import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_text_to_grid(att_map,image,head,step):
    
    grid_size = (7, 7)
    
    H,W = att_map.shape
    
    mask = att_map.reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    fig.tight_layout()
       
    ax.imshow(image)
    ax.imshow(mask/np.max(mask), alpha=0.6, cmap='rainbow')
    ax.axis('off')
    #plt.show()
    if not os.path.exists('results/step%d' % step):
        os.makedirs('results/step%d' % step)
    plt.savefig("results/step%d/head%d.png" % (step,head))
    plt.close()



image_num = 2
#head_num = 0
image = Image.open('images/batch5/%d.jpg' % image_num)

batch5 = torch.load('att_map.pth')['ScaledDotProductAttention.forward'] # keys中只有这一个项,是一个长度为720的list。

batch5 = batch5[600:]

for head_num in range(8):
    for step in range(11):
        layer_num = (step+1)*3
        layer_num = layer_num *2
        att_map = batch5[layer_num-1][image_num,head_num,:,:]#(1 49)
        print(att_map.shape)
        visualize_text_to_grid(att_map, image,head_num,step)
