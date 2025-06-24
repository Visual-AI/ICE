#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
import os 

import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from skimage import measure
from scipy.optimize import linear_sum_assignment as LinearSumAssignment


def process_mask(image, pred, label_to_mask, target_label=[], save_path=''):
  image = np.array(image)

  for i, label in enumerate(label_to_mask.keys()):
    if label in target_label:
      bin_mask = np.zeros((512, 512), dtype=np.uint8)
      for mask in label_to_mask[label]:
        bin_mask[pred.reshape(512, 512) == mask] = 255

      if label != 'background':
        bin_mask_save = bin_mask.copy()
        separate_instances(bin_mask_save, save_path, label)
      else:
        img = PIL.Image.fromarray(bin_mask)
        img.save(os.path.join(save_path, "background.png".format(label)))

      # update original image
      image[bin_mask == 255] = 0

  # save the original image with out the masked area
  updated_img = PIL.Image.fromarray(image)

  # updated mask, if values > 0, set to 1
  non_masked_area = np.array(updated_img)
  non_masked_area[non_masked_area > 0] = 1

  return non_masked_area


def separate_instances(binary_mask, save_path, label):
    # Label connected components in the binary mask
    labeled_mask = measure.label(binary_mask, connectivity=2)

    # Get the number of unique labels (excluding background)
    num_instances = labeled_mask.max()

    counter = 0
    # Loop over each labeled instance
    for i in range(1, num_instances + 1):
        # Create a mask for the current instance
        instance_mask = np.where(labeled_mask == i, 255, 0).astype(np.uint8)

        # Convert to PIL Image for saving
        instance_image = Image.fromarray(instance_mask)

        # if 1's value is less than 10% of the total pixels, ignore it
        # check how many pixels > 0
        pixels_percentage = np.where(instance_mask > 0, 1, 0).sum() / (512 * 512)
        if pixels_percentage > 0.025:
            # Save the individual instance mask
            output_filename = os.path.join(save_path, f'mask_{label}_{counter}.png')
            instance_image.save(output_filename)

            print(f"Saved: {output_filename}")
            counter += 1


def save_mask(image, pred, label_to_mask, dir=None, dir_spec=None):
	# create a directory for saving the images
	save_path = os.path.join(dir, dir_spec)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	for i, label in enumerate(label_to_mask.keys()):
		image = image.reshape(512*512,-1)
		bin_mask = np.zeros_like(image)
		for mask in label_to_mask[label]:
			bin_mask[(pred.reshape(512*512)==mask).flatten(),:] = 1

		# save image using pil
		img = PIL.Image.fromarray((image*bin_mask).reshape(512,512,-1))
		print(label)
		img.save(save_path + "/{}.png".format(label))
        

def process_image(image):
  # to numpy
  image = np.array(image)
  s = tf.shape(image)
  w, h = s[0], s[1]   
  c = tf.minimum(w, h)
  w_start = (w - c) // 2
  h_start = (h - c) // 2
  image = image[w_start : w_start + c, h_start : h_start + c, :]
  image = tf.image.resize(image, (512, 512))
  return image


def find_edges(M):
  edges = np.zeros((512,512))
  m1 = M[1:510,1:510] != M[0:509,1:510]
  m2 = M[1:510,1:510] != M[2:511,1:510]
  m3 = M[1:510,1:510] != M[1:510,0:509]
  m4 = M[1:510,1:510] != M[1:510,2:511]
  edges[1:510,1:510] = (m1 | m2 | m3 | m4).astype(int)
  x_new = np.linspace(0, 511, 512)
  y_new = np.linspace(0, 511, 512)
  x_new,y_new=np.meshgrid(x_new,y_new)
  x_new = x_new[edges==1]
  y_new = y_new[edges==1]
  return x_new,y_new

def vis_without_label(M,image,index=None,save=False,dir=None,num_class=26):
  fig = plt.figure(figsize=(20, 20))
  ax = plt.subplot(1, 3, 1)
  ax.imshow(image)
  ax.set_title("Input",fontdict={"fontsize":30})
  plt.axis("off")

  x,y = find_edges(M)
  ax = plt.subplot(1, 3, 2)
  ax.imshow(image)
  ax.imshow(M,cmap='jet',alpha=0.5, vmin=-1, vmax=num_class)
  ax.scatter(x,y,color="blue", s=0.5)
  ax.set_title("Overlay",fontdict={"fontsize":30})
  plt.axis("off")

  ax = plt.subplot(1, 3, 3)
  ax.imshow(M, cmap='jet',alpha=0.5, vmin=-1, vmax=num_class),
  ax.set_title("Segmentation",fontdict={"fontsize":30})
  plt.axis("off")

  if save:
    fig.savefig(open(dir+"/example_{}.png".format(index), 'wb'), format='png',bbox_inches='tight', dpi=200)
    plt.close(fig)


def semantic_mask(image, pred, label_to_mask):
  num_fig = len(label_to_mask)
  plt.figure(figsize=(20, 20))
  for i,label in enumerate(label_to_mask.keys()):
      ax = plt.subplot(1, num_fig, i+1)
      image = image.reshape(512*512,-1)
      bin_mask = np.zeros_like(image)
      for mask in label_to_mask[label]:
        bin_mask[(pred.reshape(512*512)==mask).flatten(),:] = 1
      ax.imshow((image*bin_mask).reshape(512,512,-1))
      ax.set_title(label,fontdict={"fontsize":30})
      ax.axis("off")

def _fast_hist(label_true, label_pred, n_class):
    # Adapted from https://github.com/janghyuncho/PiCIE/blob/c3aa029283eed7c156bbd23c237c151b19d6a4ad/utils.py#L99
    pred_n_class = np.maximum(n_class,label_pred.max()+1)
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(pred_n_class * label_true[mask] + label_pred[mask],\
                       minlength=n_class * pred_n_class).reshape(n_class, pred_n_class)
    return hist

def hungarian_matching(pred,label,n_class):
  # X,Y: b x 512 x 512
  batch_size = pred.shape[0]
  tp = np.zeros(n_class)
  fp = np.zeros(n_class)
  fn = np.zeros(n_class)
  all = 0
  for i in range(batch_size):
    hist = _fast_hist(label[i].flatten(),pred[i].flatten(),n_class)
    row_ind, col_ind = LinearSumAssignment(hist,maximize=True)
    all += hist.sum()
    fn += (np.sum(hist, 1) - hist[row_ind,col_ind])
    tp += hist[row_ind,col_ind]
    hist = hist[:, col_ind] # re-order hist to align labels to calculate FP
    fp += (np.sum(hist, 0) - np.diag(hist))
  return tp,fp,fn,all