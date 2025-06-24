# ---------------------------------------------------------------------------------------------
# Main file of ICE (Intrinsic Concept Extraction) framework
# Stage One: Automatic Concept Localization via Diffusion Model
#
# Title: ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models (CVPR 2025)
# ArXiv: https://arxiv.org/abs/2503.19902
# Copyright 2025, by Fernando Julio Cendra (fcendra@connect.hku.hk)
# ---------------------------------------------------------------------------------------------
import os
import logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import nltk
import torch
import open_clip
import skimage
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from keras_cv.src.models.stable_diffusion.image_encoder import ImageEncoder
from stage_one_utils.third_party.keras_cv.stable_diffusion import StableDiffusion 

from stage_one_utils.SPLICE import splice
from stage_one_utils.diffseg.utils import process_image, process_mask
from stage_one_utils.diffseg.segmentor import DiffSeg

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
is_noun = lambda pos: pos[:2] == 'NN'


# Filter noun phrases from the list of concepts
def filter_noun_phrases(concepts):
    # Concatenate all the concepts of list into one string
    tokenized = nltk.word_tokenize(concepts)
    nouns = [(i, word) for i, (word, pos) in enumerate(nltk.pos_tag(tokenized)) if is_noun(pos)]
    return nouns


def center_crop(image):
    s = np.shape(image)
    w, h = s[0], s[1]
    c = min(w, h)
    w_start = (w - c) // 2
    h_start = (h - c) // 2
    image = np.array(image)
    image = image[w_start:w_start + c, h_start:h_start + c]
    image = Image.fromarray(image)
    return image


class AutomaticConceptLocalization:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Splice params
        self.top_k_centroids = 4
        self.top_k = 5

        # Init models
        self.i2t_model_init()
        self.t2i_model_init()

    def i2t_model_init(self):
        '''Initialize the image to text (i2t) retriever via SPliCE model'''
        l1_penalty = 0.2
        model = 'open_clip:ViT-B-32'
        self.vocab_ ='laion'
        
        # Load SPliCE model, image to text-concept extraction via CLIP model
        self.splicemodel = splice.load(model, self.vocab_, 15000, self.device, l1_penalty=l1_penalty, return_weights=True)
        self.preprocess = splice.get_preprocess(model)
        self.tokenizer = splice.get_tokenizer(model)

        # Load CLIP model
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip_model.to(self.device)

    def t2i_model_init(self):
        '''Initialize the text to image (t2i) segmentor via DiffSeg model'''
        KL_THRESHOLD = [1.1]*3 # KL_THRESHOLD controls the merging threshold # ORI: 1.0
        NUM_POINTS = 16  
        REFINEMENT = True

        # Load Stable Diffusion model for open-vocabulary image segmentation
        with tf.device('/GPU:1'):
            self.image_encoder = ImageEncoder()
            self.vae=tf.keras.Model(
                self.image_encoder.input,
                self.image_encoder.layers[-1].output,
            )
            self.sd_model = StableDiffusion(img_width=512, img_height=512)

        self.segmentor = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)

    def get_nearest_data(self, image, neg_prompts=None):
        '''
        Given an unlabelled image, we would like to "decomposed" the image into a set of concept texts.
        These concept texts can be found in the vocabulary of our diffusion model.

        To effectively decomposed the image to text, we make use of SPliCE model: https://github.com/AI4LIFE-GROUP/SpLiCE
        '''
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        weights, l0_norm, cosine = splice.decompose_image(img, self.splicemodel, self.device)
        vocab = splice.get_vocabulary(self.vocab_, 15000)
        _, indices = torch.sort(weights, descending=True)
        remove_words = ['banner', 'silhouette', 'noir', 'han', 'teaser', 'animation', 'warcraft', 'shadow']

        concept_list = []
        for idx in indices.squeeze():
            if neg_prompts is None:
                if weights[0, idx.item()].item() == 0:
                    break
                if vocab[idx.item()] not in remove_words:
                    concept_list.append(vocab[idx.item()])

            else:
                # filter out vocab words that are similar to neg_prompt
                if weights[0, idx.item()].item() == 0:
                    break
                input_tokens = self.tokenizer([vocab[idx.item()]]).to(self.device)
                input_encoded = self.splicemodel.encode_text(input_tokens)

                neg_prompt_tokens = self.tokenizer(neg_prompts).to(self.device)
                neg_prompt_encoded = self.splicemodel.encode_text(neg_prompt_tokens)

                sim = cosine_similarity(input_encoded.cpu().detach().numpy(), neg_prompt_encoded.cpu().detach().numpy())
                if sim < 0.85 and vocab[idx.item()] not in remove_words: #0.75
                    concept_list.append(vocab[idx.item()])
                else:
                    print("Filtered out: ", vocab[idx.item()])
                    print("sim: ", sim)
                    continue

        concept_list_tokens = self.tokenizer(concept_list).to(self.device)
        concept_list_encoded = self.splicemodel.encode_text(concept_list_tokens)

        # Perform k-means
        kmeans = KMeans(n_clusters=4, random_state=0).fit(concept_list_encoded.cpu().detach().numpy())

        # Get top-k clusters thru query features
        query_feats = self.splicemodel.extract_image(img)
        query_feats = query_feats.reshape(1, -1)
        query_feats = query_feats.cpu().detach().numpy()

        # Calculate cosine similarity between query feats and cluster centers
        sim = cosine_similarity(query_feats, kmeans.cluster_centers_)

        # Get the top-k clusters
        topk_clusters = sim.argsort()[:, -self.top_k_centroids:][0]

        # Reverse the cluster order
        topk_clusters = topk_clusters[::-1]

        # Get the concepts in the top-k clusters
        topk_concepts = []
        for i in topk_clusters:
            # Get all the concept in cluster i
            cluster_concepts = [concept_list[j] for j in range(len(concept_list)) if kmeans.labels_[j] == i]
            topk_concepts.append(cluster_concepts)

        return topk_concepts, concept_list

    # using clip model, extract top 5 captions for the given image
    def get_clip_captions(self, image, concepts, top_k):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(concepts).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(top_k)
            
            top_k_concepts = []
            for idx in indices:
                top_k_concepts.append(concepts[idx])

            return top_k_concepts
        
    def segment(self, image, prompt, counter, save_path):
        '''Given an image and a prompt, segment the image based on the prompt'''
        with tf.device('/GPU:0'):
            # By adding "things" and "background" to the prompt, 
            # It better distinguishes the target object from the rest.
            cond_prompt = prompt + " things background"
            image = process_image(image)

            image = tf.cast(image, tf.float32) / 127.5 - 1
            latent = self.vae(tf.expand_dims(image, axis=0), training=False)
            image, weight_64, weight_32, weight_16, weight_8, x_weights_64, x_weights_32, x_weights_16, x_weights_8 = self.sd_model.text_to_image(
                cond_prompt,
                batch_size=1,
                latent=latent,
                timestep=150,
            )
            pred = self.segmentor.segment(weight_64, weight_32, weight_16, weight_8) # b x 512 x 512
            x_weight = self.segmentor.aggregate_x_weights([x_weights_64[0], x_weights_32[0], x_weights_16[0], x_weights_8[0]],weight_ratio=[1.0,1.0,1.0,1.0])
            
            nouns = [(i,word) for i, word in enumerate(cond_prompt.split(" "))]
            label_to_mask = self.segmentor.get_semantics(pred[0], x_weight[0], nouns, voting="mean")
            non_masked_area = process_mask(image[0], pred[0], label_to_mask, target_label=[prompt, "background"], save_path=save_path)

            if counter == 0:
                # Save the original image
                save_img = Image.fromarray((image[0]).reshape(512,512,-1))
                save_img.save(os.path.join(save_path, "img.jpg"))
        
        return non_masked_area
    
    def run(self):
        print("***** Run stage one automatic concept localization *****")
        image_path = self.args.image_path
        save_path = os.path.join(self.args.output_path, Path(image_path).parts[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = Image.open(image_path)
        image = center_crop(image)

        # Save the original image
        image.save(os.path.join(save_path, "original.jpg"))
        mask_percentage = 1.0
        counter = 0
        neg_prompts = []

        while mask_percentage > 0.1:
            # Get concepts from the image
            if len(neg_prompts) > 0: 
                _, concept_list = self.get_nearest_data(image, noun_list[0])
            else: 
                _, concept_list = self.get_nearest_data(image)

            # Get top-k concepts from the image
            top_k_concepts = self.get_clip_captions(image, concept_list, self.top_k)

            noun_list = []
            for concept in top_k_concepts:
                noun = filter_noun_phrases(concept)
                if len(noun) > 0:
                    noun_list.append(list(noun[0])[1])
            
            print("Top 1 text-retrived: ", noun_list[0])
            neg_prompts.append(noun_list[0])

            indicator = self.segment(image, noun_list[0], counter, save_path=save_path)
            indicator_rescaled = skimage.transform.resize(indicator, (image.size[0], image.size[1]), order=0, preserve_range=True, anti_aliasing=False)
            image = np.array(image) * indicator_rescaled

            # Convert to uint8 and then to PIL image
            image = (image).astype(np.uint8)
            image = Image.fromarray(image)
            mask_percentage = np.where(indicator > 0, 1, 0).sum() / (512 * 512)
            counter += 1

            # If nnumber of counyer is unreasonably high, break the loop
            if counter > 10:
                print("Too many iterations, breaking the loop...")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic Concept Localization via Diffusion Model')
    parser.add_argument('--image_path', type=str, help='Path to the image', default='data/../img.jpg')
    parser.add_argument('--output_path', type=str, help='Path to save the masked images', default='datasets')
    args = parser.parse_args()

    # Check if the output path exists, if not, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    '''Start Automatic Concept Localization process...'''
    try:
        extractor = AutomaticConceptLocalization(args)
        extractor.run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("You can now proceed to Stage Two of ICE framework.")
