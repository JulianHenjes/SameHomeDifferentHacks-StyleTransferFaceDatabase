# Face Dataset: CelebA
# Style Dataset: https://www.kaggle.com/ikarus777/best-artworks-of-all-time/data
# Tensorflow Hub Style Transfer Model: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb

import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import normalize

from PIL import Image

# Load Tensorflow Hub Module
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

#Path to datasets and output path
content_path = "C:\\Users\\hench\\Desktop\\Documents\\University\\Computer Vision\\CWMaterial\\FaceDatabase\\TrainCelebA\\TrainCelebA"
style_path = "C:\\Users\\hench\\Desktop\\Documents\\Projects\\HackAtHome\\Styles\\images\\images"
output_path = "C:\\Users\\hench\\Desktop\\Documents\\Projects\\HackAtHome\\Output"
#Start on face: start_num, generate the next batch_size faces
start_num = 5146
batch_size = 6000 - start_num

style_names = [x[0] for x in os.walk(style_path)][1:]

# Recommended style size is (256,256), as this is what the model was originally trained on
def load_image(filepath,filename,resize=(256,256),doResize=False,doCrop=False):
    image = Image.open(filepath+"\\"+filename)
    if doResize:
        image = image.resize(resize,Image.ANTIALIAS)
    if doCrop:
        w,h = image.size
        image = image.crop((w//2-resize[0],h//2-resize[1],w//2+resize[0],h//2+resize[1]))
    image = np.array(image)[:,:,0:3]
    return image

for i in range(start_num,start_num + batch_size):
    try:
        # Load content image i
        im_content = load_image(content_path,f"{i:06}.jpg",doResize=True,resize=(1024,1024))
        print(f"Content Image: {i:06}.jpg")
        # Choose a random artists' style to apply
        style = style_names[random.randint(0,len(style_names)-1)]
        # Choose a random sample from the artists' work
        style_samples = [f for f in os.listdir(style) if os.path.isfile(os.path.join(style, f))]
        sample = style_samples[random.randint(0,len(style_samples)-1)]
        print(f"Style Image: {sample}")
        #Load style image
        im_style = load_image(style,sample,doCrop=True)

        # Convert to numpy arrays and normalise to work with tensorflow
        # Must be 4D Input (1, width, height, 3)

        # Content
        content_image = np.array([np.array(im_content)[:, :, 0:3].astype('float32')])
        content_image *= 1/(content_image.max())

        # Style
        style_image = np.array([np.array(im_style)[:, :, 0:3].astype('float32')])
        style_image *= 1/(style_image.max())

        # Perform style transfer
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

        # Convert output to PIL image
        im_output = tf.keras.preprocessing.image.array_to_img(outputs[0][0])
        # Save to output directory
        im_output.save(output_path+"\\"+f"{i:06}.{sample}","jpeg")
        print(f"Output: {i:06}.{sample}")
        print("")
    except:
        print("Error Occurred, Image Skipped...")


