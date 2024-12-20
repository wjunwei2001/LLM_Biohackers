# LLM_Biohackers
- Wang Junwei
- Goh Eng Zhong Joshua
- Qu Zhetao
- Jonathan Ng

## Problem Statement & Solution
Lack of training in how to use ultrasound for proficiency identifying anomalies and pathologies in the images acquired for formulateion of a diagnosis. The barrier to training is the lack of access to patients with said pathologies and scarcity of quality images.

![2023-07-30 13 59 25](https://github.com/wjunwei2001/LLM_Biohackers/assets/96434745/44cc9194-8e1d-4f8f-8911-a5461c89a6ed)


## Data Processing & Augmentation
Our dataset initially contained extra data (*_mask.png files) as it originated from a different project with a different use case. Our `data_extraction.py` file filters and extracts the relevant files (non *_mask.png files) to obtain a dataset of only malignant breast cancer ultrasound images. 

From there, we run the data_augmentation.py script to generate a wider variety of data to be used for future training of the model, preventing overfitting of data. In this script, we adopt various techniques (Random Width & Height Shifting, Shearing, Horizontal Flipping) to generate a greater variety of data. 
We do so as training of the GAN model will be more effective on a larger dataset.

The dataset was obtained from https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset 


## GAN - Image-to-Image generation
Generative Adversarial Networks use neural networks **Generative Modeling** for automatic discovery and learning of regularities or patterns in input data to be able to new examples that plausibly could have been drawn from the original dataset. The GAN model consists of 2 neural networks, **Generator** and **Discriminator**, that are trained in tandem for each batch of training data with the Generator's objective to "fool" the discriminator. The Adam optimizer with some custom parameters (betas) that are known to work well for GANs.

Discriminator is trained for a few epochs, then the generator is trained for a few epochs, and repeat. This way both the generator and the discriminator get better at doing their jobs. 

<img width="346" alt="Screenshot 2023-07-30 at 1 16 09 PM" src="https://github.com/wjunwei2001/LLM_Biohackers/assets/96434745/716488f6-3fd8-46ac-a130-2988f206f2b2">

The generator generates a fake image given a seed of a latent tensor (matrix of random numbers), and the discriminator attempts to detect whether a given sample is real/fake (originates from dataset). 

The output of the discriminator is a single number between 0 and 1, which can be interpreted as the probability of the input image being real i.e. picked from the original dataset. Since the discriminator is a binary classification model, the binary cross entropy loss function is used to quantify how well it is able to differentiate between real and generated images.

GANs are extremely sensitive to hyperparameters, activation functions and regularization. Notably, we have tied down the to the following parameters to reach the current best model
- beta value (0.9)
- learning rate (0.00005)
- epochs ran (300-1000)
- loss function (binary_cross_entropy)

The implementation of the GAN was inspired by the tutorial at https://jovian.com/aakashns/06b-anime-dcgan?action=duplicate_notebook 

### Running the GAN training script
Define the hyperparameters of your choice before running the 1st cell in `trainGAN.ipynb`. Batches of outputted sample images from the generator will be saved in the "generated" folder for user monitoring of the model. 

![2023-07-30 13 59 31](https://github.com/wjunwei2001/LLM_Biohackers/assets/96434745/3b75edcd-062c-448d-b821-cd0f2728d98b)


Once training job is done, run the 2nd and 3rd cells to obtain the graphical visualisation of the model training performance. Visualizing
losses is quite useful for monitoring the training process. For GANs, we expect the generator's loss to reduce over time, without the discriminator's loss getting too high.

![2023-07-30 13 59 29](https://github.com/wjunwei2001/LLM_Biohackers/assets/96434745/dbb806e4-ca7d-4a30-beb7-63b419eaa90c)

If ideal, run the 4th and 5th cell to save the model weight and obtain a mp4 recording of the full training loop.


## LLM - ultrasound image caption
The LLM is microsoft/git-base from Hugging Face. This model is called the GIT (short for GenerativeImage2Text) model, base-sized version. It was introduced in the paper GIT: A Generative Image-to-text Transformer for Vision and Language by Wang et al (https://arxiv.org/abs/2205.14100). GIT is a Transformer decoder conditioned on both CLIP image tokens and text tokens. The model is trained using "teacher forcing" on a lot of (image, text) pairs. The goal for the model is simply to predict the next text token, giving the image tokens and previous text tokens. The model has full access to (i.e. a bidirectional attention mask is used for) the image patch tokens, but only has access to the previous text tokens (i.e. a causal attention mask is used for the text tokens) when predicting the next text token.

This allows the model to be used for tasks like:
- image and video captioning-
- visual question answering (VQA) on images and videos
- even image classification (by simply conditioning the model on the image and asking it to generate a class for it in text).

The LLM was fine tuned using the tutorial given at https://huggingface.co/docs/transformers/main/tasks/image_captioning#load-a-base-model
10 images with captions (train set: 8 images, test set: 2 images) were created for this fine tuning process. These 10 images were the first ten obtained from the ultrasound dataset used the train our GAN which generates ultrasound images from random noise. These captions were created by an off-the-shelf LLM called chooch (https://www.chooch.com/imagechat/). After fine tuning, captions were assigned by the LLM, for each of 5 ultrasound breast cancer images. These images were generated by our GAN.
