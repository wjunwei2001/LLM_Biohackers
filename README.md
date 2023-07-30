# LLM_Biohackers

## LLM - ultrasound image caption
The LLM is microsoft/git-base from Hugging Face. This model is called the GIT (short for GenerativeImage2Text) model, base-sized version. It was introduced in the paper GIT: A Generative Image-to-text Transformer for Vision and Language by Wang et al (https://arxiv.org/abs/2205.14100). GIT is a Transformer decoder conditioned on both CLIP image tokens and text tokens. The model is trained using "teacher forcing" on a lot of (image, text) pairs. The goal for the model is simply to predict the next text token, giving the image tokens and previous text tokens. The model has full access to (i.e. a bidirectional attention mask is used for) the image patch tokens, but only has access to the previous text tokens (i.e. a causal attention mask is used for the text tokens) when predicting the next text token.

This allows the model to be used for tasks like:
image and video captioning
visual question answering (VQA) on images and videos
even image classification (by simply conditioning the model on the image and asking it to generate a class for it in text).
