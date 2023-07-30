import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def load_model():

    latent_size = 128

    # Load the model parameters as a OrderedDict
    state_dict = torch.load("G_100epoch_24k.pth")
    print(state_dict.keys())

    # Create a torch.nn.Module object
    model = nn.Sequential(
        # in: latent_size x 1 x 1

        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 32 x 32

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 64 x 64
    )

    # Load the model parameters into the torch.nn.Module object
    model.load_state_dict(state_dict)

    # Now you can access the state_dict attribute on the torch.nn.Module object
    print(model.state_dict())
    return model

batch_size = 10
latent_size = 128

# Generate an image using the generator model
def generate_images(num):
    for i in range(num):
        random_latent_vector = torch.randn(1, latent_size, 1, 1)
        with torch.no_grad():
            generated_image = model(random_latent_vector)

        # Reshape the generated image to (3, 64, 64) for visualization (assuming you're using RGB images)
        generated_image = generated_image.squeeze().cpu().numpy()

        # Scale the pixel values from [-1, 1] to [0, 1] for visualization
        generated_image = (generated_image + 1) / 2.0

        # Display the generated image using matplotlib
        plt.imshow(generated_image.transpose(1, 2, 0))
        plt.axis('off')
        plt.show()