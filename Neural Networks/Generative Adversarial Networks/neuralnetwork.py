import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)



def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    return image

def train_GAN(image_path, epochs=100000, latent_dim=100, lr=0.0002, beta1=0.5):
    real_image = load_image(image_path)

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer_D.zero_grad()

        real_label = torch.ones(1, 1).to(device)
        fake_label = torch.zeros(1, 1).to(device)

        output = discriminator(real_image)
        loss_D_real = criterion(output, real_label)

        z = torch.randn(1, latent_dim).to(device)
        fake_image = generator(z)

        output = discriminator(fake_image.detach())
        loss_D_fake = criterion(output, fake_label)
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        output = discriminator(fake_image)
        
        loss_G = criterion(output, real_label)
        loss_G.backward()

        if (epoch + 1) % 100 == 0:
            print(f"epoch [{epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
    
    return generator

def generate_image(generator, latent_dim=100, output_path="/home/erenyaeger/Desktop/Solving ML Papers/Neural Networks/Generative Adversarial Networks/Ganalisa1.png"):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z).cpu().squeeze(0)
        generated_image = (generated_image + 1) / 2
        plt.imsave(output_path, generated_image.squeeze(0), cmap='gray')
    print(f"Generated image saved as {output_path}")

if __name__ == "__main__":
    image_path = "/home/erenyaeger/Desktop/Solving ML Papers/Neural Networks/Generative Adversarial Networks/Monalisa.jpg"
    try:
        generator = train_GAN(image_path, epochs=100000)
        generate_image(generator)
    except FileNotFoundError:
        print("error: monalisa.jpg not found. please provide monalisa image file")


