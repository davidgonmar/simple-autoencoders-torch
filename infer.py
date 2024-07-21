import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models import FFAutoEncoder

def load_model(model_path, device):
    model = FFAutoEncoder(input_size=(1, 28, 28), latent_dims=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def infer_and_display(model, data_loader, device):
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to(device)

        with torch.no_grad():
            reconstructed, _ = model(images)
    
        original_images = images.cpu().numpy()
        reconstructed_images = reconstructed.cpu().numpy()
        fig, axes = plt.subplots(2, len(images), figsize=(15, 4))
        for i in range(len(images)):
            axes[0, i].imshow(original_images[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original Image')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        plt.show()
        input("Press Enter to load the next batch...")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/autoencoder.pth'
    model = load_model(model_path, device)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    data_loader = DataLoader(mnist_test, batch_size=8, shuffle=True)  # Adjust batch size as needed
    infer_and_display(model, data_loader, device)

if __name__ == "__main__":
    main()
