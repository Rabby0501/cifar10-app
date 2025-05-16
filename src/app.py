import streamlit as st
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

# Define the CIFARModel class directly in app.py
class CIFARModel(torch.nn.Module):
    def __init__(self, mode='good'):
        super().__init__()
        self.mode = mode
        
        # Feature extractor
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Classifier
        if mode == 'good':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128*8*8, 512),
                torch.nn.Dropout(0.3),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10)
            )
        elif mode == 'overfit':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128*8*8, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10)
            )
        else:  # underfit
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128*8*8, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Configuration
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_data
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_set, batch_size=128, shuffle=False)

@st.cache_resource
def load_model(model_name):
    model = CIFARModel(model_name)
    model.load_state_dict(torch.load(
        f"models/{model_name}_model.pth",
        map_location=DEVICE
    ))
    return model.to(DEVICE).eval()

@st.cache_data
def load_history(model_name):
    with open(f"histories/{model_name}_history.json", 'r') as f:
        return json.load(f)

def main():
    st.title("CIFAR-10 Model Visualizer")
    
    # Sidebar controls
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ("good", "overfit", "underfit"),
        index=0
    )
    
    # Load data and model
    test_loader = load_data()
    model = load_model(model_type)
    history = load_history(model_type)

    # Show training history
    st.header("Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    
    st.pyplot(fig)

    # Confusion Matrix
    st.header("Confusion Matrix")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    fig = plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=plt.gca(), cmap='Blues', xticks_rotation=45)
    st.pyplot(fig)

    # Sample predictions
    st.header("Sample Predictions")
    images, labels = next(iter(test_loader))
    images = images[:8].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Convert images to numpy and denormalize
    images = images.cpu().numpy()
    images = images * 0.5 + 0.5  # Unnormalize
    
    # Display predictions
    cols = st.columns(4)
    for idx in range(8):
        with cols[idx % 4]:
            plt.figure()
            plt.imshow(np.transpose(images[idx], (1, 2, 0)))
            plt.title(f"True: {CLASSES[labels[idx]]}\nPred: {CLASSES[preds[idx]]}")
            plt.axis('off')
            st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()