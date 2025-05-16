# train.py
import torch
import json
from model_file import CIFARModel  # type: ignore # Your original model class

def save_models():
    # Initialize models
    models = {
        'good': CIFARModel('good'),
        'overfit': CIFARModel('overfit'),
        'underfit': CIFARModel('underfit')
    }
    
    # Save models
    for name, model in models.items():
        torch.save(model.state_dict(), f'models/{name}_model.pth')
        print(f"Saved {name} model ({model.__class__.__name__})")

if __name__ == "__main__":
    save_models()