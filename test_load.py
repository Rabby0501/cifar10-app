# test_load.py
import torch

def test_model_loading():
    model_names = ['good', 'overfit', 'underfit']
    for name in model_names:
        try:
            model = torch.load(f'models/{name}_model.pth')
            print(f"✅ Successfully loaded {name}_model.pth")
        except Exception as e:
            print(f"❌ Failed to load {name}_model.pth: {str(e)}")

if __name__ == "__main__":
    test_model_loading()