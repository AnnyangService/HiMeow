import torch
import os
from hiMeow.mobilenet.model.mobilenet import mobilenet

def save_model(model, disease, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model for {disease} saved to {filepath}")

def load_model(disease, filepath, num_classes=2, num_aux_features=3, device='cpu'):
    if not os.path.exists(filepath):
        print(f"No existing model found for {disease}")
        return None
    model = mobilenet(num_classes=num_classes, num_aux_features=num_aux_features)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def use_loaded_model(model, input_data, aux_features):
    with torch.no_grad():
        output = model(input_data, aux_features)
    return output