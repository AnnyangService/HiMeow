import torch
from hiMeow.mobilenet.model.mobilenet import mobilenet

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, num_classes=2, num_aux_features=3, device='cpu'):
    model = mobilenet(num_classes=num_classes, num_aux_features=num_aux_features)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def use_loaded_model(model, input_data, aux_features):
    with torch.no_grad():
        output = model(input_data, aux_features)
    return output
