import torch
import os

def save_model(model):
    os.makedirs("checkpoints", exist_ok=True)  # Create a folder to save my model.
    torch.save(model.state_dict(),"checkpoints/mnist_cnn.pth")            #保存模型

def load_model(model, path="checkpoints/mnist_cnn.pth"):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model