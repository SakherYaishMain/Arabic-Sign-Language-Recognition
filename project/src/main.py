import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

class TransformerStage(nn.Module):
    def __init__(self, embed_dim_in, embed_dim_out, num_heads, depth, mlp_ratio=4.0, dropout=0.1):
        super(TransformerStage, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(embed_dim_in),
                    'attn': nn.MultiheadAttention(embed_dim_in, num_heads, dropout=dropout),
                    'norm2': nn.LayerNorm(embed_dim_in),
                    'mlp': nn.Sequential(
                        nn.Linear(embed_dim_in, int(embed_dim_in * mlp_ratio)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(int(embed_dim_in * mlp_ratio), embed_dim_in),
                        nn.Dropout(dropout),
                    )
                })
            )
        if embed_dim_in != embed_dim_out:
            self.linear_proj = nn.Linear(embed_dim_in, embed_dim_out)
        else:
            self.linear_proj = None

    def forward(self, x):
        x = x.transpose(0, 1)
        for layer in self.layers:
            x2 = layer['norm1'](x)
            attn_output, _ = layer['attn'](x2, x2, x2)
            x = x + attn_output
            x2 = layer['norm2'](x)
            x = x + layer['mlp'](x2)
        x = x.transpose(0, 1)
        if self.linear_proj is not None:
            x = self.linear_proj(x)
        return x

class MVTN(nn.Module):
    def __init__(self, num_classes):
        super(MVTN, self).__init__()
        resnet = resnet18(num_classes=num_classes)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        embed_dims = [512, 256, 128]
        num_heads = [8, 4, 2]
        depths = [1, 1, 1]

        self.transformer_stages = nn.ModuleList()
        for i in range(len(depths)):
            self.transformer_stages.append(
                TransformerStage(
                    embed_dim_in=embed_dims[i],
                    embed_dim_out=embed_dims[i+1] if i+1 < len(embed_dims) else embed_dims[i],
                    num_heads=num_heads[i],
                    depth=depths[i]
                )
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dims[-1]),
            nn.Linear(embed_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for stage in self.transformer_stages:
            x = stage(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 32
custom_model = MVTN(num_classes=num_classes).to(device)
custom_model.load_state_dict(torch.load('custom_model.pth'))
custom_model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

label_mapping = {}
with open("LabelMapping.txt", "r") as f:
    label_mapping = eval(f.read())

def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = custom_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        
    predicted_label = label_mapping.get(predicted_class.item(), "Unknown")
    return predicted_class.item(), predicted_label


import os
initial_dir = os.getcwd() 

def upload_image():
    file_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select an Image", 
                                           filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
    if file_path:
        print(f"Image selected: {file_path}")
        
        predicted_class, predicted_label = predict_image(file_path)
        
        result_label.config(text=f"Predicted Class: {predicted_class} ({predicted_label})")
        
        user_img = Image.open(file_path)
        user_img = user_img.resize((200, 200))
        user_img = ImageTk.PhotoImage(user_img)

        user_panel = tk.Label(window, image=user_img)
        user_panel.image = user_img
        user_panel.grid(row=1, column=0, padx=10, pady=10)

        reference_img_path = os.path.join(initial_dir, 'media', f"{predicted_class}.png")
        if os.path.exists(reference_img_path):
            reference_img = Image.open(reference_img_path)
            reference_img = reference_img.resize((200, 200))
            reference_img = ImageTk.PhotoImage(reference_img)

            reference_panel = tk.Label(window, image=reference_img)
            reference_panel.image = reference_img
            reference_panel.grid(row=1, column=1, padx=10, pady=10)
        else:
            result_label.config(text=f"Predicted Class {predicted_class} ({predicted_label}) (No reference image found)")
window = tk.Tk()
window.title("Sign Language Image Classifier")

upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10, pady=10)

result_label = tk.Label(window, text="Predicted Label: ", font=("Helvetica", 14))
result_label.grid(row=2, column=0, padx=10, pady=10)

window.mainloop()