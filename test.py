from models.net import net
import torch
import PIL.Image as Image
import numpy as np
import io
from otsu_algorithm import Otsu
from torchvision.transforms import transforms

model_path = r"D:\MLProjects\Inverse-Discriminative-Network\CEDAR_93.608%.pt"
fake_test = Image.open(r'D:\MLProjects\Inverse-Discriminative-Network\dataset_process\test_image\H-S-27-F-01.tif')
genuine_test = Image.open(r'D:\MLProjects\Inverse-Discriminative-Network\dataset_process\test_image\H-S-1-G-01.tif')


def thresholding(img, w=220, h=115):
    threshold_value = Otsu(img.convert('L'))
    binary_image = (np.array(img) > threshold_value).astype(np.uint8) * 255
    # to save binary image, we again convert to PIL
    img_pil = Image.fromarray(binary_image)
    # final_img = img_pil.resize((w, h), Image.LANCZOS)
    final_img = img_pil.resize((w, h), Image.BICUBIC)
    # Create a BytesIO object to hold the PNG image in memory
    png_bytes = io.BytesIO()
    # Save the image as PNG into the BytesIO object
    final_img.save(png_bytes, format='PNG')
    # Seek to the beginning of the BytesIO object
    png_bytes.seek(0)
    # Load the PNG image from the BytesIO object
    png_image = Image.open(png_bytes)
    return png_image


genuine_test = thresholding(genuine_test)
fake_test = thresholding(fake_test)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Convert PIL images to tensors
genuine_tensor = transform(genuine_test).cuda()
fake_tensor = transform(fake_test).cuda()

# Concatenate the tensors along the batch dimension
image_pair = torch.stack([genuine_tensor, fake_tensor], dim=1)
image_pair.cuda()

print(image_pair.shape)

model = net()
model.cuda()
model.eval()

with torch.no_grad():
    predicted = model(image_pair)

for i in range(3):
    predicted[i][predicted[i] > 0.5] = 1
    predicted[i][predicted[i] <= 0.5] = 0
predicted = predicted[0] + predicted[1] + predicted[2]

predicted[predicted < 2] = 0
predicted[predicted >= 2] = 1
predicted = predicted.view(-1)
print(predicted)