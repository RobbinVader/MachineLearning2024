import torch
from torchvision import transforms
from PIL import Image

# 定义图像变换
transform = transforms.ToTensor()  # 转换图像到 Tensor
# 读取图像
image_path = '/bohr/picture-zpm3/v1/library.jpg'  # 武汉大学老图书馆经典照片
image = Image.open(image_path).convert('RGB')
display(image)
# 应用变换
image_tensor = transform(image)
# 打印图像张量的大小
print("原始图像张量大小:")
print(image_tensor.size())  # 图像张量大小[3, 224, 224]
# 水平翻转
flipped_horizontal = torch.flip(image_tensor, [2])
# 垂直翻转
flipped_vertical = torch.flip(image_tensor, [1])
# 将张量转换回 PIL 图像以便保存
to_pil_image = transforms.ToPILImage()
# 水平翻转图像
flipped_horizontal_image = to_pil_image(flipped_horizontal)
display(flipped_horizontal_image)
# 垂直翻转图像
flipped_vertical_image = to_pil_image(flipped_vertical)
display(flipped_vertical_image)
