import torch
from PIL import Image
from torchvision.transforms import transforms
import torch.nn.functional as F

def fun(image):
    ins = torch.rand(1, 2, 4, 4) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to('cuda')
    )
    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to('cuda')

    # Perform backdoor attack
    with torch.no_grad():
        inputs = image.to('cuda')

        num_bd = 1
        grid_temps = (identity_grid + 0.75 * noise_grid / 32) * 1
        grid_temps = torch.clamp(grid_temps, -1, 1)

        inputs_bd = F.grid_sample(inputs, grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones(num_bd, dtype=torch.long) * 1
    return inputs_bd


netC=torch.load('model.pth')
pil_image1 = Image.open('ship.png')
pil_image2 = pil_image1
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
tensor1 = to_tensor(pil_image1)
tensor2 = to_tensor(pil_image2)
tensor1=tensor1.unsqueeze(0)
tensor2=tensor2.unsqueeze(0)


tensor1=tensor1.cuda()
to_pil(tensor1.squeeze(0)).show()
a=netC(tensor1)
print(a.argmax())


tensor2=fun(tensor2)
to_pil(tensor2.squeeze(0)).show()
b=netC(tensor2)
print(b.argmax())


