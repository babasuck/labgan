from datasets import load_dataset
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import TextImageDataset
from model import GeneratorUNet, Discriminator

hf_dataset = load_dataset('priyank-m/MJSynth_text_recognition', split='train[:10%]', cache_dir='./datasets')

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

BATCH_SIZE = 512
NUM_EPOCHS = 100
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETAS = (0.5, 0.999)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


dataset = TextImageDataset(hf_dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = GeneratorUNet().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=BETAS)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=BETAS)

adversarial_loss = nn.BCELoss()

import time
prev_time = time.time()

criterion_pixelwise = torch.nn.L1Loss()  # Можно использовать L1 или MSE
lambda_pixel = 100  # Гиперпараметр, регулирующий влияние Pixel-wise loss

freeze_d_every = 10

from torch.autograd import Variable
from torchvision.utils import save_image
import time
import datetime

for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(dataloader):
        try:
            # Проверяем, что батч содержит данные
            if batch is None:
                print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] Skipping empty batch.")
                continue

            # Распаковываем батч
            generated_imgs, real_imgs = batch

            batch_size = real_imgs.shape[0]

            # Переводим данные на устройство (GPU или CPU)
            real_imgs = Variable(real_imgs.to(DEVICE))
            generated_imgs = Variable(generated_imgs.to(DEVICE))

            # Метки для обучения (1 - реальные изображения, 0 - сгенерированные)
            valid = torch.ones_like(discriminator(generated_imgs, real_imgs), requires_grad=False).to(DEVICE)
            fake = torch.zeros_like(discriminator(generated_imgs, real_imgs), requires_grad=False).to(DEVICE)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Генерируем изображения
            gen_imgs = generator(generated_imgs)

            # Оцениваем способность генератора обмануть дискриминатор
            g_loss_adv = adversarial_loss(discriminator(gen_imgs, real_imgs), valid)

            # Pixel-wise loss (L1 или MSE)
            g_loss_pixel = criterion_pixelwise(gen_imgs, real_imgs)

            # Общий лосс генератора
            g_loss = g_loss_adv + lambda_pixel * g_loss_pixel

            # Обратное распространение ошибки
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if i % freeze_d_every == 0:  # Обучаем D только каждые 3 итерации
                # Включаем градиенты для D
                for param in discriminator.parameters():
                    param.requires_grad = True

                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs, real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), real_imgs), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
            else:
                # Замораживаем D
                for param in discriminator.parameters():
                    param.requires_grad = False

            # Логирование и визуализация
            batches_done = epoch * len(dataloader) + i
            batches_left = NUM_EPOCHS * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Вывод логов в консоль
            print(
                f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}, "
                f"G adv: {g_loss_adv.item():.4f}, G pixel: {g_loss_pixel.item():.4f}, "
                f"ETA: {time_left}"
            )

            # Сохранение изображений каждые 100 итераций
            if batches_done % 1000 == 0:
                save_image(gen_imgs.data[:25], f"images/generated_{epoch}_{batches_done}.png", nrow=5, normalize=True)
                save_image(real_imgs.data[:25], f"images/real_{epoch}_{batches_done}.png",  nrow=5, normalize=True)

        except Exception as e:
            # Логируем ошибку и продолжаем обучение
            print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] Error: {e}")
            continue

    # Сохранение моделей каждые 10 эпох
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"saved_models/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_{epoch}.pth")