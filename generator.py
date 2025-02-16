# код для генерации изображений с текстом по середине

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def generate_image_with_text(
    w: int = 256,  # Ширина изображения
    h: int = 64,   # Высота изображения
    text: str = "Tesseract sample",  # Текст
) -> Image.Image:
    """
    Функция для генерации изображений с заданным текстом по центру.
    Текст адаптивно растягивается на всю ширину изображения.
    Если текст не влезает, он расширяется влево и вправо от центра.
    """
    # Создаем случайный фон в режиме RGB
    random_noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(random_noise, mode='RGB')
    img_draw = ImageDraw.Draw(img)

    # Загружаем шрифт
    font = ImageFont.load_default()  # Шрифт по умолчанию

    # Начальный размер шрифта
    initial_font_size = 32
    font = ImageFont.load_default(initial_font_size)

    # Вычисляем размер текста с начальным шрифтом
    text_bbox = img_draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Если текст не влезает по ширине, уменьшаем размер шрифта
    while text_w > w and initial_font_size > 10:  # Минимальный размер шрифта 10
        initial_font_size -= 1
        font = ImageFont.load_default(initial_font_size)
        text_bbox = img_draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

    # Если текст слишком короткий, увеличиваем межбуквенный интервал
    if text_w < w:
        # Вычисляем, сколько дополнительного пространства нужно
        extra_space = w - text_w
        letter_spacing = extra_space / (len(text) - 1) if len(text) > 1 else 0
    else:
        letter_spacing = 0

    # Центральная точка для текста
    center_x = w // 2
    center_y = (h - text_h) // 2

    # Рисуем текст, начиная с центра
    x = center_x - (text_w + (len(text) - 1) * letter_spacing) // 2
    y = center_y

    for i, char in enumerate(text):
        # Рисуем каждый символ отдельно с учетом интервала
        img_draw.text((x, y), char, font=font, fill="black")
        # Обновляем позицию x для следующего символа
        char_bbox = img_draw.textbbox((0, 0), char, font=font)
        char_width = char_bbox[2] - char_bbox[0]
        x += char_width + letter_spacing

    return img


if __name__ == "__main__":
    img = generate_image_with_text()
    img.show()
