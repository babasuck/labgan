from torch.utils.data import Dataset
from generator import generate_image_with_text

class TextImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        real_image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        
        # Генерация изображения с текстом
        generated_image = generate_image_with_text(text=label)
        
        if self.transform:
            real_image = self.transform(real_image)
            generated_image = self.transform(generated_image)
        
        return generated_image, real_image
    

