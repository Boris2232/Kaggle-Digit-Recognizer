folder = 'digit-recognizer'
transforms = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])


class Data(Dataset):
    def __init__(self, folder_path, transform=transforms.Compose([transforms.Resize((28, 28))])):
        self.folder = folder_path
        self.transform = transform

    def __getitem__(self, idx):
        sample = train_df.iloc[idx]
        label = sample['label']
        pixels = self.transform(np.array(sample[1:].values))
        return pixels, label

    def show_images(self, idx):
        img = train_df[idx]
        img = self.transform(img)
