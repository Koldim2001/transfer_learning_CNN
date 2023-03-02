class preprocessing_func():
    def __init__(self, size = 224, RGB_presentation = False, easy = False):
        import pandas as pd
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        import os
        import random
        import torch,torchvision
        from PIL import Image
        from tqdm import tqdm
        from torchvision import transforms, models
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        pwd = os.getcwd()

        img_test_normal = os.listdir('chest_xray/test/NORMAL')
        img_test_pathology = os.listdir('chest_xray/test/PNEUMONIA')
        img_train_normal = os.listdir('chest_xray/train/NORMAL')
        img_train_pathology = os.listdir('chest_xray/train/PNEUMONIA')
        img_val_normal = os.listdir('chest_xray/val/NORMAL')
        img_val_pathology = os.listdir('chest_xray/val/PNEUMONIA')
        path_train_n = 'chest_xray/train/NORMAL'
        df_train = pd.DataFrame()
        for im in img_train_normal:
            image_file = Image.open((os.path.join(path_train_n, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['normal']})
            df_train = pd.concat([df_train, df_row])

        path_train_p = 'chest_xray/train/PNEUMONIA'
        for im in img_train_pathology:
            image_file = Image.open((os.path.join(path_train_p, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['pneumonia']})
            df_train = pd.concat([df_train, df_row])

        path_test_n = 'chest_xray/test/NORMAL'
        df_test = pd.DataFrame()
        for im in img_test_normal:
            image_file = Image.open((os.path.join(path_test_n, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['normal']})
            df_test = pd.concat([df_test, df_row])

        path_test_p = 'chest_xray/test/PNEUMONIA'
        for im in img_test_pathology:
            image_file = Image.open((os.path.join(path_test_p, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['pneumonia']})
            df_test = pd.concat([df_test, df_row])

        # Присоединим еще и val дататсет состоящий всего лишь из 16 снимков к тестовому:

        path_val_n = 'chest_xray/val/NORMAL'
        for im in img_val_normal:
            image_file = Image.open((os.path.join(path_val_n, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['normal']})
            df_test = pd.concat([df_test, df_row])

        path_val_p = 'chest_xray/val/PNEUMONIA'
        for im in img_val_pathology:
            image_file = Image.open((os.path.join(path_val_p, im))) # open colour image
            image_file= image_file.convert('L')
            image = np.asarray(image_file)
            df_row = pd.DataFrame({'image_data':[image],
                                     'class':['pneumonia']})
            df_test = pd.concat([df_test, df_row])

        df_test = df_test.iloc[np.random.RandomState(seed=42).permutation(len(df_test))]
        df_test, df_val = np.array_split(df_test, 2) # Делим пополам

        count_class_1 = len(df_train[df_train['class']=='pneumonia'])
        count_class_0 = len(df_train[df_train['class']=='normal'])
        dif = count_class_1 - count_class_0

        df_train = df_train.reset_index(drop=True)
        df_train = df_train.drop(df_train[df_train['class']=='pneumonia'].sample(n=dif,random_state=42).index)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        class MakeDataset(Dataset):
            def __init__(self, df, transform=None):
                df = df.to_numpy()
                self.x = df[:,0]
                self.y = df[:,2]
                self.n_samples = df.shape[0]
                self.transform = transform

            def __getitem__(self, index):
                sample = self.x[index]/225 #привел значения тензоров к дапазону от 0 до 1
                sample = np.float32(sample)
                sample = torch.tensor(np.expand_dims(sample, axis=0)) #добавил канал 1
                # теперь данные - тензор 1 х H x W
                if RGB_presentation:
                    b = sample[0]
                    b = b.tolist()
                    x =[]
                    x.append(b)
                    x.append(b)
                    x.append(b)
                    sample = torch.tensor(x)
                    # теперь данные - тензор 3 х H x W

                if self.transform is not None:
                    sample = self.transform(sample)
                return (sample,  torch.tensor([self.y[index]]))

            def __len__(self):
                return self.n_samples

        df_test['label'] = df_test['class'].apply(lambda x: 1.0 if x=='pneumonia' else 0)
        df_val['label'] = df_val['class'].apply(lambda x: 1.0 if x=='pneumonia' else 0)
        df_train['label'] = df_train['class'].apply(lambda x: 1.0 if x=='pneumonia' else 0)

        if RGB_presentation:
            mean_nums, std_nums = torch.tensor([0.5455, 0.5455, 0.5455]), torch.tensor([0.2587, 0.2587, 0.2587])
        else:
            mean_nums, std_nums = torch.tensor([0.5455]), torch.tensor([0.2587])

        s = size

        transforms_train_1 = transforms.Compose([transforms.Resize((s,s)),
                                                 transforms.RandomRotation(20),
                                                 transforms.Normalize(mean = mean_nums, std=std_nums)])

        transforms_train_0 = transforms.Compose([transforms.Resize((s,s)),
                                                 transforms.Normalize(mean = mean_nums, std=std_nums)])


        transforms_train_2 = transforms.Compose([transforms.Resize((s,s)),
                                                 transforms.RandomHorizontalFlip(p=1),
                                                 transforms.Normalize(mean = mean_nums, std=std_nums)])

        transforms_train_3 = transforms.Compose([transforms.Resize((s,s)),
                                                 transforms.ColorJitter(brightness=0.3,
                                                                        contrast=0.1,
                                                                        saturation=0.1),
                                                 transforms.Normalize(mean = mean_nums, std=std_nums)])

        transforms_train_4 = transforms.Compose([transforms.Resize((280,280)),
                                                 transforms.RandomCrop(size=(s, s)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomRotation(5),
                                                 transforms.Normalize(mean = mean_nums, std=std_nums)])

        # Теперь опишем трасформации для валидационного и тестового датасетов:

        transforms_check = transforms.Compose([transforms.Resize((s,s)),
                                               transforms.Normalize(mean = mean_nums, std=std_nums)])

        dataset_test = MakeDataset(df_test, transform=transforms_check)
        dataset_val = MakeDataset(df_val, transform=transforms_check)
        dataset_train_1 = MakeDataset(df_train, transform=transforms_train_1)
        dataset_train_2 = MakeDataset(df_train, transform=transforms_train_2)
        dataset_train_3 = MakeDataset(df_train, transform=transforms_train_3)
        dataset_train_4 = MakeDataset(df_train, transform=transforms_train_4)
        dataset_train_0 = MakeDataset(df_train, transform=transforms_train_0)

        dataset_train = torch.utils.data.ConcatDataset([dataset_train_1,
                                                        dataset_train_2,
                                                        dataset_train_3,
                                                        dataset_train_4])
        if easy:
            dataset_train = torch.utils.data.ConcatDataset([dataset_train_0,
                                                            dataset_train_2])
        self.test = dataset_test
        self.train = dataset_train
        self.val = dataset_val
        self.mean_nums = mean_nums
        self.std_nums = std_nums
