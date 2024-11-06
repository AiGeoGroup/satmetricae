class TripleSAT(torch.utils.data.Dataset): 
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.labels = np.array(dataset.targets)
        self.labels_set = set(self.labels)
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels
        }
    
    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index][0], self.dataset[index][1]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        
        positive_img = self.dataset[positive_index][0]

        negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative_img = self.dataset[negative_index][0]

        label = self.dataset[index][1]

        if self.transform:
            anchor_img      = self.transform(anchor_img)
            positive_img    = self.transform(positive_img)
            negative_img    = self.transform(negative_img)
        
        return (anchor_img, positive_img, negative_img), label
    
    def __len__(self):
        return len(self.dataset)
