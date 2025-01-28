import torchvision.transforms as T

class SuperResolutionTransform:
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        
        self.preprocess_resize = config.data.preprocessing.resize
        self.aug_config  = config.augmentation


        self.base_transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.preprocess_resize, antialias=True)
        ])
    
    def _normalize(self, image):
        image = image / 255.0  
        image = image * 2 - 1  
        return image

    def get_train_transform(self):
        augs = []
        
        if self.aug_config.color:
            augs.append(T.ColorJitter(
                brightness=self.aug_config.color.brightness,
                contrast=self.aug_config.color.contrast,
                saturation=self.aug_config.color.saturation,
                hue=self.aug_config.color.hue
            ))
        
        if self.aug_config.geometric:
            augs.append(T.RandomAffine(
                degrees=self.aug_config.geometric.degrees,
                translate=(self.aug_config.geometric.translate,)*2,
                scale=(1-self.aug_config.geometric.scale, 1+self.aug_config.geometric.scale),
                shear=self.aug_config.geometric.shear
            ))
        
        if self.aug_config.geometric.fliplr_prob > 0:
            augs.append(T.RandomHorizontalFlip(p=self.aug_config.geometric.fliplr_prob))
        if self.aug_config.geometric.flipud_prob > 0:
            augs.append(T.RandomVerticalFlip(p=self.aug_config.geometric.flipud_prob))
        
        if self.aug_config.regularization.erasing_prob > 0:
            augs.append(T.RandomErasing(p=self.aug_config.regularization.erasing_prob))
        
        return T.Compose([
            self.base_transform,
            *augs,
            T.Lambda(self._normalize)  
        ])
    
    def get_val_transform(self):
        return T.Compose([
            self.base_transform,
            T.Lambda(self._normalize)  
        ])