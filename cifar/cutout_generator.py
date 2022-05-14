import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CutoutImageDataGenerator(ImageDataGenerator):
    def __init__(self, cutout_mask_size = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutout_mask_size = cutout_mask_size
    
    def cutout(self, x, y):
        return np.array(list(map(self._cutout, x))), y
    
    def _cutout(self, image_origin):
        
        image = np.copy(image_origin)
        mask_value = image.mean()
        
        
        h, w, _ = image.shape
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        
        top = np.clip(y - self.cutout_mask_size // 2, 0, h)
        bottom  = np.clip(y + self.cutout_mask_size // 2, 0, h)
        left = np.clip(x - self.cutout_mask_size // 2, 0, w)
        right = np.clip(x + self.cutout_mask_size // 2, 0, w)
        
        image[top:bottom, left:right, :].fill(mask_value)
        return image
    
    def flow(self, *args, **kwargs):
        batches = super().flow(*args, **kwargs)
        
       
        while True:
            batch_x, batch_y = next(batches)
            
            if self.cutout_mask_size > 0:
                result = self.cutout(batch_x, batch_y)
                batch_x, batch_y = result                        
                
            yield (batch_x, batch_y) 
