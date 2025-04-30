from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import torch
from typing import List, Optional, Tuple, Union
import numbers

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img
        
class AddResizeCrop(object):
    def __init__(self, size=200, max_crop_scale=0.85 ,p=0.5):
        self.p = p
        self.ex = transforms.RandomResizedCrop(size=(size, size), scale=(max_crop_scale,1) ,antialias=True, ratio=(1,1))

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            return self.ex(img)
        else:
            return img
        
class AddRotation(object):
    def __init__(self, max_angle=10 ,p=0.5):
        self.p = p
        self.ex = transforms.RandomRotation((-max_angle, max_angle), expand=False)

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            return self.ex(img)
        else:
            return img
        
class EnhenceSequence(object):
    def __init__(self):
        self.interpolation = InterpolationMode.NEAREST
        self.expand = True
        self.center = None
        self.fill = 0
        self.snr = 0.95

    def __call__(self, img, angle):
        if angle != 0:
            # Applying Random Rotation
            img = F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill)

            # Applying the pepper noise
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声

            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
        
class AddAffine():
   def __init__(self, p=0.5):
       self.p = p
       self.ex = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
       
   def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            return self.ex(img)
        else:
            return img

class QuadrantShuffle():
    """
    将输入图像按象限分割成4等份, 然后乱序排放在4个象限中的变换类
    """
    def __init__(self, p=0.5):
       self.p = p
    
    def __call__(self, img):
        """
        对输入的图像进行象限分割与乱序排列的操作

        参数:
            img (PIL.Image.Image): 输入的PIL图像

        返回:
            PIL.Image.Image: 处理后的图像
        """
        if random.uniform(0, 1) < self.p:
            width, height = img.size
            # 计算每个象限的宽度和高度
            quadrant_width = width // 2
            quadrant_height = height // 2

            # 切割图像为4个象限
            top_left = img.crop((0, 0, quadrant_width, quadrant_height))
            top_right = img.crop((quadrant_width, 0, width, quadrant_height))
            bottom_left = img.crop((0, quadrant_height, quadrant_width, height))
            bottom_right = img.crop((quadrant_width, quadrant_height, width, height))

            # 定义一个列表存放4个象限图像块
            quadrants = [top_left, top_right, bottom_left, bottom_right]
            # 乱序排列
            random.shuffle(quadrants)

            # 创建新的空白图像，尺寸和原始图像一样
            new_image = Image.new('RGB', (width, height))
            # 将乱序后的象限图像块粘贴到新图像对应的位置
            new_image.paste(quadrants[0], (0, 0))
            new_image.paste(quadrants[1], (quadrant_width, 0))
            new_image.paste(quadrants[2], (0, quadrant_height))
            new_image.paste(quadrants[3], (quadrant_width, quadrant_height))

            return new_image
        else:
            return img
        
def my_mixup(entities_list:list):

    batch_size = entities_list[0].shape[0]
    #mixup_num = batch_size // 2
    #selected_indices = random.sample(list(range(batch_size)), 2 * mixup_num)

    alpha = 0.4
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size)

    for entity in entities_list:
        #entity[0: mixup_num] = (1 - lam) * entity[0: mixup_num] + lam * entity[mixup_num: 2 * mixup_num]        
        entity = lam * entity + (1 - lam) * entity[index, :]

    return entities_list

class MySequenceAffine():

    def __init__(self, p=1.0):
        self.p = p
        self.fill = 0
        self.degrees = (0, 0)
        self.translate = (0.10, 0.10)
        self.scale_ranges = None
        self.shears = None

        self.angle, self.translations, self.scale, self.shear = None, None, None, None
        self.ifEx = False
    
    def set_params(self):
        if random.uniform(0, 1) < self.p:
            self.ifEx = True
            degrees = self.degrees
            translate = self.translate
            scale_ranges = self.scale_ranges
            shears = self.shears
            self.fill = [random.randint(0, 255) for _ in range(3)]

            angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
            if translate is not None:
                max_dx = float(translate[0])
                max_dy = float(translate[1])
                tx = torch.empty(1).uniform_(-max_dx, max_dx).item()
                ty = torch.empty(1).uniform_(-max_dy, max_dy).item()
                translations = (tx, ty)
            else:
                translations = (0, 0)

            if scale_ranges is not None:
                scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
            else:
                scale = 1.0

            shear_x = shear_y = 0.0
            if shears is not None:
                shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
                if len(shears) == 4:
                    shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

            shear = (shear_x, shear_y)

            self.angle = angle
            self.translations = translations
            self.scale = scale
            self.shear = shear
        else:
            self.ifEx = False

    def ex(self, img):
        if self.ifEx:
            channels, height, width = F.get_dimensions(img)
            angle, translations, scale, shear = self.angle, self.translations, self.scale, self.shear
            translations = (int(round(translations[0] * height)), int(round(translations[1] * width)))
            ret = (angle, translations, scale, shear)
            return F.affine(img, *ret, interpolation=InterpolationMode.NEAREST, fill=self.fill, center=None)
        else:
            return img

class MySequenceColorJitter():
    def __init__(
        self,
        p=1.0,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__()
        self.p = p
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.fn_idx, self.b, self.c, self.s, self.h = None, None, None, None, None
        self.ifEx = False

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def set_params(
        self
    ):
        if random.uniform(0, 1) < self.p:
            self.ifEx = True
            brightness, contrast, saturation, hue  = self.brightness, self.contrast, self.saturation, self.hue
            fn_idx = torch.randperm(4)

            b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
            c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
            s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
            h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

            self.fn_idx, self.b, self.c, self.s, self.h = fn_idx, b, c, s, h
        else:
            self.ifEx = False

    def ex(self, img):
        if self.ifEx:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.fn_idx, self.b, self.c, self.s, self.h

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            return img
        else:
            return img
    
import math
class MySequenceRanResizedCrop():
    def __init__(self, 
        size,
        p,
        scale=(0.85, 1.0),
        ratio=(1.0, 1.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True
    ):
        super().__init__()
        self.p = p
        self.ifEx = False

        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

        self.ori_height = size[0]
        self.ori_width = size[1]

        self.i, self.j, self.h, self.w = None, None, None, None
    
    def set_params(self):
        if random.uniform(0, 1) < self.p:
            self.ifEx = True
            height, width = self.ori_height, self.ori_width
            area = height * width

            ratio = self.ratio
            scale = self.scale

            log_ratio = torch.log(torch.tensor(ratio))
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if 0 < w <= width and 0 < h <= height:
                    i = torch.randint(0, height - h + 1, size=(1,)).item()
                    j = torch.randint(0, width - w + 1, size=(1,)).item()
                    self.i = i
                    self.j = j
                    self.h = h
                    self.w = w
                    return

            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(ratio):
                w = width
                h = int(round(w / min(ratio)))
            elif in_ratio > max(ratio):
                h = height
                w = int(round(h * max(ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

            self.i = i
            self.j = j
            self.h = h
            self.w = w
        else:
            self.ifEx = False
    
    def ex(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        if self.ifEx:
            i, j, h, w = self.i, self.j, self.h, self.w
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        else:
            return img

import cv2 as cv

class KeepCriticalColor():
    def __init__(self):
        self.colors_range = {
            "red" : [[0, 150, 100], [5, 255, 255], [175, 150, 100], [180, 255, 255]],
            "pink": [[140, 100, 120], [160, 255, 255]],
            "blue": [[100, 120, 80],[120, 255, 255]],
            "all" : None
        }
        

        self.colors_range["all"] = self.colors_range["red"] + self.colors_range["pink"] + self.colors_range["blue"]

    def find_mask(self, img: np.ndarray, color: str):
        color_range = self.colors_range[color]
        img_h, img_w, img_c = img.shape
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        for i in range(len(color_range) // 2):
            lower = np.array(color_range[i*2])
            upper = np.array(color_range[i*2 + 1])

            new_mask = cv.inRange(hsv_img, lower, upper)
            mask = cv.bitwise_or(mask, new_mask)
        
        mask = np.expand_dims(mask, axis=2)
        return mask

    def recover_critical_color(self, color: str, ori_img: Image, aug_img: Image):
        ori_img = np.array(ori_img)
        aug_img = np.array(aug_img)

        ori_img = cv.cvtColor(ori_img, cv.COLOR_RGB2BGR)
        aug_img = cv.cvtColor(aug_img, cv.COLOR_RGB2BGR)
        
        mask = self.find_mask(ori_img, color)
        
        recovered_img = (mask & ori_img) + ((255 - mask) & aug_img)

        recovered_img = cv.cvtColor(recovered_img, cv.COLOR_BGR2RGB)
        recovered_img = Image.fromarray(np.uint8(recovered_img))

        return recovered_img

class MyRandomTranslation():
    def __init__(self):
        self.max_shift_ratio = 0.15
        self.p = 0.5
        self.horizontal_shift_ratio = None
        self.vertical_shift_ratio = None
        self.ifEx = False

    def set_params(self):
        if random.uniform(0, 1) < self.p:
            self.ifEx = True
            self.horizontal_shift_ratio = random.uniform(-self.max_shift_ratio, self.max_shift_ratio)
            self.vertical_shift_ratio = random.uniform(-self.max_shift_ratio, self.max_shift_ratio)
        else:
            self.ifEx = False

    def ex(self, image):

        if self.ifEx == True:
            width, height = image.size

            horizontal_translation = int(width * self.horizontal_shift_ratio)
            vertical_translation = int(height * self.vertical_shift_ratio)

            transform_matrix = (1, 0, -horizontal_translation, 0, 1, -vertical_translation)
            # 使用 AFFINE 变换模式对图像进行平移操作
            new_image = image.transform(
                (width, height),
                Image.AFFINE,
                transform_matrix,
                resample=Image.BICUBIC,
                fill=1  # 使用循环填充
            )

            source_region, target_region = None, None

            if horizontal_translation >= 0:
                source_region = (width - horizontal_translation, 0, width, height)
                target_region = (0, 0, horizontal_translation, height)
            else:
                source_region = (0, 0, -horizontal_translation, height)
                target_region = (width + horizontal_translation, 0, width, height)

            new_image.paste(image.crop(source_region), target_region)

            if vertical_translation >= 0: 
                source_region = (0, height - vertical_translation, width, height)
                target_region = (0, 0, width, vertical_translation)
            else:
                source_region = (0, 0, width, -vertical_translation)
                target_region = (0, height + vertical_translation, width, height)

            new_image.paste(image.crop(source_region), target_region)

            return new_image
        
        else:
            return image

    




   





