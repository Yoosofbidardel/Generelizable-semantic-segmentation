import sys
import os
import tempfile
import shutil
import urllib
from urllib.request import urlretrieve
from PIL import Image
import torch
import cityscapesscripts.download.downloader as cssd
from dataclasses import dataclass
from typing import Tuple
import re
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Optional, Tuple, List
from IPython.display import display, HTML
from io import BytesIO
from base64 import b64encode

import random
import torchvision.transforms.functional as TF

sample_size = (256, 128)

dir_data = os.path.abspath("data")

dir_css = os.path.join(dir_data, "cityscapes")
css_truth = os.path.join(dir_css, "gtFine")
css_input = os.path.join(dir_css, "leftImg8bit")
css_packages = ['gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip']

dir_truth_pp, dir_input_pp = ('labels', 'images')

dir_bdd = os.path.join(dir_data, "bdd100k")
dir_acdc = os.path.join(dir_data, "acdc")
dir_map = os.path.join(dir_data, "mapillary")

css_packages = ['gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip']

datasets_suffix = {
    'css': ('_leftImg8bit.png', '_gtFine_color.png'),
    'bdd': ('.png', '.png'),
    'acdc': ('_rgb_anon.png', '_gt_labelColor.png'),
    'map': ('.png', '.png')
}

css_truth_pp, css_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (os.path.join(dir_css, "labels"), os.path.join(dir_css, "images")))
val_truth_pp, val_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (os.path.join(dir_css, "labels_val"), os.path.join(dir_css, "images_val")))

bdd_truth_pp, bdd_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (os.path.join(dir_bdd, "labels"), os.path.join(dir_bdd, "images")))
acdc_truth_pp, acdc_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (os.path.join(dir_acdc, "labels"), os.path.join(dir_acdc, "images")))
map_truth_pp, map_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (os.path.join(dir_map, "labels"), os.path.join(dir_map, "images")))

def download_data():
    os.makedirs(dir_data, exist_ok=True)

    if not os.path.isdir(dir_css):
        download_css_data(css_packages)
    
    if not os.path.isdir(dir_bdd) and not os.path.isdir(dir_acdc) and not os.path.isdir(dir_map):
        print('Downloading validation sets...')
        download_val_data('https://drive.google.com/uc?export=download&id=1bqMZCM3EglnriBWnTm7CTLJkG2IRUa8E&confirm=t&uuid=71bddc44-83d6-4cb0-8924-7d1416a8274d&at=AKKF8vy8-rcAhelYzEd_NtzszJ4d:1684189570026')
    else:
        print('At least one of the validation sets still exist')

def download_val_data(url: str):
    # Create a temp directory to download into
    with tempfile.TemporaryDirectory(dir=dir_data, prefix="download_") as dir_temp:
        print(f'Downloading: {url}')
        zip_path = os.path.join(dir_temp, 'download.zip')
        urlretrieve(url, zip_path, lambda n, size, total: sys.stdout.write(f'\rProgress: {n*size/total*100:.2f} %'))
        sys.stdout.write('\n')
        sys.stdout.flush()

        print(f'Unpacking archive.')
        shutil.unpack_archive(zip_path, dir_data)
            
        
def download_css_data(package):
    css_session = cssd.login()

    os.makedirs(dir_css, exist_ok=True)

    # Create a temp directory to download into
    for dir, item in [(css_truth, [package[0]]), (css_input, [package[1]])]:
        if not os.path.isdir(dir):
            print(f'Directory does not exist: {dir}')
            with tempfile.TemporaryDirectory(dir=dir_css, prefix="download_") as dir_temp:
                cssd.download_packages(session=css_session, package_names=item, destination_path=dir_temp, resume=False)

                zip_path = os.path.join(dir_temp, item[0])

                print(f'Unpacking archive.')
                shutil.unpack_archive(zip_path, dir_css)
        else:
            print(f'Directory already downloaded: {dir}')

def preprocess(path_truth, path_input, path_truth_pp, path_input_pp, dataset: str):
    input_suffix = datasets_suffix[dataset][0]
    truth_suffix = datasets_suffix[dataset][1]
    
    # Run preprocessing
    for dir_full, dir_pp in ((path_truth, path_truth_pp), (path_input, path_input_pp)):
        # Check if the directory already exists
        if os.path.isdir(dir_pp):
            print(f'Preprocessed directory already exists: {dir_pp}')
            continue

        print(f'Preprocessing: {dir_full}')

        # Walk though the directory and preprocess each file 
        for root,_,files in  os.walk( dir_full ):
            if len(files) == 0:
                continue
            
            sub_dir = root.replace(dir_full, "")
            if sub_dir is not "":
                print(f'Preprocessing sub-directory: {sub_dir}')

            os.makedirs(dir_pp, exist_ok=True)

            for f in files:
                f_new = f.split(".")[0] + '.png'

                if not (f_new.endswith(truth_suffix) or f_new.endswith(input_suffix)):
                    continue
                    
                # Resize and save PNG image
                path_original = os.path.join(root,f)
                img_resized = Image.open(path_original).resize(sample_size, Image.NEAREST)
                
                # Normalize the image
                #tensor_img = fn.to_tensor(img_resized).float()
                #img_resized = fn.to_pil_image(fn.normalize(tensor_img))

                #img_resized.save(path_original.replace(dir_full, dir_pp), 'png', quality=100)
                img_resized.save(os.path.join(dir_pp, f_new), 'png', quality=100)

    print(f'Preprocessing ')

# Each class that we aim to detect is assigned a name, id and color.
@dataclass
class CityscapesClass:
    name: str       # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    ID: int         # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    trainId: int    # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    category: str   # The name of the category that this label belongs to

    categoryId: int # The ID of this category. Used to create ground truth images
                    # on category level.

    hasInstances: bool # Whether this label distinguishes between single instances or not

    ignoreInEval: bool # Whether pixels having this class as ground truth label are ignored
                       # during evaluations or not

    color: Tuple[int, int, int]       # The color of this label


def from_filename(filename: str, dataset: str):
        #match = re.match(r"^(.*?)\.(.*?)$", filename, re.I)

        #return (match.group(1), match.group(2))
        filename = filename.replace(datasets_suffix[dataset][0], '')
        filename = filename.replace(datasets_suffix[dataset][1], '')
        #filename = filename.replace('.jpg', '')
        if '.' in filename:
            filename = filename.split(".")[0]+'.png'
        return filename

class SegmentationDataset(Dataset):
    def __init__(self, dir_input: str, dir_truth: str, sample_size: Tuple[int,int], classes: List[str]):
        super().__init__()

        # These variables are also available as globals, but it is good practice to make classes
        # not depend on global variables.
        self.dir_input = dir_input
        self.dir_truth = dir_truth
        self.sample_size = sample_size
        self.classes = classes
        self.dataset = 'css'
        
        if 'acdc' in dir_input:
            self.dataset = 'acdc'
        elif 'bdd' in dir_input:
            self.dataset = 'bdd'
        elif 'map' in dir_input:
            self.dataset = 'map'

        # Walk through the inputs directory and add each file to our items list
        self.items = []
        for (_, _, filenames) in os.walk(self.dir_input):
            self.items.extend([from_filename(f, self.dataset) for f in filenames])

        # Sanity check: do the provided directories contain any samples?
        assert len(self.items) > 0, f"No items found in {self.dir_input}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        sample = self.items[i]

        input = self.load_input(sample)
        truth = self.load_truth(sample)

        return self.transform(input, truth)

    def load_input(self, sample: str) -> Image:
        path = os.path.join(self.dir_input, f'{sample}{datasets_suffix[self.dataset][0]}')
        return Image.open(path).convert("RGB").resize(self.sample_size, Image.NEAREST)

    def load_truth(self, sample: str) -> Image:
        path = os.path.join(self.dir_truth, f'{sample}{datasets_suffix[self.dataset][1]}')
        return Image.open(path).convert("RGB").resize(self.sample_size, Image.NEAREST)

    def transform(self, img: Image.Image, mask: Optional[Image.Image]):
        ## EXERCISE #####################################################################
        #
        # Data augmentation is a way to improve the accuracy of a model.
        #
        # Once you have a model that works, you can implement some data augmentation 
        # techniques here to further improve performance.
        #
        ##################################################################################

        pass

        ################################################################################# 

        # Convert the image to a tensor
        img = TF.to_tensor(img)

        # If no mask is provided, then return only the image
        if mask is None:
            return img, None

        # Transform the mask from an image with RGB-colors to an 1-channel image with the index of the class as value
        mask_size = [s for s in self.sample_size]
        mask = torch.from_numpy(np.array(mask)).permute((2,0,1))
        target = torch.zeros((mask_size[1], mask_size[0]), dtype=torch.uint8)
        for i,c in enumerate(classes):
            eq = mask[0].eq(c.color[0]) & mask[1].eq(c.color[1]) & mask[2].eq(c.color[2])
            target[eq] = c.trainId    
            
        return img, target

    def masks_to_indices(self, masks: torch.Tensor) -> torch.Tensor:
        _, indices = masks.softmax(dim=1).max(dim=1)
        return indices

    def to_image(self, indices: torch.Tensor) -> Image.Image:
        target = torch.zeros((3, indices.shape[0], indices.shape[1]),
                             dtype=torch.uint8, device=indices.device, requires_grad=False)

        for i, lbl in enumerate(self.classes):
            eq = indices.eq(lbl.trainId)

            target[0][eq] = lbl.color[0]
            target[1][eq] = lbl.color[1]
            target[2][eq] = lbl.color[2]

        return TF.to_pil_image(target.cpu(), 'RGB')

# List of classes that we want to detect in the input
classes = [
    #                 name                     ID    trainId   category            catId     hasInstances   ignoreInEval   color
    CityscapesClass(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityscapesClass(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityscapesClass(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityscapesClass(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityscapesClass(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    CityscapesClass(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    CityscapesClass(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    CityscapesClass(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    CityscapesClass(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    CityscapesClass(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    CityscapesClass(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    CityscapesClass(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    CityscapesClass(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    CityscapesClass(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    CityscapesClass(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    CityscapesClass(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    CityscapesClass(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    CityscapesClass(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    CityscapesClass(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (0  ,  0,  0) ),
    CityscapesClass(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    CityscapesClass(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    CityscapesClass(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    CityscapesClass(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    CityscapesClass(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    CityscapesClass(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    CityscapesClass(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    CityscapesClass(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    CityscapesClass(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    CityscapesClass(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    CityscapesClass(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    CityscapesClass(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    CityscapesClass(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    CityscapesClass(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    CityscapesClass(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    CityscapesClass(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (0  ,0  ,  0) ),
]

def display_samples():
    # HTML templates for displaying random samples in a table
    template_table = '<table><thead><tr><th>Subset</th><th>Amount</th><th>Size</th><th>Input sample</th><th>Truth sample</th></tr></thead><tbody>{0}</tbody></table>'
    template_row = '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td></tr>'
    template_img = '<img src="data:image/png;base64,{0}"/>'

    # Display a random sample of each split of the dataset
    rows = []
    for name, ds_sub in ds_split.items():
        # Draw a random sample from the dataset so that we can convert it back to an image
        input, truth = random.choice(ds_sub)
        #print(torch.unique(truth))

        input = TF.to_pil_image(input)
        truth = ds_sub.to_image(truth)

        # Create a buffer to save each retrieved image into such that we can base64-encode it for diplay in our HTML table
        with BytesIO() as buffer_input, BytesIO() as buffer_truth:
            input.save(buffer_input, format='png')
            truth.save(buffer_truth, format='png')

            # Store one row of the dataset
            images = [template_img.format(b64encode(b.getvalue()).decode('utf-8')) for b in (buffer_input, buffer_truth)]
            rows.append(template_row.format(name, len(ds_sub), '&times;'.join([str(s) for s in input.size]), *images))

    # Render HTML table
    table = template_table.format(''.join(rows))
    display(HTML(table))

def prep_init():
    download_data()

    preprocess(os.path.join(css_truth, "train"), os.path.join(css_input, "train"), css_truth_pp, css_input_pp, 'css')
    preprocess(os.path.join(css_truth, "val"), os.path.join(css_input, "val"), val_truth_pp, val_input_pp, 'css')
    preprocess(os.path.join(dir_bdd, "labels"), os.path.join(dir_bdd, "images"), bdd_truth_pp, bdd_input_pp, 'bdd')
    preprocess(os.path.join(dir_acdc, "labels"), os.path.join(dir_acdc, "images"), acdc_truth_pp, acdc_input_pp, 'acdc')
    preprocess(os.path.join(dir_map, "labels"), os.path.join(dir_map, "images"), map_truth_pp, map_input_pp, 'map')

    # Create one instance of the CityscapesDataset for each split type
    global ds_split 
    ds_split = {
        'css': SegmentationDataset(css_input_pp, css_truth_pp, sample_size, classes),
        'css_val': SegmentationDataset(val_input_pp, val_truth_pp, sample_size, classes),
        'bdd': SegmentationDataset(bdd_input_pp, bdd_truth_pp, sample_size, classes),
        'acdc': SegmentationDataset(acdc_input_pp, acdc_truth_pp, sample_size, classes),
        'map': SegmentationDataset(map_input_pp, map_truth_pp, sample_size, classes),
    }

    display_samples()
