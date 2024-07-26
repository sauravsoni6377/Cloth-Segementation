import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
import yaml
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from segment_anything import build_sam, SamPredictor,sam_model_registry 
from huggingface_hub import hf_hub_download


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/saurav-soni/Downloads/segmentClothes/config/tshirt.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    if os.path.exists(settings['checkpoint']):
        checkpt = settings['checkpoint']
        checkpt = torch.load(checkpt)
        sam = sam_model_registry["vit_b"](checkpoint=checkpt)
        predictor = SamPredictor(sam)

    else :
        checkpt = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
        torch.save(checkpt,'checkpoint/sam.pth')
        sam = sam_model_registry["vit_b"](checkpoint=checkpt)
        predictor = SamPredictor(sam)

    num_images = settings['num_images']
    dataset_path = settings['dataset']
    output_dir = settings['output_dir']

    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    for i in range(num_images):
        image_folder = os.path.join(dataset_path, f'img/{i}/')
        for image_file in os.listdir(image_folder):
            basename = os.path.basename(image_file)
            type_img = basename.split('_')[0]

            if type_img == 'img':
                img_file = os.path.join(image_folder, image_file)
                image = cv2.imread(img_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                '''
                Segformer-based cloth segmentation
                '''
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits.cpu()
                print(logits.shape)
                upper_cloth = logits[:, 4:5, ...]
                print(upper_cloth.shape)
                upsampled_tensor = F.interpolate(upper_cloth, size=(1080, 1080), mode='bilinear', align_corners=False)
                threshold = 0.5  # Adjust the threshold as needed
                probabilities = torch.sigmoid(upsampled_tensor)
                binary_mask = (probabilities > threshold).float()
                mask_array = (binary_mask.squeeze().numpy() * 255).astype('uint8')

                # Calculate the average of segmented points
                mask_indices = np.where(mask_array == 255)
                average_x = int(np.mean(mask_indices[1]))
                average_y = int(np.mean(mask_indices[0]))
                print(f"Average coordinates of segmented points: ({average_x}, {average_y})")

                '''
                This is for SAM based cloth segmentation
                '''

                predictor.set_image(image)
                print(image.shape)
                input_point = np.array([[average_x, average_y]])
                input_label = np.array([1])
                masks_array,_,_ = predictor.predict( \
                            point_coords=input_point, \
                            point_labels=input_label, \
                            multimask_output=False
                            )
                print(masks_array.shape)

                # mask_image = Image.fromarray(mask_array)
                masks_array = masks_array.squeeze(0)
                mask_image = Image.fromarray(masks_array)
                save_directory = os.path.join(output_dir, f'{i}')
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                mask_image.save(os.path.join(save_directory, basename))
