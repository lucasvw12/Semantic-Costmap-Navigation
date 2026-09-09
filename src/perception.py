import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

DETECTOR_NAME = "IDEA-Research/grounding-dino-tiny"
SAM_NAME = "nielsr/slimsam-50-uniform"

DETECT_SHORTEST_EDGE = 480
DETECT_LONGEST_EDGE = 640
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MAX_BOXES = 8
USE_SAM = True
MASK_STRIDE = 2
TORCH_THREADS = 8

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, min(TORCH_THREADS, os.cpu_count() or TORCH_THREADS)))


class Perception:
    def __init__(self, detector_name=DETECTOR_NAME, sam_name=SAM_NAME,
                 use_sam=USE_SAM):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_sam = use_sam

        self.processor = AutoProcessor.from_pretrained(detector_name)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(detector_name)
            .to(self.device)
            .eval()
        )

        self.detect_size = {
            "shortest_edge": DETECT_SHORTEST_EDGE,
            "longest_edge": DETECT_LONGEST_EDGE,
        }

        if self.use_sam:
            self.sam_processor = SamProcessor.from_pretrained(sam_name)
            self.sam_model = SamModel.from_pretrained(sam_name).to(self.device).eval()
        else:
            self.sam_processor = None
            self.sam_model = None

    def warmup(self, prompt, width=640, height=480):
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        results = self.detect(blank, prompt)
        boxes = results[0]["boxes"]

        if len(boxes) == 0:
            boxes = torch.tensor(
                [[0.0, 0.0, width / 2.0, height / 2.0]],
                device=self.device
            )

        self.segment(blank, boxes)

    def detect(self, rgb_image, prompt):
        image = Image.fromarray(rgb_image)
        inputs = self.processor(
            images=image,
            text=prompt,
            size=self.detect_size,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        if MAX_BOXES and len(results[0]["boxes"]) > MAX_BOXES:
            keep = torch.topk(results[0]["scores"], MAX_BOXES).indices
            results[0]["boxes"] = results[0]["boxes"][keep]
            results[0]["scores"] = results[0]["scores"][keep]
            results[0]["text_labels"] = [
                results[0]["text_labels"][index] for index in keep.tolist()
            ]

        return results

    def segment(self, rgb_image, boxes):
        if not self.use_sam:
            return self.box_coordinates(rgb_image.shape, boxes)

        image = Image.fromarray(rgb_image)
        boxes = boxes.detach().cpu().numpy().tolist()
        boxes = [boxes]
        inputs = self.sam_processor(
            images=image,
            input_boxes=boxes,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output = self.sam_model(**inputs)

        masks = self.sam_processor.post_process_masks(
            output.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        best_indices = output.iou_scores[0].argmax(dim=1)
        best_masks = masks[0][torch.arange(masks[0].shape[0]), best_indices]
        cleaned_masks = self.clean_segmentation_mask(best_masks)

        coordinates = []
        for mask in cleaned_masks:
            ys, xs = np.nonzero(mask[::MASK_STRIDE, ::MASK_STRIDE])
            coordinates.append(
                np.column_stack((xs * MASK_STRIDE, ys * MASK_STRIDE))
            )
        return coordinates

    def box_coordinates(self, image_shape, boxes):
        height, width = image_shape[0], image_shape[1]
        coordinates = []

        for box in boxes.detach().cpu().numpy():
            x0 = int(max(0, np.floor(box[0])))
            y0 = int(max(0, np.floor(box[1])))
            x1 = int(min(width, np.ceil(box[2])))
            y1 = int(min(height, np.ceil(box[3])))

            if x1 <= x0 or y1 <= y0:
                coordinates.append(np.empty((0, 2), dtype=np.int32))
                continue

            xs, ys = np.meshgrid(
                np.arange(x0, x1, MASK_STRIDE),
                np.arange(y0, y1, MASK_STRIDE)
            )
            coordinates.append(
                np.column_stack((xs.ravel(), ys.ravel())).astype(np.int32)
            )

        return coordinates

    def clean_segmentation_mask(self, masks):
        kernel = np.ones((3, 3), np.uint8)
        cleaned_masks = []

        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks)

        masks = masks.squeeze()

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        masks_np = masks.to(torch.uint8).cpu().numpy()

        for mask_np in masks_np:
            eroded_mask = cv2.erode(
                mask_np,
                kernel,
                iterations=1
            )

            cleaned_mask = cv2.dilate(
                eroded_mask,
                kernel,
                iterations=1
            )

            cleaned_masks.append(cleaned_mask)

        return cleaned_masks
