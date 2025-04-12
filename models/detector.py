import torch
from torchvision import models, transforms
from PIL import Image

class FruitDetector:
    def __init__(self, score_threshold=0.8, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.score_threshold = score_threshold

    def detect(self, pil_image):
        image_tensor = self.transform(pil_image).to(self.device)
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        boxes = predictions['boxes']
        scores = predictions['scores']
        valid_idxs = scores > self.score_threshold
        filtered_boxes = boxes[valid_idxs]
        return filtered_boxes.cpu().numpy()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import sys

    if len(sys.argv) < 2:
        print("Uso: python detector.py <ruta_de_la_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    pil_img = Image.open(image_path).convert("RGB")
    
    detector = FruitDetector(score_threshold=0.8)
    boxes = detector.detect(pil_img)
    
    fig, ax = plt.subplots(1)
    ax.imshow(pil_img)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
