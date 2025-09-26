import cv2
import numpy as np

from cccv import AutoModel, SRBaseModel

model: SRBaseModel = AutoModel.from_pretrained("https://github.com/EutropicAI/cccv_demo_remote_model")

img = cv2.imdecode(np.fromfile("../assets/test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("../assets/test_remote_example_out.jpg", img)
