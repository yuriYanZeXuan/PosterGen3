# %%
import matplotlib.pyplot as plt
from PIL import Image
# Load and split image into 4 parts by center cross
img = Image.open('/Users/yanzexuan/code/PosterGen3/dev/image.png')
width, height = img.size
cx, cy = width // 2, height // 2

# Crop 4 quadrants
top_left = img.crop((0, 0, cx, cy))
top_right = img.crop((cx, 0, width, cy))
bottom_left = img.crop((0, cy, cx, height))
bottom_right = img.crop((cx, cy, width, height))

# Display in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(top_left)
axes[0, 0].set_title('Top Left')
axes[0, 0].axis('off')

axes[0, 1].imshow(top_right)
axes[0, 1].set_title('Top Right')
axes[0, 1].axis('off')

axes[1, 0].imshow(bottom_left)
axes[1, 0].set_title('Bottom Left')
axes[1, 0].axis('off')

axes[1, 1].imshow(bottom_right)
axes[1, 1].set_title('Bottom Right')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


