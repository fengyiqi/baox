import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# List of image filenames
image_files = [f"mfbo_1D_step_{i}.png" for i in range(20)]


# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Load the first image to set the plot
img = plt.imread(image_files[0])
im = ax.imshow(img)
plt.tight_layout()
ax.axis('off')  # Hide axes

# Update function for animation
def update(frame):
    im.set_array(plt.imread(image_files[frame]))
    return [im]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(image_files), interval=500)  # 500ms per frame

# Save as GIF
ani.save("qMFCEI_1d_5.gif", writer="pillow", fps=2)  # fps = 2 (2 frames per second)

print("GIF saved as animation.gif")
