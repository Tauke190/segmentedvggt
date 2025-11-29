import os
import shutil

def sample_images(src_folder, dst_folder, n, sort_by='name'):
    """
    Samples n images from src_folder and copies them to dst_folder.
    Sampling is deterministic and covers the whole set.
    sort_by: 'name' or 'mtime'
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # List all image files
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if sort_by == 'mtime':
        images.sort(key=lambda x: os.path.getmtime(os.path.join(src_folder, x)))
    else:
        images.sort()  # by name

    N = len(images)
    if n > N:
        raise ValueError("n cannot be greater than the number of images in the source folder.")

    # Deterministic sampling: take every k-th image
    step = N / n
    indices = [int(i * step) for i in range(n)]
    sampled = [images[i] for i in indices]

    for img in sampled:
        shutil.copy2(os.path.join(src_folder, img), os.path.join(dst_folder, img))

    print(f"Sampled {n} images from {N} and copied to {dst_folder}")

# Example usage:
sample_images(r'C:\Users\avina\OneDrive\Desktop\cup_001\cup\575_84340_166477\images', r'examples\cup\images', 15, sort_by='name')