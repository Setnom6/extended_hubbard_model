import os
import glob
import imageio.v2 as imageio
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

baseName = "state_similarity_spectrum"
outputName = "Bx_depenency"
extension = ".png"
outputGif = os.path.join(current_dir, "figures", outputName + ".gif")
outputMp4 = os.path.join(current_dir, "figures", outputName + ".mp4")

# Total duration for GIF/MP4
totalDuration = 12.0

globPattern = os.path.join(current_dir, "figures", "iteration", f"{baseName}*{extension}")
pngFiles = sorted(glob.glob(globPattern))

if not pngFiles:
    print("No png files found.")
    raise SystemExit(1)

nFrames = len(pngFiles)
frameDuration = totalDuration / nFrames
fps = nFrames / totalDuration

print(f"Generating GIF and MP4 with {nFrames} frames, total duration {totalDuration:.3f}s "
      f"({frameDuration:.3f}s per frame, {fps:.2f} fps)...")

# === Determine target shape (first image, rounded up to even numbers) ===
firstImage = imageio.imread(pngFiles[0])
h, w = firstImage.shape[:2]
h += h % 2  # make even
w += w % 2  # make even
channels = firstImage.shape[2] if firstImage.ndim == 3 else 1
targetShape = (h, w, channels)

def fitToShape(image, targetShape):
    """Pads image with black pixels so it matches targetShape exactly."""
    hh, ww, cc = targetShape

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    h0, w0, c0 = image.shape

    image = image[:min(h0, hh), :min(w0, ww), :min(c0, cc)]

    fitted = np.zeros(targetShape, dtype=image.dtype)

    fitted[:image.shape[0], :image.shape[1], :image.shape[2]] = image
    return fitted

# === Create GIF ===
with imageio.get_writer(outputGif, mode='I', loop=0) as writer:
    for filePath in pngFiles:
        image = imageio.imread(filePath)
        image = fitToShape(image, targetShape)
        writer.append_data(image, {'duration': frameDuration})

print(f"GIF saved as {outputGif}")

# === Create MP4 ===
with imageio.get_writer(outputMp4, fps=fps, codec='libx264', macro_block_size=1, pixelformat='yuv420p') as writer:
    for filePath in pngFiles:
        image = imageio.imread(filePath)
        image = fitToShape(image, targetShape)
        writer.append_data(image)

print(f"MP4 saved as {outputMp4}")



