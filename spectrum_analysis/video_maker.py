import os
import glob
import imageio.v2 as imageio

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

baseName = "state_similarity_spectrum"
extension = ".png"
outputGif = os.path.join(current_dir, "figures", "A_depenency.gif")

# Total GIF duration in seconds (uniformly distributed across all frames)
totalDuration = 12.0

globPattern = os.path.join(current_dir, "figures", "iteration", f"{baseName}*{extension}")
print(globPattern)
pngFiles = sorted(glob.glob(globPattern))

if not pngFiles:
    print("No png files found.")
    raise SystemExit(1)

nFrames = len(pngFiles)
frameDuration = float(totalDuration) / float(nFrames)  # seconds per frame

print(f"Generating GIF with {nFrames} frames, total duration {totalDuration:.3f}s "
      f"({frameDuration:.3f}s per frame)...")

# loop=0 for infinite loop; duration is set per-frame in append_data for reliability
with imageio.get_writer(outputGif, mode='I', loop=0) as writer:
    for filePath in pngFiles:
        image = imageio.imread(filePath)
        writer.append_data(image, {'duration': frameDuration})  # duración en segundos por cuadro


print(f"GIF saved as {outputGif}")