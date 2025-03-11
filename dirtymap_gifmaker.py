import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import pickle

dm_file = open("dirtymap.pickle","rb")
dm_dict = pickle.load(dm_file)
dm = dm_dict["dirtymap"]
frequencies = dm_dict["freq"]
nframes=frequencies.shape[0]
nx = dm_dict["nx"]
ny = dm_dict["ny"]

dm = dm.reshape([nx,ny,nframes])
maxInArray = np.max(dm)

if not os.path.exists("/tmp/animate"):
    os.mkdir("/tmp/animate")
print("Plotting...")
for i in range(nframes):
    plt.imshow(dm[:,:,i], vmin=0, vmax=maxInArray, interpolation="none", origin="lower")
    plt.title(f"Freq: {frequencies[i]:.4f} MHz")
    plt.savefig("/tmp/animate/dm"+str(i)+".png")
    plt.close()
    
    print("\x1b[2K",str((i+1)/nframes * 100)+"% complete", end='\r')

print("\nSaving gif...")
with iio.imopen('dirty_map_animation.gif', "w") as gif:
    for i in range(nframes):
        filename = "/tmp/animate/dm"+str(i)+".png"
        image = iio.imread(filename)
        gif.write(image, loop=0)
        os.remove(filename)
        print("\x1b[2K",str((i+1)/nframes * 100)+"% complete", end='\r')
print("\n")
