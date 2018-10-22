
import numpy as np
import cv2
from PIL import Image
import os

np.set_printoptions(threshold=np.inf)


def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

def save_samples(sampleno,movies,dirname,frame_size):
    k = sampleno
    for g in range(frame_size):
        another=np.empty([40,40],dtype=np.float)
        new_ar=movies[k][g][:][:]
        for i in range(len(new_ar)):
            for j in range(len(new_ar)):
                another[i][j] = new_ar[i][j][0]

        img = Image.fromarray(another, 'L')
        img.save(dirname + 'Frames/' + str(g)+'.png')

        # Create video
        img_array = []
        for i in range(frame_size):
            img_array.append(cv2.imread(dirname + 'Frames/' + str(i)+'.png'))

        height,width,layers = img_array[0].shape
        size = (width,height)
        fps = 1

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(dirname + 'video.avi',fourcc,fps,size)

        for j in range(len(img_array)):
            video.write(img_array[j])

        cv2.destroyAllWindows()
        video.release()




# Main Code

sample_size = 10
frame_size = 15
noisy_movies, shifted_movies = generate_movies(n_samples=sample_size)

for sample_no in range(sample_size):
    rootdir = 'Samples/'
    sampledir = rootdir + 'Sample'+str(sample_no+1)+'/'
    noisy_dirname = sampledir + 'Noisy/'
    shifted_dirname = sampledir +'Shifted/'
    
    if not os.path.exists(sampledir):
        os.makedirs(noisy_dirname+'Frames/')
        os.makedirs(shifted_dirname+'Frames/')
    
    save_samples(sample_no,noisy_movies,noisy_dirname,frame_size)
    save_samples(sample_no,shifted_movies,shifted_dirname,frame_size)
    
    


   
