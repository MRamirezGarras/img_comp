Compare a series of pictures and return those that are more similar.

#Setup

You need to have two folders in the same folder as mail.py:
- images: here you have the images to compare
- transformed_images: to store the images after resizing, so they all have the same size

#Required packages

- Torch
- Torchvision
- Pillow
- Pandas
- Numpy

#Output

The code prints the pictures that are more similar than a certain threshold. Besides, it 
creates a dataframe called sim_matrix with all the values for each comparison. 