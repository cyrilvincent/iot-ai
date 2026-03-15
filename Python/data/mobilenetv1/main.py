import os

# Download sample images for testing
os.system("wget -q -O test.jpg https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg")
os.system("wget -q -O test2.jpg https://www.hartz.com/wp-content/uploads/2022/04/small-dog-owners-1.jpg")

# Train the model
import train

# Test the model
import test
