# GPU-Accelerated-Sobel-Edge-Detection-on-Large-Image-Dataset-Using-CUDA.
## AIM

To implement GPU-accelerated Sobel Edge Detection using CUDA-enabled PyTorch and process image data efficiently on the GPU.

## ALGORITHM
1.Start the program.
2.Import the required libraries such as PyTorch, OpenCV, NumPy, and Matplotlib.
3.Check whether a CUDA-enabled GPU is available.
4.Load the input image and convert it to grayscale.
5.Convert the image into a PyTorch tensor and transfer it to GPU memory.
6.Define Sobel X and Sobel Y kernels.
7.Apply convolution using the Sobel kernels to compute horizontal and vertical gradients.
8.Calculate the edge magnitude using the gradients.
9.Normalize the output image.
10.Transfer the result back to CPU memory.
11.Save and display the edge-detected image.
12.Measure the execution time.
13.Stop the program.

## Program:
```
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# Load image
img = cv2.imread("Screenshot 2026-06-01 102726.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
else:

    # Convert image to tensor and move to GPU
    img_tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    # Sobel X Kernel
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Sobel Y Kernel
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    start_time = time.time()

    # Apply Sobel filters
    gx = torch.nn.functional.conv2d(
        img_tensor,
        sobel_x,
        padding=1
    )

    gy = torch.nn.functional.conv2d(
        img_tensor,
        sobel_y,
        padding=1
    )

    # Edge Magnitude
    edges = torch.sqrt(gx**2 + gy**2)

    # Normalize
    edges = torch.clamp(edges, 0, 255)

    # Move result back to CPU
    result = edges.squeeze().cpu().numpy().astype(np.uint8)

    end_time = time.time()

    print("Execution Time:", end_time - start_time, "seconds")

    # Save output
    cv2.imwrite("output_edge.png", result)

    # Display Input and Output
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(result, cmap='gray')
    plt.title("Edge Detected Image")
    plt.axis('off')

    plt.show()

    print("Output image saved as output_edge.png")
```
## Output:
<img width="672" height="367" alt="image" src="https://github.com/user-attachments/assets/d187dfa6-fc28-4b28-b550-fe134eea2d5f" />
## RESULT:

Thus, the GPU-accelerated Sobel Edge Detection algorithm was successfully implemented using CUDA-enabled PyTorch. The image was processed on the GPU, and the edges were detected efficiently with reduced execution time, demonstrating the effectiveness of GPU computing for image processing applications.
