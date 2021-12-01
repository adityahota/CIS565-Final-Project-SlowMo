# cuFLAVR

## CIS 565 Fall 2021
### Team 03: Aditya Hota, Richard Chen, Kaan Erdogmus

We are attempting to implement the FLAVR architecture at a low level using CUDA kernels, cuBLAS and cuDNN. Our work is based off the model presented by Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, and Du Tran in their paper _FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation_.

We are attempting to increase the frame rate of videos using the proposed model in the paper, in hopes of making it easier to render ray-traced movies. By using a fraction of the frames needed to achieve a high frame rate, rendering time for videos will be reduced.

<img width="555" alt="FLAVR" src="https://user-images.githubusercontent.com/12516225/144300538-59f3b06e-97c2-46ae-8395-61522818ec74.png">
Figure 1. Sampling procedure of FLAVR network, obtained from FLAVR paper
