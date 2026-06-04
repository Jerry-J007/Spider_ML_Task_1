# LINKS  
Kaagle link - https://www.kaggle.com/code/jerusonj/sml-task-1-bonus

collab link - https://colab.research.google.com/drive/1ya0qDR2o2k8miXgZs7DisYOUZn1Ux8FO?usp=sharing

# DESCRIPTION
1. Objective
The goal of this bonus task was to explore unsupervised representation learning by implementing an Autoencoder on the Fashion-MNIST dataset. The objective was to successfully compress 28x28 high-dimensional images into a lower-dimensional latent representation (the bottleneck) and reconstruct the original images with minimal information loss(encoding and decoding).

2.Architecture Design
A fully connected Multi-Layer Perceptron (MLP) architecture was built using PyTorch, structured in a symmetrical hourglass design:
The Encoder: Squeezes the 784 flattened input pixels through progressively smaller hidden layers (784 - > 256 - > 64 - >Latent Dimension).
The Decoder: Reconstructs the image by expanding the latent vector back through mirrored hidden layers (Latent Dimension - > 64 - > 256 - > 784).

3.Training
Loss Function: nn.MSELoss() (Mean Squared Error) was utilized to mathematically measure the pixel-by-pixel physical difference between the original input image and the reconstructed output.

Optimizer: The Adam optimizer was used with a learning rate of 0.001.

The model was trained over 3 epochs, discarding classification labels and strictly comparing outputs to inputs.

4. The  Experiment
To understand how compression affects reconstruction quality, the model was instantiated and trained three separate times, varying only the size of the bottleneck (latent_dim):

Latent = 16: High compression (representing the image using only 16 numbers).

Latent = 32: Medium compression (the baseline).

Latent = 64: Low compression (retaining more feature data).

5. Results and Visualization
Two primary methods were used to evaluate the experimental results:

Visual Comparison Grid: A Matplotlib script was written to pass a single batch of unseen test images through all three trained models simultaneously. The output was a side-by-side visual grid (Original | Latent 16 | Latent 32 | Latent 64). Visually, the Latent=16 reconstructions were highly blurred, while Latent=64 captured much sharper details.

Comparative Loss Curves: The training and validation MSE loss histories for all three models were plotted on a single graph. The graph mathematically proved the "Information Trade-Off" principle: the model with the 16-dimension bottleneck exhibited the highest reconstruction error, while the 64-dimension bottleneck achieved the lowest loss.

# Conclusion: 
The experiment successfully demonstrated that restricting the latent space forces the network to prioritize broad shapes over fine details. Increasing the bottleneck dimension significantly improves reconstruction quality and lowers MSE loss, at the cost of less efficient data compression.
