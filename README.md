# 🧠 Image Generation using Generative Adversarial Networks (GANs)

This project implements a basic **Generative Adversarial Network (GAN)** to generate synthetic images that resemble a given dataset. GANs are one of the most exciting advancements in deep learning, enabling machines to create new data such as images, audio, and text.

---

## 🎯 Objective

To train a GAN that learns the distribution of a dataset (e.g., handwritten digits, basic shapes, or simple images) and generates new, similar images from random noise vectors.

---

## 🧠 GAN Architecture Overview

The GAN model consists of two neural networks trained simultaneously in a zero-sum game:
- **Generator (G):** Takes a random noise vector and outputs an image.
- **Discriminator (D):** Receives a real or fake image and decides if it’s authentic.

The generator tries to fool the discriminator, while the discriminator tries to correctly classify images as real or fake. Over time, the generator becomes better at producing realistic images.

---

## 🧪 Project Workflow

1. **Load and Preprocess Dataset**
   - Normalize images to range [-1, 1]
   - Reshape and batch the data

2. **Build Generator**
   - Input: Random noise vector (`z`)
   - Layers: Dense → BatchNorm → LeakyReLU → Reshape → Conv2DTranspose
   - Output: Generated image

3. **Build Discriminator**
   - Input: Image (real or fake)
   - Layers: Conv2D → LeakyReLU → Dropout
   - Output: Real/Fake probability

4. **Compile and Train**
   - Train discriminator on real and generated images
   - Train generator via combined GAN model
   - Use Binary Crossentropy loss
   - Save generated samples after each few epochs

---

## 📈 Results

- Generator starts from pure noise and gradually produces more realistic samples.
- Visualization of generated images shows improvement with training.
- Loss curves indicate adversarial training dynamics.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Open `gans (2).ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
3. Run all cells in order.
4. Adjust hyperparameters:
   - Latent dimension
   - Learning rate
   - Batch size
   - Epochs

---

## 📌 Possible Extensions

- Add Conditional GAN (cGAN) for labeled image generation
- Use a more complex dataset (e.g., Fashion-MNIST or CelebA)
- Improve model architecture with deeper layers or dropout
- Implement loss smoothing or label flipping

---

## 👩‍💻 Author

**Alaa Shorbaji**  
Artificial Intelligence Instructor – Armed Forces  
Master’s Researcher in Deep Learning and XAI  
GitHub: [your_username]  
LinkedIn: [your_link]

---

## 📜 License

Educational use only. Please credit the author if you use this code.
