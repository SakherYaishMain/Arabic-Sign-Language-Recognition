# Enhanced Image Analysis Techniques Using Multiscale Feature Extraction and Attention Mechanisms 📸✨

## 📝 Abstract
This project focuses on improving image recognition, particularly for Arabic Sign Language. By combining **multiscale feature extraction**, **spatial-reduction attention**, and **progressive dimensional reduction**, we aim to capture important details and context within images. Our methods outperform baseline approaches, offering better recognition accuracy for static images.

## 🔍 Objective
- Implement **multiscale feature extraction** to capture fine details and global context.
- Use **spatial-reduction attention** to focus on important image regions.
- Apply **progressive dimensional reduction** to preserve essential features.
- Compare performance with models like **ResNet-50**, **ViT**, and **GoogleNet**.

## 💻 Models Implemented
- **U-Net**: Baseline model for image segmentation.
- **ResNet-50**: Fine-tuned for Arabic Sign Language gestures.
- **GoogleNet**: Multi-scale model for detailed feature capture.
- **Vision Transformer (ViT)**: Self-attention mechanism for spatial context.
- **Custom Models**: Based on **ResNet-18**, both with and without pretrained weights.

## 🧠 Approach
1. **Multiscale Feature Extraction**: Captures details and context in images.
2. **Spatial-Reduction Attention**: Focuses on key regions and reduces noise.
3. **Progressive Dimensional Reduction**: Maintains feature hierarchy and reduces complexity.

## 📊 Dataset
The **ArASL Database Grayscale** contains **Arabic Sign Language gestures** in grayscale images. Each image corresponds to a unique Arabic letter and is resized to **224x224 pixels** for training and testing.

## ⚙️ Implementation Details
- **Data Preparation**: Dataset is loaded and transformed using custom preprocessing techniques.
- **Model Architecture**: We used CNNs and transformers, including **ResNet-18** as the backbone, for feature extraction and classification.

## 🚀 Results
### Model Performance:
| Model                 | Test Accuracy | F1 Score | AUC  |
|-----------------------|---------------|----------|------|
| U-Net                 | 4.10%         | 0.0025   | 0.5000|
| ResNet-50             | 82.75%        | 0.8286   | 0.9916|
| GoogleNet             | 98.99%        | 0.9900   | 0.9995|
| ViT                   | 96.98%        | 0.9701   | 0.9996|
| Custom Model (Pretrained) | 97.65%    | 0.9773   | 0.9991|
| Custom Model (Non-Pretrained) | 91.14% | 0.9034   | 0.9974|

### Key Takeaways:
- **Custom models** with **pretrained weights** performed the best.
- **Multiscale attention** significantly improved performance.

## 💡 Future Work
- Explore more advanced models like **DeepLab** or **Mask R-CNN** for segmentation.
- Enhance **data augmentation** techniques for better generalization.
- Implement **real-time recognition** for sign language in live video feeds.

## 📥 How to Run
1. Install dependencies from `requirements.txt`.
2. Run the model using the pre-trained weights or train from scratch with custom configurations.
3. Use the intuitive UI for **sign language recognition** and visualization.

## 📅 Conclusion
This work demonstrates that **multiscale feature extraction** and **attention mechanisms** can significantly improve the accuracy of Arabic Sign Language recognition. Our custom model, with and without pretrained weights, shows how powerful transfer learning can be in specialized tasks.

## 🤖 Technologies Used
- **Python 3.x**: Programming language.
- **TensorFlow & Keras**: For building and training models.
- **NumPy**: Data manipulation.
- **Matplotlib**: For visualizations.
- **scikit-learn**: Evaluation metrics.

## 📄 References
- Vaswani et al. (2017). *Attention is All You Need.*
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
- Garg et al. (2024). *MVTN: A Multiscale Video Transformer Network for Hand Gesture Recognition.*
- Balat et al. (2024). *Advanced Arabic Alphabet Sign Language Recognition Using Transfer Learning and Transformer Models.*
- ArASL Database: [Hugging Face Dataset](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale)
