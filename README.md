## Finding Nemo - Multi-Label Deep Sea Image Classification
In this project, we tested various machine learning and deep learning algorithms to classify marine animal species in deep-sea images, requiring fine-tuning CNNs such as ResNet, DenseNet, EfficientNet, etc. This program was created for the course "MSE 546: Advanced Machine Learning" term-long project. Based on the Kaggle competition: https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection

### The full report and results of this project can be seen at: 
https://docs.google.com/presentation/d/1VZVHRA__KtrJDhEmUTf-QkEKO1lc2-MAntAnwPunsb8/edit?usp=sharing

### Technologies:
- Python
- PyTorch
- NumPy
- Jupyter

### Objective: 
Multi-label classification identifying multiple marine organisms in an image

### Input Data: 
Raw image data (5950 images), annotations for 290 fine-grained categories, mapping of fine grained categories to one of 20 super-categories, and species metadata

### What are the Images of?: 
Pictures of marine organisms from the upper ocean <800m.

### Output: 
A vector of 290 values, each representing the probability of a corresponding species being present

### Evaluation Metric: 
MAP@20, which measures how well the top 20 predicted species match the actual species present

#### Why did we pick this evaluation metric?
- MAP@20 evaluates the model’s ability to rank the correct categories within the top 20 predictions
- A higher MAP@20 score means the model is correctly ranking species in the top 20 per image
- MAP@20 was chosen as the evaluation metric because precision alone doesn’t consider ranking, only the fraction of correct predictions. MAP@20 is also better suited for imbalanced datasets, like this one.
- Partially handles our dataset’s imbalanced number of images per category
- Helps assess if species are ranked well, rather than being overshadowed by common ones

### Challenges: 
Class imbalances, computationally heavy, low resolution images

### Required compute: 
Ideally GPUs, with adequate storage to support the difficulty of the task

### Group: 
Elize Kooji, Mateo Alvarez, Ria Narang, Thomas Kleinknect, Matthew Erxleben
