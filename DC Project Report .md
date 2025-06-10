![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.001.png)

**DESIGN CREDIT** 

**PROJECT**
## **INSTRUCTOR : Professor Pratik Mazumder**
**V NIRANJANA (B23CH1046) JIGYASA TIWARI (B23EE1029)**

**Global vs Grid: A Comparative Study of Feature Extraction in Image Classification**

This project presents a comparative study of global versus grid-based feature extraction methods in image classification. Leveraging a fixed pretrained deep learning model, we extract features from both the entire image and from local grids within each image. 

These features are then evaluated using standard classifiers, allowing us to directly  compare  the  classification  accuracy  of  the  two  approaches.The objective is to determine which approach is better for visual recognition tasks.
# **Datasets Used :**
## **Oxford 102 Flower Dataset**
![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.002.png)
## **Flower Dataset Overview**
- Training Data: 6,552 images across 102 flower species (64 images/class on average)
- Validation  Data:  818  images  (8  images/class  average)  with  same  class structure
- Average size of the image : 224x224 pixels (sizes vary).
- Test Data: Unlabeled images for final predictions (exact count unspecified)
- cat\_to\_name.json: Maps numeric labels (1-102) to flower names

2![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.003.png)
## **Animal Image Classification Dataset**
![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.004.png)
### **Animal Dataset Overview**
- #### 1000 images in each of the different animal classes
- Validation  Data:  818  images  (8  images/class  average)  with  same  class structure
- Image Resolution: Uniform 256x256 pixels for all images, ensuring clarity and consistency.
- #### File Format: JPG
- #### Color Space: RGB
# **Pretrained Model Used :**
The success of feature extraction hinges on using models pretrained on large- scale vision datasets. For this study, we leverage **Google's Vision Transformer (ViT-Base)**  pretrained  on  ImageNet-21k,  accessed  via  Hugging  Face's Transformers library. This model provides a robust foundation for both global and grid-based feature extraction without requiring task-specific training from scratch.

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.005.png)
## **Key Components:**
#### 1\.**ViTImageProcessor**
- Acts as the bridge between image data and machine learning models.
- Resizing: All images → 224×224 pixels

  Normalization: Pixel values scaled to [-1,1] using ImageNet mean/std![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.006.png)

  2 ![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.007.png)Tensor Conversion: Outputs PyTorch tensors 


## **2.ViTModel**
- Used without classification head for pure feature extraction
- Outputs image embeddings from tensors, enabling classification, similarity analysis etc 

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.008.png)
# **Feature Extraction Explained :**
Feature extraction is performed using a pretrained Vision Transformer (ViT) model  to  convert  raw  images  into  numerical  representations.  For  global features, the entire image is processed through ViT whereas for grid-based features,  the  image  is  split  into  tiles  (e.g.,  2×2  grids),  each  processed independently through ViT, with their [CLS] embeddings concatenated into a high-dimensional vector.

**Code Snippet**  

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.009.png)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.010.jpeg)

2

**Grid Feature Extraction :** 

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.011.png)

**Code Execution in Flower Dataset :** 
### **feature extraction → classifier training → model evaluation**
After feature extraction, the process moves to training and evaluating machine learning classifiers using the extracted features:
## **Classifier Training:**
- Base Models:
  - Logistic Regression (max\_iter=1000)
  - Decision Tree (default params)
- Extended Models:
  - Support Vector Machine (SVM)
  - Ridge Classifier
  - Random Forest (n\_estimators=100)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.012.png)

4

- The  classifiers  learn  to  associate  patterns  in  the  feature  vectors  with specific image classes during training.
- Once  trained,  the  models  are  evaluated  on  a  separate  validation  set  to measure how accurately they can predict the class of unseen images.

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.013.png)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.014.png)

**Accuracies Obsereved :** 



|**Model**|**Training Time (s)**|**Validation Accuracy**|
| - | - | - |
|**Global + Logistic Regression**|53|99\.40%|
|**Global + Decision Tree**|46\.9|72\.50%|
|**Grid + Logistic Regression**|65\.8|99\.50%|
|**Grid + Decision Tree**|213|62\.00%|

**Visualisations :** 

4![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.015.jpeg)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.016.jpeg)

**Test Images predicted :** 

The first image on the left ->test image predicted.The next four images are reference samples  from  the  training  set  of  the  same  class,  visually  confirming  the  model's prediction.

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.017.jpeg)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.018.png)

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.019.png)

5
# **A Comparitive Study :**
## **Change in Dataset :**
- Using Animal Classification Dataset, smaller scale with low class counts to check computational efficiency.
- Used  the  same  pretrained  ViT  model  (google/vit-base-patch16-224)  and identical  2×2  grid  splitting  for  both  global  and  grid  feature  extraction methods
- Data loading manually splits class folders into train/test sets, unlike the previous  dataset’s  predefined  train/validation  folders  used  with torchvision.datasets.ImageFolder.
- Consistent Classifiers: Logistic Regression and Decision Trees

**Observations :** 

**Feature Type Classifier Validation Accuracy ![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.020.png)**Global Logistic Regression 99.67% Global Decision Tree 95.22% Grid (2x2) Logistic Regression 99.78% Grid (2x2) Decision Tree 91.56%

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.021.jpeg) 4


![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.022.jpeg)

**Test Predictions:** 

![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.023.png)  ![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.024.png)![](Aspose.Words.4bf1bbdf-2e8c-455f-b820-a4b4aeb0c024.025.png)

**Results Observed :** 

Across both datasets the accuracy difference b/w global and grid-based features remains minimal, reinforcing key insights about ViT’s feature extraction capabilities : 

- Logistic Regression validation accuracies for global and grid features are nearly identical,  showing  the  ViT  [CLS]  token  captures  rich  spatial  information compactly.
- Grid  features  increase  dimensionality  without  adding  meaningful  new information, often duplicating details in the [CLS] token.
- Logistic  Regression  performs  well  with  compact  global  features,  whereas Decision  Trees  struggle  with  high-dimensional,  sparse  space  created  by  grid features.
- ViT’s  patch-based  architecture  makes  explicit  grid-based  aggregation  largely redundant for this classification task. 4

**THANK YOU** 

6
