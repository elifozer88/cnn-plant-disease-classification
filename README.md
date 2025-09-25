# cnn-plant-disease-classification
CNN ile bitki hastalığı sınıflandırması projesi
# 🌱 CNN ile Bitki Hastalığı Sınıflandırması
*Plant Disease Classification using Convolutional Neural Networks*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Proje Amacı

Bu projede, **Convolutional Neural Network (CNN)** mimarisi kullanarak bitki yapraklarındaki hastalıkları otomatik olarak tespit eden bir derin öğrenme modeli geliştiriyoruz. Tarımsal verimliliği artırmak ve erken hastalık tespiti yapmak için tasarlanmış bir yapay zeka çözümüdür.

## 📊 Veri Seti Hakkında Bilgi

- **Veri Kaynağı**: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Toplam Görüntü Sayısı**: 87,000+
- **Sınıf Sayısı**: 38 farklı bitki-hastalık kombinasyonu
- **Görüntü Boyutu**: 256x256 (224x224'e yeniden boyutlandırıldı)
- **Format**: RGB renkli görüntüler
- **Bitki Türleri**: Domates, Patates, Mısır, Elma vb.

### Veri Dağılımı
- **Eğitim Seti**: %70 (~61,000 görüntü)
- **Doğrulama Seti**: %20 (~17,000 görüntü)  
- **Test Seti**: %10 (~9,000 görüntü)

## 🛠️ Kullanılan Yöntemler

### 1. Veri Ön İşleme
- **Görüntü Yeniden Boyutlandırma**: 224x224 piksel
- **Normalizasyon**: Piksel değerleri [0,1] aralığına
- **Veri Artırma (Data Augmentation)**:
  - Döndürme (±25°)
  - Yatay/Dikey kaydırma (%15)
  - Yakınlaştırma (%15)
  - Yatay çevirme
  - Parlaklık ayarı

### 2. CNN Model Mimarisi
```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Params   
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)     896      
 batch_normalization         (None, 222, 222, 32)     128      
 max_pooling2d              (None, 111, 111, 32)     0        
 dropout                    (None, 111, 111, 32)     0        
 
 conv2d_1 (Conv2D)          (None, 109, 109, 64)     18496    
 batch_normalization_1       (None, 109, 109, 64)     256      
 max_pooling2d_1            (None, 54, 54, 64)       0        
 dropout_1                  (None, 54, 54, 64)       0        
 
 conv2d_2 (Conv2D)          (None, 52, 52, 128)      73856    
 batch_normalization_2       (None, 52, 52, 128)      512      
 max_pooling2d_2            (None, 26, 26, 128)      0        
 dropout_2                  (None, 26, 26, 128)      0        
 
 flatten (Flatten)          (None, 86528)             0        
 dense (Dense)              (None, 512)               44302848 
 dropout_3                  (None, 512)               0        
 dense_1 (Dense)            (None, 38)                19494    
=================================================================
Total params: 44,416,486
