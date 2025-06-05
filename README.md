# OpenCV Face Detector

Bu proje, Python ve OpenCV kullanılarak gerçek zamanlı yüz tanıma, duygu analizi ve isim tanıma işlemlerini gerçekleştiren bir yüz algılama sistemidir. Haar Cascade sınıflandırıcısı kullanılarak yüzler tespit edilir ve çeşitli modüller aracılığıyla analiz yapılır.

## 🚀 Özellikler

* Gerçek zamanlı yüz tespiti (kamera ile)
* Duygu analizi
* İsim tanıma
* OpenCV tabanlı çözüm
* Modüler Python kod yapısı

## 🧰 Gereksinimler

Proje için aşağıdaki Python kütüphaneleri gereklidir:

* `opencv-python`
* `numpy`

`requirements.txt` dosyası yer almadığı için kütüphaneleri manuel kurabilirsiniz:

```bash
pip install opencv-python numpy
```

## 📥 Kurulum

1. Bu GitHub reposunu klonlayın:

```bash
git clone https://github.com/ayse-nur-mert/Opencv-Face-Detector.git
cd Opencv-Face-Detector
```

2. Gerekli kütüphaneleri yükleyin:

```bash
pip install opencv-python numpy
```

3. OpenCV'nin yüz tespiti için gerekli olan dosyanın bulunduğundan emin olun:

* `haarcascade_frontalface_default.xml` (zaten proje içinde mevcut)

## ▶️ Kullanım

Ana uygulamayı başlatmak için:

```bash
python main.py
```

Kamera açılır ve ekranda gerçek zamanlı yüz algılama işlemi başlar. Duygu ve isim tanıma gibi işlemler de entegre şekilde çalışır.
