# Fashion-MNIST ile Kıyafet Sınıflandırma (CNN, PyTorch)

Computer Vision dersi ödevi. Fashion-MNIST veri seti ile 10 sınıflı bir görüntü sınıflandırma modeli (CNN)
PyTorch ile eğitildi, Gradio ile web uygulamasına dönüştürüldü ve telefonla çekilen 5 fotoğrafla test edildi.

## Sonuçlar
| Metrik | Değer |
|---|---|
| Test Accuracy | 0.9183 |
| Precision (macro) | 0.9194 |
| Recall (macro) | 0.9183 |
| F1-score (macro) | 0.9186 |
| Eğitim süresi | 230 sn (15 epoch, Apple GPU / MPS) |
| Tahmin süresi | ~0.9 ms / görüntü |
| Telefon fotoğrafları | 3/5 doğru |

## Dosyalar
- `CNN_FashionMNIST_PyTorch.ipynb` – veri hazırlığı, model, eğitim, değerlendirme, telefon fotoğrafı testi
- `app.py` – Gradio web uygulaması
- `gercek_fotograflar/` – telefonla çekilen 5 fotoğraf
- `outputs/` – eğitilmiş model (`fashion_cnn.pth`), grafikler, `sonuclar.json`

## Çalıştırma
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook CNN_FashionMNIST_PyTorch.ipynb   # Run All
python app.py                                     # http://127.0.0.1:7860
```

## Veri seti
Fashion-MNIST – https://github.com/zalandoresearch/fashion-mnist
(Notebook ilk çalıştırıldığında otomatik indirilir.)
