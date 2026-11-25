# Kurulum ve Çalıştırma (Mac)

## 1. Terminal'i açın
Spotlight (Cmd + Boşluk) → "Terminal" yazın → Enter.

## 2. Python'u kontrol edin
```bash
python3 --version
```
3.10 veya üzeri bir sürüm görmelisiniz. Hata alırsanız https://www.python.org/downloads/ adresinden
Python 3.12'yi indirip kurun ve Terminal'i kapatıp yeniden açın.

## 3. Proje klasörüne geçin
```bash
cd ~/Downloads/CNN_FashionMNIST_PyTorch
```

## 4. Sanal ortam oluşturup etkinleştirin
```bash
python3 -m venv venv
source venv/bin/activate
```
Satırın başında `(venv)` yazısı görünmelidir. Terminal'i her yeniden açtığınızda 3. ve 4. adımdaki
`cd` ve `source venv/bin/activate` komutlarını tekrar çalıştırın.

## 5. Kütüphaneleri kurun (birkaç dakika sürer)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Notebook'u çalıştırın
```bash
jupyter notebook CNN_FashionMNIST_PyTorch.ipynb
```
Tarayıcıda notebook açılır. Üst menüden **Run → Run All Cells** seçin.
- İlk çalıştırmada veri seti otomatik indirilir (~30 MB).
- Mac'te eğitim Apple GPU'su (mps) ile yapılır; yaklaşık 5-10 dakika sürer.
- Bittiğinde `outputs/` klasöründe model (`fashion_cnn.pth`), grafikler ve `sonuclar.json` oluşur.
- Notebook'u kaydedin (Cmd + S): çıktılarıyla birlikte teslim edilecek.

## 7. Uygulamayı çalıştırın
Yeni bir Terminal penceresinde:
```bash
cd ~/Downloads/CNN_FashionMNIST_PyTorch
source venv/bin/activate
python app.py
```
Tarayıcıda http://127.0.0.1:7860 adresini açın, fotoğraf yükleyin.
Rapor için bu ekranın görüntüsünü alın (Cmd + Shift + 4).

## Sorun çıkarsa
- `pip` hata verirse: `python3 -m pip install -r requirements.txt`
- `mps` ile ilgili bir hata görürseniz notebook'un 1. hücresinde `device = torch.device("cpu")` yazıp yeniden çalıştırın.
