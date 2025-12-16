# NLP Projesi: LoRA ile LLM Fine-Tuning

Bu proje, **Qwen2.5-Coder-1.5B-Instruct** modelini **Deep** ve **Diverse** veri setleri kullanarak LoRA yöntemi ile eğitmek ve kodlama yeteneğini artırmak amacıyla yapılmıştır.

## 📂 Dosya İçeriği
- `train.py`: Modelin eğitimi için kullanılan V4 konfigürasyonlu kod.
- `eval.py`: Modelleri AtCoder (Easy) benchmark testine sokan kod.
- `requirements.txt`: Gerekli kütüphaneler.
- `*.json`: Eğitim sırasındaki Loss değerlerini içeren log dosyaları.

## 📊 Benchmark Sonuçları (Pass@1)
Eğitilen modeller AtCoder platformu soruları ile test edilmiştir.

| Model Kategorisi | En İyi Checkpoint | Pass@1 (%) | Çözülen Soru |
| :--- | :--- | :--- | :--- |
| **Deep_instruction** | checkpoint-step-200 | %26.8 | 11/41 |
| **Diverse_instruction** | checkpoint-step-200 | %31.7 | 13/41 |

## 📈 Eğitim Grafikleri
Detaylı Loss grafikleri proje raporunda mevcuttur.

<img width="846" height="547" alt="Unknown" src="https://github.com/user-attachments/assets/05c0df66-c2e7-4a02-afab-cde92cc3e29c" />

<img width="846" height="547" alt="Unknown-2" src="https://github.com/user-attachments/assets/905e967c-eea6-45b6-ab3a-f0c6a5bab3a7" />



