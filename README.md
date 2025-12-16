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
<img wi<img width="846" height="547" alt="Unknown" src="https://github.com/user-attachments/assets/20ba7504-1938-472b-a875-432257fbb8d2" />
dth="846" height="547" alt="Unknown-2" src="https://github.com/user-attachments/assets/49af52c8-92d0-4ebd-9814-9b81709cb9f1" />
