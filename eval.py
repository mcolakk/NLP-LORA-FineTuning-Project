# eval.py
# ==============================================================================
# NLP PROJESİ - BENCHMARK TEST BAŞLATICI
# Bu script, eğitilen modelleri CodeGen (LiveCodeBench) aracı ile test eder.
# Platform: AtCoder | Zorluk: Easy
# ==============================================================================

import os
import sys
import subprocess

def run_benchmark(model_type):
    """
    Belirtilen model tipi için CodeGen benchmark testini başlatır.
    """
    print(f"\n{'='*50}")
    print(f"🚀 TEST BAŞLATILIYOR: {model_type}")
    print(f"{'='*50}")
    
    # PDF'te istenen komut yapısı [cite: 78-79]
    # --model_type: deep_instruction veya diverse_instruction
    # --platform: atcoder
    # --difficulty: easy
    
    command = [
        "python", "CodeGen/livecodebench_eval.py",
        "--model_type", model_type,
        "--platform", "atcoder",
        "--difficulty", "easy"
    ]
    
    try:
        # Komutu çalıştır ve çıktıları ekrana yansıt
        subprocess.run(command, check=True)
        print(f"✅ {model_type} testi tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"❌ HATA: {model_type} testi başarısız oldu.")
        print("Lütfen 'CodeGen' klasörünün kurulu olduğundan emin olun.")

if __name__ == "__main__":
    # Eğer CodeGen klasörü yoksa uyar
    if not os.path.exists("CodeGen"):
        print("UYARI: 'CodeGen' klasörü bulunamadı.")
        print("Testi çalıştırmadan önce: git clone https://github.com/naholav/CodeGen.git")
        print("Ve gerekli kurulumları yaptığınızdan emin olun.")
    
    # 1. DEEP Modelini Test Et
    run_benchmark("deep_instruction")
    
    # 2. DIVERSE Modelini Test Et
    run_benchmark("diverse_instruction")