import sys
import subprocess

def main_menu():
    while True:
        print("\n===== YÜZ İŞLEME SİSTEMİ =====")
        print("1. Duygu Tanıma")
        print("2. Yüz Tanıma (İsim)")
        print("3. Yüz Kaydetme")
        print("4. Çıkış")
        
        try:
            choice = input("\nSeçiminizi yapın (1-4): ")
            
            if choice == '1':
                subprocess.run([sys.executable, "emotion.py"])
            elif choice == '2':
                subprocess.run([sys.executable, "name.py"])
            elif choice == '3':
                subprocess.run([sys.executable, "recognize.py"])
            elif choice == '4':
                print("Programdan çıkılıyor...")
                break
            else:
                print("Geçersiz seçim! Lütfen 1-4 arası bir sayı girin.")
        except:
            print("\nProgram sonlandırıldı.")
            break

if __name__ == "__main__":
    main_menu()