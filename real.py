import os
import time
import glob
import joblib
import random
import pandas as pd
import numpy as np
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

CONFIDENCE_THRESHOLD = 50.0  
SEQUENCE_LENGTH = 5         

VALID_DEMO_ATTACKS = ['benign', 'bot', 'ddos attack-hoic', 'dos attacks-goldeneye', 'dos attacks-slowloris', 'ssh-bruteforce']

print(f"{CYAN}{'='*95}{RESET}")
print(f"{BOLD}{CYAN}  СИСТЕМА ЗА АВТОМАТИЧНО ЗАСИЧАНЕ НА КИБЕР АТАКИ (LSTM PURE INFERENCE) - NUMPY ENGINE{RESET}")
print(f"{CYAN}{'='*95}{RESET}")

# ============================================
# 1. ЗАРЕЖДАНЕ НА АРТИФАКТИТЕ
# ============================================
print(f"{YELLOW}[*] Зареждане на Оптимизираните Слоеве (NumPy Engine)...{RESET}")
try:
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl') 
    weights_dict = joblib.load('cicids_numpy_weights.pkl')
    num_features = len(feature_names)
    num_classes = len(label_encoder.classes_)
    print(f"{GREEN}[+ ] AI компонентите са заредени! Използвани характеристики: {num_features}{RESET}\n")
except Exception as e:
    print(f"{RED}[- ] Грешка при инициализация: {e}{RESET}"); exit(1)

# ============================================
# 2. МАТЕМАТИЧЕСКИ LSTM ДВИГАТЕЛ (ПОДДЪРЖА СЕКВЕНЦИИ T=5)
# ============================================
def numpy_lstm_predict(sequence_data):
    w1, u1, b1 = weights_dict['lstm_layer_1']
    units1 = w1.shape[1] // 4
    h1_prev, c1_prev = np.zeros(units1), np.zeros(units1)
    
    for t in range(SEQUENCE_LENGTH):
        xt = sequence_data[t]
        z = np.dot(xt, w1) + np.dot(h1_prev, u1) + b1
        i = 1 / (1 + np.exp(-z[:units1]))
        f = 1 / (1 + np.exp(-z[units1:2*units1]))
        c_bar = np.tanh(z[2*units1:3*units1])
        o = 1 / (1 + np.exp(-z[3*units1:]))
        c1_curr = f * c1_prev + i * c_bar
        h1_curr = o * np.tanh(c1_curr)
        h1_prev, c1_prev = h1_curr, c1_curr
        
    gamma1, beta1, mean1, var1 = weights_dict['bn_layer_1']
    h1_bn = gamma1 * ((h1_prev - mean1) / np.sqrt(var1 + 1e-3)) + beta1
    
    w2, u2, b2 = weights_dict['lstm_layer_2']
    units2 = w2.shape[1] // 4
    z2 = np.dot(h1_bn, w2) + b2
    i2 = 1 / (1 + np.exp(-z2[:units2]))
    f2 = 1 / (1 + np.exp(-z2[units2:2*units2]))
    c_bar2 = np.tanh(z2[2*units2:3*units2])
    o2 = 1 / (1 + np.exp(-z2[3*units2:]))
    h2_curr = o2 * np.tanh(i2 * c_bar2)
    
    gamma2, beta2, mean2, var2 = weights_dict['bn_layer_2']
    h2_bn = gamma2 * ((h2_curr - mean2) / np.sqrt(var2 + 1e-3)) + beta2
    
    w_out, b_out = weights_dict['output_dense']
    logits = np.dot(h2_bn, w_out) + b_out
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# ============================================
# 3. ИНТЕЛЕГЕНТНО ИЗВЛИЧАНЕ НА БАЛАНСИРАНИ ДАННИ
# ============================================
csv_files = glob.glob("*.csv") + glob.glob("datasets/*.csv") + glob.glob("IP_attack/datasets/*.csv")
csv_files = list(set(csv_files))

if not csv_files:
    print(f"{RED}[- ] Липсват CSV файлове за симулация!{RESET}"); exit(1)

# Търсим файл, който гарантирано съдържа атаки, за да не е празно демото
df_demo = None
true_labels = None
random.shuffle(csv_files)

for demo_file in csv_files:
    try:
        df_raw = pd.read_csv(demo_file, low_memory=False, encoding='latin-1', nrows=150000)
        df_raw.columns = df_raw.columns.str.strip()
        
        if 'Label' in df_raw.columns:
            df_raw['Label_str'] = df_raw['Label'].astype(str).str.strip()
            # Почистване на инфинити стойностите още тук
            df_raw = df_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            df_filtered = df_raw[df_raw['Label_str'].str.lower().isin(VALID_DEMO_ATTACKS)]
            df_filtered = df_filtered.drop_duplicates(subset=[col for col in feature_names if col in df_filtered.columns])
            
            df_benign = df_filtered[df_filtered['Label_str'].str.lower() == 'benign']
            df_attack = df_filtered[df_filtered['Label_str'].str.lower() != 'benign']
            
            # Ако файлът съдържа и двата типа трафик, го избираме
            if len(df_attack) >= 15:
                n_b = min(25, len(df_benign))
                n_a = min(25, len(df_attack))
                
                print(f"{YELLOW}[*] Симулация с балансиран файл: {os.path.basename(demo_file)}{RESET}")
                print(f"[*] Структуриране: {n_b} чисти пакета и {n_a} злонамерени пакета (Хронологично).")
                
                df_demo = pd.concat([
                    df_benign.sort_index().head(n_b), 
                    df_attack.sort_index().head(n_a)
                ]).reset_index(drop=True)
                break
    except:
        continue

# Застраховка в случай, че нито един файл няма филтрирани атаки
if df_demo is None:
    print(f"{YELLOW}[!] Използване на базов файл (нисък брой атаки): {os.path.basename(csv_files[0])}{RESET}")
    df_demo = pd.read_csv(csv_files[0], nrows=50).fillna(0)

true_labels = df_demo['Label'].astype(str).str.strip().values if 'Label' in df_demo.columns else None
df_demo = df_demo.reindex(columns=feature_names).fillna(0)

print(f"{GREEN}[+ ] Потокът е готов. Стартиране на мониторинга...{RESET}")
print(f"\n{BOLD}{'ВРЕМЕ':<10} | {'ПАКЕТЕН БУФЕР':<18} | {'УВЕРЕНОСТ':<10} | {'ПРОГНОЗА НА AI':<24} | {'СТАТУС':<15}{RESET}")
print("-" * 102)

# ============================================
# 4. СТРИЙМИНГ НА ЖИВО С ПЛЪЗГАЩ СЕ ПРОЗОРЕЦ
# ============================================
packet_buffer, packet_counter = [], 0

try:
    for idx, row in df_demo.iterrows():
        packet_df = pd.DataFrame([row], columns=feature_names)
        
        # КРИТИЧЕН ФИКС: Спираме ValueError, като гарантираме липса на безкрайности непосредствено преди скалирането
        packet_df = packet_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        packet_scaled = scaler.transform(packet_df).astype('float32')[0]
        packet_buffer.append(packet_scaled)
        
        if len(packet_buffer) < SEQUENCE_LENGTH:
            print(f"{time.strftime('%H:%M:%S'):<10} | Пакет #{idx+1:03d}          | ---        | Пълнене на буфера...     | СЪБИРАНЕ НА КОНТЕКСТ")
            time.sleep(0.1); continue
            
        if len(packet_buffer) > SEQUENCE_LENGTH: 
            packet_buffer.pop(0)
            
        current_sequence = np.array(packet_buffer, dtype='float32')
        prediction_probs = numpy_lstm_predict(current_sequence)
        
        predicted_class_idx = np.argmax(prediction_probs)
        confidence = np.max(prediction_probs) * 100
        attack_name = label_encoder.inverse_transform([predicted_class_idx])[0]
        original = true_labels[idx] if true_labels is not None else "Benign"
        
        timestamp = time.strftime("%H:%M:%S")
        packet_info = f"Window (Pkt {idx-3:02d}-{idx+1:02d})"
        
        if confidence < CONFIDENCE_THRESHOLD:
            status_color, status_text, display_name = YELLOW, "ЗА ПРОВЕРКА", "Suspicious"
        elif attack_name.lower() in ['benign', 'normal']:
            status_color, status_text, display_name = GREEN, "БЕЗОПАСЕН", "Benign (Нормален)"
        else:
            status_color, status_text, display_name = RED, "АТАКА!", attack_name

        verification = f" {GREEN}✓{RESET}" if attack_name.lower() == original.lower() else f" {RED}✗ (Реално: {original}){RESET}"
        print(f"{timestamp:<10} | {packet_info:<18} | {status_color}{confidence:>5.1f}%{RESET} | {status_color}{display_name:<24}{RESET} | {status_color}{BOLD}{status_text:<15}{RESET}{verification}")
        time.sleep(0.5)
        packet_counter += 1
except KeyboardInterrupt:
    print(f"\n{YELLOW}[!] Прекъснато от потребителя.{RESET}")