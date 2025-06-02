import json
import os
import queue
import sys
from datetime import datetime

import sounddevice as sd
import torch
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vosk import Model, KaldiRecognizer
import subprocess  # для запуска браузера

# ===== КОНФИГУРАЦИЯ =====
MODEL_PATH = r"C:\Voice Helper\model"
RUGPT_MODEL = r"C:\Voice Helper\rugpt3medium"  # Изменено на ruGPT3medium
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def speak(engine, text):
    """Произнесение текста"""
    print(f"Ответ: {text}")
    engine.say(text)
    engine.runAndWait()

def listen(recognizer):
    """Запись и распознавание речи"""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    print("\nСлушаю...")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result.get("text", "").lower()  # Приводим к нижнему регистру

def generate_response(pipe, prompt):
    """Генерация ответа с помощью RuGPT"""
    try:
        response = pipe(
            prompt,
            max_new_tokens=50,  # сокращаем длину генерации
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.85,
            repetition_penalty=1.2  # штраф за повторение
        )
        text = response[0]['generated_text']
        # Обрезаем исходный prompt из текста ответа (если есть)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        # На случай очень длинного ответа — ограничиваем длину вручную
        return text.split('\n')[0].strip()
    except Exception as e:
        print(f"Ошибка генерации ответа: {e}")
        return "Извините, не удалось сгенерировать ответ."

def check_system():
    """Проверка доступности ресурсов"""
    print("\n=== СИСТЕМНАЯ ИНФОРМАЦИЯ ===")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Устройство: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("==========================\n")

def init_models():
    """Инициализация всех моделей"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Модель Vosk не найдена по пути: {MODEL_PATH}. "
                f"Скачайте с https://alphacephei.com/vosk/models и распакуйте в указанную папку."
            )
        print("Инициализация Vosk...")
        vosk_model = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

        print("Инициализация TTS...")
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)

        print("\nЗагрузка RuGPT-3 Medium...")  # Изменено на RuGPT-3 Medium
        tokenizer = AutoTokenizer.from_pretrained(RUGPT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            RUGPT_MODEL,
            torch_dtype=TORCH_DTYPE
        ).to(DEVICE)

        rugpt_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1
        )

        print("Все модели успешно загружены!")
        return recognizer, engine, rugpt_pipe

    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации моделей: {str(e)}")

def handle_command(command, engine):
    """
    Обработка базовых команд программно.
    Возвращает True, если команда распознана и выполнена,
    False — если команда неизвестна.
    """
    if any(x in command for x in ["стоп", "выход", "пока", "закройся", "закрыться"]):
        speak(engine, "До свидания! Завершаю работу.")
        sys.exit(0)
    elif "время" in command:
        now = datetime.now().strftime('%H:%M')
        speak(engine, f"Сейчас {now}")
        return True
    elif "открой браузер" in command or "запусти браузер" in command:
        speak(engine, "Открываю браузер.")
        # Пример для Windows - запускаем стандартный браузер по умолчанию
        try:
            if sys.platform.startswith('win'):
                os.startfile("http://www.google.com")
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', 'http://www.google.com'])
            else:  # linux и др.
                subprocess.Popen(['xdg-open', 'http://www.google.com'])
        except Exception as e:
            speak(engine, "Не удалось открыть браузер.")
            print(f"Ошибка открытия браузера: {e}")
        return True
    # Можно добавить другие команды здесь
    return False

def main():
    check_system()

    try:
        recognizer, engine, rugpt_pipe = init_models()
        speak(engine, "Голосовой помощник запущен и готов к работе.")

        while True:
            try:
                command = listen(recognizer)
                if not command:
                    continue

                print(f"\nРаспознано: {command}")

                # Обработка базовых команд
                if handle_command(command, engine):
                    continue

                # Если команда не распознана, генерируем ответ
                answer = generate_response(rugpt_pipe, command)
                speak(engine, answer)

            except KeyboardInterrupt:
                speak(engine, "Принудительное завершение.")
                break
            except Exception as e:
                print(f"Ошибка в основном цикле: {e}")
                speak(engine, "Произошла ошибка, повторите команду.")

    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
