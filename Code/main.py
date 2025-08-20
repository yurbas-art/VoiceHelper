import os
import sys
import json
import random
import queue
import re
import subprocess
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import sounddevice as sd
import torch
import soundfile as sf
from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from difflib import get_close_matches
import time

# === Настройки ===
MODEL_PATH = r"C:\\VoiceHelper\\model"
RUGPT_PATH = r"C:\\VoiceHelper\\rugpt3medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 16000
API_KEY = "113c5dcb2f6164b9797c19b806fbc21b"
CITY = "Владивосток"

# === Проверка PyTorch ===
if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 6):
    print("Пожалуйста, обновите PyTorch до версии 2.6.0 или выше")
    sys.exit(1)

# === Фразы ===
JOKES = [
    "Ты знаешь, программисты — это особый вид людей...",
    "Удалил system32, и всё стало летать! Правда в BIOS...",
    "Почему компьютер не купается? Потому что у него Windows!",
    "Сказали выбрать: жена или компьютер... Запускаюсь...",
    "Ошибка 404: анекдот не найден. Но я старалась!"
]

CALL_REPLIES = ["А?", "Я здесь!", "Да?"]
NAME_REPLIES = [
    "Я Лира, голосовой помощник. Приятно познакомиться!",
    "Меня зовут Лира. Я умею отвечать и слушать.",
    "Я Лира. Работаю даже без интернета!"
]
ABOUT_REPLIES = [
    "Я Лира. Мои создатели — Мельников Юрий и Пинчук Максим.",
    "Работаю офлайн, использую Python и Vosk.",
    "Голосовой ассистент Лира. Без интернета, но с душой."
]
IDLE_REPLIES = [
    "Я тут. Просто скажи, что нужно.",
    "Могу рассказать анекдот или включить таймер.",
    "Спроси что-нибудь!"
]

YES_WORDS = ["да", "ага", "точно", "конечно", "естественно"]
NO_WORDS = ["нет", "неа", "ни в коем случае"]

chat_history = []
commands = {}

# === Логика команд через декоратор ===
def command(trigger):
    def wrapper(func):
        commands[trigger] = func
        return func
    return wrapper

# === Помощники ===
def clean(text):
    return re.sub(r'[.,!?…]+$', '', text.strip())

def log_interaction(text):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()}: {text}\n")

def is_similar(phrase, examples):
    return get_close_matches(phrase, examples, n=1, cutoff=0.6)

def listen(recognizer):
    q = queue.Queue()
    def callback(indata, frames, time, status):
        q.put(bytes(indata))
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16', channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                return res.get("text", "").lower()

def speak(text):
    if not text.strip():
        text = random.choice(IDLE_REPLIES)
    print("Лира:", text)
    tts.tts_to_file(text=text, file_path="output.wav")
    data, samplerate = sf.read("output.wav")
    sd.play(data, samplerate)
    sd.wait()

def search_brief(query):
    try:
        html = requests.get(f"https://duckduckgo.com/html/?q={query}", headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        result = soup.find("a", class_="result__snippet")
        return result.text.strip() if result else "Не знаю ответа."
    except:
        return "Ошибка при поиске."

def generate_reply(prompt):
    context = "\n".join(chat_history[-3:] + [prompt])
    result = rugpt(context, max_new_tokens=50, do_sample=True)[0]['generated_text']
    response = clean(result.replace(prompt, "")).strip()
    endings = [" А ты что скажешь?", " Надеюсь, помогла.", " Если что, я рядом."]
    return response + random.choice(endings)

# === Команды ===

@command("как тебя зовут")
@command("твое имя")
@command("кто ты")
def cmd_name(cmd):
    speak(random.choice(NAME_REPLIES))

@command("кто тебя создал")
@command("о себе")
@command("как ты работаешь")
def cmd_about(cmd):
    speak(random.choice(ABOUT_REPLIES))

@command("анекдот")
def cmd_joke(cmd):
    speak(random.choice(JOKES))

@command("монетк")
@command("подбрось")
def cmd_coin(cmd):
    speak(random.choice(["Орёл", "Решка"]))

@command("как дела")
def cmd_mood(cmd):
    speak("Всё хорошо, спасибо!")

@command("день недели")
def cmd_day(cmd):
    days = {
        "Monday": "понедельник", "Tuesday": "вторник", "Wednesday": "среда",
        "Thursday": "четверг", "Friday": "пятница", "Saturday": "суббота", "Sunday": "воскресенье",
    }
    day = datetime.now().strftime("%A")
    speak(f"Сегодня {days.get(day, day)}")

@command("время")
@command("на часах")
@command("который час")
def cmd_time(cmd):
    now = datetime.now().strftime('%H:%M')
    speak(f"Сейчас {now}")

@command("погода")
def cmd_weather(cmd):
    try:
        data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric&lang=ru").json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        speak(f"Во {CITY} сейчас {temp} градусов, {desc}.")
    except:
        speak("Не удалось получить погоду.")

@command("открой браузер")
def cmd_browser(cmd):
    speak("Открываю браузер.")
    subprocess.Popen(["start", "http://www.google.com"], shell=True)

@command("пока")
@command("выход")
@command("стоп")
def cmd_exit(cmd):
    speak("Пока!")
    sys.exit(0)

# === Обработка команд ===
def handle_cmd(cmd):
    for trigger, func in commands.items():
        if trigger in cmd:
            func(cmd)
            return True
    if any(q in cmd for q in ["что такое", "кто такой", "что значит"]):
        q = re.sub(r"(что такое|кто такой|что значит)", "", cmd).strip()
        res = search_brief(q)
        speak(f"{q.capitalize()} — это {clean(res)}")
        return True
    return False

# === Главная функция ===
def main():
    print("Загрузка...")
    print("Устройство:", DEVICE)

    vosk_model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    global rugpt, tts
    tokenizer = AutoTokenizer.from_pretrained(RUGPT_PATH)
    model = AutoModelForCausalLM.from_pretrained(RUGPT_PATH, torch_dtype=TORCH_DTYPE).to(DEVICE)
    rugpt = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

    from TTS.api import TTS
    tts = TTS(model_name="tts_models/ru/v3_1_ru", progress_bar=False, gpu=torch.cuda.is_available())

    speak("Привет! Я Лира. Чем могу помочь?")

    while True:
        try:
            cmd = listen(recognizer)
            print("Вы сказали:", cmd)
            log_interaction(f"Ты: {cmd}")

            if "лира" in cmd:
                cmd = cmd.replace("лира", "").strip()

            if handle_cmd(cmd):
                continue

            if not cmd or len(cmd.split()) < 2:
                continue

            reply = generate_reply(cmd)
            chat_history.append(f"Ты: {cmd}")
            chat_history.append(f"Лира: {reply}")
            log_interaction(f"Лира: {reply}")
            speak(reply)

        except KeyboardInterrupt:
            speak("Завершаю.")
            break
        except Exception as e:
            print("Ошибка:", e)
            speak("Произошла ошибка.")

if __name__ == "__main__":
    main()
