import librosa
import numpy as np
import joblib
import telebot
import os

import config.TOKEN

# Telegram boti uchun token
BOT_TOKEN = 'TOKEN'
bot = telebot.TeleBot(BOT_TOKEN)

# Chaqaloq yig'isini tahlil qilish uchun tayyor modelni yuklash
model = joblib.load('baby_cry_model.pkl')

# Audio fayldan ovozni o'qish funksiyasi
def load_audio_from_file(file_path, sr=22050):
    print(f"{file_path} fayli yuklanmoqda...")
    audio, _ = librosa.load(file_path, sr=sr)
    print("Fayl yuklandi!")
    return audio

# Xususiyatlarni ajratib olish funksiyasi
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Chaqaloq yig'isini aniqlash funksiyasi
def predict_cry(audio):
    features = extract_features(audio)
    prediction = model.predict([features])[0]

    if prediction == 'hungry':
        return "Chaqaloq och bo'lishi mumkin."
    elif prediction == 'sleepy':
        return "Chaqaloq uxlamoqchi bo'lishi mumkin."
    elif prediction == 'discomfort':
        return "Chaqaloq bezovta."
    else:
        return "Noma'lum sabab."

# /start komandasiga javob berish
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Salom! Iltimos, audio faylni yuklang yoki ovozli xabar yozing.")

# Audio qabul qilish va tahlil qilish
@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio(message):
    # Foydalanuvchiga kutilayotganligi haqida xabar berish
    bot.reply_to(message, "Iltimos kuting, natija tez orada ma'lumot qilinadi...")

    try:
        # Faylni yuklab olish
        file_info = bot.get_file(message.audio.file_id if message.audio else message.voice.file_id)
        file_path = file_info.file_path
        downloaded_file = bot.download_file(file_path)

        # Lokalga saqlash (vaqtinchalik)
        save_path = "temp_audio.ogg"
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Audioni tahlil qilish
        audio = load_audio_from_file(save_path)
        result = predict_cry(audio)

        # Natijani yuborish
        bot.reply_to(message, result)
    except Exception as e:
        bot.reply_to(message, "Kechirasiz, audio tahlil qilishda xatolik yuz berdi.")
        print(f"Xatolik: {e}")
    finally:
        # Vaqtinchalik faylni o'chirish
        if os.path.exists(save_path):
            os.remove(save_path)

# Botni ishga tushirish
if __name__ == "__main__":
    print("Bot ishga tushdi!")
    bot.polling()
