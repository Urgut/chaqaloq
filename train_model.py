import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dummy ma'lumotlar (ovozdan olingan xususiyatlar)
X = np.random.rand(100, 40)  # 100 ta audio namunasi, 40 ta MFCC xususiyati
y = np.random.choice(['hungry', 'sleepy', 'discomfort'], size=100)  # Tasodifiy yorliqlar

# Modelni o'qitish
model = RandomForestClassifier()
model.fit(X, y)

# Modelni saqlash
joblib.dump(model, 'baby_cry_model.pkl')
print("Model muvaffaqiyatli saqlandi: baby_cry_model.pkl")
