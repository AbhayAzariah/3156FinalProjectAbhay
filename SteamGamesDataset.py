import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# JSON File
file_path = '/Users/abhayazariah/Downloads/3156/games.json'

#Load the JSON file
dataset = {}
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        text = fin.read()
        if len(text) > 0:
            dataset = json.loads(text)
else:
    print(f"games.json file not found at {file_path}.")
    exit()

# Process data into a DataFrame
data = []
for app in dataset:
    game = dataset[app]
    data.append({
        'AppID': app,
        'Name': game.get('name', ''),
        'Categories': ', '.join(game.get('categories', [])),
        'MedianPlaytimeForever': game.get('median_playtime_forever', 0),
        'Developers': ', '.join(game.get('developers', [])),
        'PositiveVotes': game.get('positive', 0),
        'NegativeVotes': game.get('negative', 0),
        'MetacriticScore': game.get('metacritic_score', 0),
        'RequiredAge': game.get('required_age', 0),
    })

df = pd.DataFrame(data)

#Filter for AAA Developers
aaa_developers = [
    'Nintendo', 'Valve Corporation', 'Rockstar Games', 'Electronic Arts',
    'Activision Blizzard', 'Sony Computer Entertainment', 'Ubisoft',
    'Sega Games Co. Ltd', 'BioWare', 'Naughty Dog Inc', 'Square Enix Holdings Co. Ltd',
    'Capcom Company Ltd', 'Bungie Inc', 'Microsoft Corporation',
    'Bandai Namco Entertainment', 'Mojang', 'Epic Games', 'Game Freak',
    'Insomniac Games Inc', 'Infinity Ward', 'Take-Two Interactive Software Inc',
    'Gameloft', 'ZeniMax Media Inc', 'NCSOFT', 'Blizzard Entertainment Inc',
    'Zynga', 'Nexon Co. Ltd', 'Konami Holdings Corporations',
    'Bethesda Game Studios', 'Double Fine Productions Inc.', 'id Software',
    'Rare', 'Retro Studios', 'Sonic Team', 'LucasArts', 'Level-5 Company',
    'Atari', 'Thatgamecompany LLC', 'Beenox', '1C Company', 'EA Maxis',
    'Polyphony Digital', 'EA Dice', 'Intelligent Systems Co. Ltd',
    'SCE Santa Monica Studio', 'EA Canada', 'PopCap Games', 'Petroglyph Games',
    'Relic', 'Treasure Co. Ltd'
]
df = df[df['Developers'].apply(lambda x: any(dev in x for dev in aaa_developers))]

#Classify Multiplayer vs Singleplayer
df = df[df['Categories'].notna()]
df['GameMode'] = df['Categories'].apply(
    lambda x: 'Multiplayer' if 'Multi-Player' in x else 'Singleplayer'
)
df['GameModeEncoded'] = df['GameMode'].apply(lambda x: 1 if x == 'Multiplayer' else 0)

#Drop Missing Values
df = df.dropna(subset=['PositiveVotes', 'MetacriticScore', 'RequiredAge', 'MedianPlaytimeForever'])

#Regression Analysis (Random Forest)
X_reg = df[['PositiveVotes', 'MetacriticScore', 'GameModeEncoded', 'RequiredAge']]
y_reg = df['MedianPlaytimeForever']

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = rf_model.predict(X_test_reg)

# Evaluate Regression Model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"\nRegression Analysis:")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Feature Importance
importance = rf_model.feature_importances_
print("\nFeature Importance (Regression):")
for feature, score in zip(X_reg.columns, importance):
    print(f"{feature}: {score:.4f}")


bins = [0, 100, 500, 1000, float('inf')]
labels = ['Short', 'Medium', 'Long', 'Very Long']
df['PlaytimeCategory'] = pd.cut(df['MedianPlaytimeForever'], bins=bins, labels=labels)

# Prepare features and target
X_class = df[['PositiveVotes', 'MetacriticScore', 'GameModeEncoded', 'RequiredAge']]
y_class = df['PlaytimeCategory']

# Encode target
y_class_encoded = y_class.astype('category').cat.codes

# Train-test split
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class_encoded, test_size=0.2, random_state=42)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_class, y_train_class)

# Predictions
y_pred_class = nb_model.predict(X_test_class)

# Evaluate Classification Model
unique_classes = np.unique(y_class_encoded)
print("\nClassification Analysis:")
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class, labels=unique_classes, target_names=labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_class, labels=unique_classes))
