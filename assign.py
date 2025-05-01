import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    'team1_score': [180, 200, 150, 170, 190],
    'team2_score': [160, 180, 155, 175, 185],
    'team1_wickets': [4, 3, 7, 5, 2],
    'team2_wickets': [6, 5, 6, 4, 3],
    'venue_score_avg': [170, 180, 160, 170, 175],
    'winner': ['Team1', 'Team1', 'Team2', 'Team2', 'Team1']
}
df = pd.DataFrame(data)
df['winner_encoded'] = df['winner'].apply(lambda x: 0 if x == 'Team1' else 1)
X = df[['team1_score', 'team2_score', 'team1_wickets', 'team2_wickets', 'venue_score_avg']]
y = df['winner_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Prediction accuracy:", accuracy)
new_match = [[185, 170, 3, 6, 175]]
prediction = model.predict(new_match)
print("Predicted winner:", "Team1" if prediction[0] == 0 else "Team2")
