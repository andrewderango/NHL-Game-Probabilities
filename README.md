# NHL/NBA Game Probabilities

This is a machine learning model determines the probability of each outcome for any NHL or NBA game. This includes the probability of each team winning, as well as the probabilities for goal spreads. All NHL data is scraped from the NHL's own API. All NBA data is scraped from [basketball-reference.com](https://www.basketball-reference.com/) via BeautifulSoup. As these sources are updated live, the predictive model updates in real-time. There are many other features available, depending on the league. The NHL's game probability model is the most robust, and offers the most features.

### Features

- Probability of each team winning for today's games
- Power rankings of each NHL/NBA team
- View biggest upsets
- View best single-game performances by a team
- View the most consistent teams in the league

## Model Accuracy

Here are the log loss values for the model over the past 3 seasons for the NHL and NBA. 
| Season         | NHL   | NBA   |
|----------------|-------|-------|
| 2023 (Ongoing) | 0.646 | 0.639 |
| 2022           | 0.633 | 0.623 |
| 2021           | 0.638 | 0.639 |
