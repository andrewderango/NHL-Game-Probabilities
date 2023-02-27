# NHL/NBA Game Probabilities

This is a machine learning model determines the probability of each outcome for any NHL or NBA game. This includes the probability of each team winning, as well as the probabilities for goal spreads. All NHL data is scraped from the NHL's own API. All NBA data is scraped from [basketball-reference.com](https://www.basketball-reference.com/) via BeautifulSoup. As these sources are updated live, the predictive model updates in real-time. There are many other features available that show the user interesting data about the league (listed below), depending on the league. The NHL's game probability model is the most robust, and offers the most features.

### Features

| Feature                                            | NHL   | NBA | Custom Leagues |
|----------------------------------------------------|-------|-----|----------------|
| Team Power Rankings                                | - [x] | [x] | [x]            |
| Game Probabilities for Today's Games               | [x]   | [x] | [ ]            |
| Live Scores for Today's Games                      | [x]   | [ ] | [ ]            |
| Game Probabilities a Game Between any 2 Teams      | [x]   | [x] | [x]            |
| View Biggest Single-Game Upsets                    | [x]   | [x] | [x]            |
| View Best Single-Game Team Performances            | [x]   | [x] | [x]            |
| Most Consistent Teams                              | [x]   | [x] | [x]            |
| Team Game Logs                                     | [x]   | [x] | [x]            |
| Win Probability for a Team Against all Other Teams | [x]   | [x] | [x]            |
| View Model Accuracy                                | [x]   | [x] | [x]            |
| Download CSV's                                     | [x]   | [x] | [x]            |

| Unchecked | Checked |
| --------- | ------- |
| &#9744;   | &#9745; |

## Model Accuracy

Here are the log loss values for the model over the past 3 seasons for the NHL and NBA. 
| Season         | NHL   | NBA   |
|----------------|-------|-------|
| 2023 (Ongoing) | 0.646 | 0.639 |
| 2022           | 0.633 | 0.623 |
| 2021           | 0.638 | 0.639 |
