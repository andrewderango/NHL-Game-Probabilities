# NHL/NBA Game Probabilities

This is a machine learning model determines the probability of each outcome for any NHL or NBA game. This includes the probability of each team winning, as well as the probabilities for goal spreads. All NHL data is scraped from the [NHL's own API](https://statsapi.web.nhl.com/api/v1/schedule). All NBA data is scraped from [basketball-reference.com](https://www.basketball-reference.com/) via BeautifulSoup. As these sources are updated live, the predictive model updates in real-time. There are many other features available that show the user interesting data about the league (listed below), depending on the league. The NHL's game probability model is the most robust of the leagues below, and offers the most features.

### Feature Availability

| Feature                                            | NHL     | NBA     | Custom Leagues |
|----------------------------------------------------|:-------:|:-------:|:--------------:|
| Team Power Rankings                                | &check; | &check; |     &check;    |
| Game Probabilities for Today's Games               | &check; | &check; |                |
| Live Scores for Today's Games                      | &check; |         |                |
| Game Probabilities a Game Between any 2 Teams      | &check; | &check; |     &check;    |
| View Biggest Single-Game Upsets                    | &check; | &check; |     &check;    |
| View Best Single-Game Team Performances            | &check; | &check; |     &check;    |
| Most Consistent Teams                              | &check; | &check; |     &check;    |
| Team Game Logs                                     | &check; | &check; |     &check;    |
| Win Probability for a Team Against all Other Teams | &check; | &check; |     &check;    |
| View Model Accuracy                                | &check; | &check; |     &check;    |
| Download CSV's                                     | &check; | &check; |     &check;    |

## Model Accuracy

Here are the log loss values for the model over the past 3 seasons for the NHL and NBA. 
| Season         | NHL   | NBA   |
|----------------|-------|-------|
| 2023           | 0.640 | 0.631 |
| 2022           | 0.633 | 0.623 |
| 2021           | 0.638 | 0.639 |

## Featues
The following is a brief explanation of some of the core features. 

### Live Scraping 
The program scrapes data from the NHL API every time it is run, so the data is updated automatically. The time it takes to scrape all game data is up to 3 seconds and depends on how late in the season it is.

### Power Rankings
The following is an image of the final power rankings for the 2022-23 NHL season. Shows the team's POWER ranking, record, goal differential, strength of schedule, etc.
<img width="1261" alt="Screenshot 2023-10-24 at 4 24 35 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/9fb2d900-09f7-4d23-8045-83edb306dd5c">

### Today's Games
The following is an image of the game probabilities for all the games being played today. The home and away scores are updated live.
<img width="1261" alt="Screenshot 2023-10-24 at 4 26 17 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/b0ce4b27-30bd-4b5d-8b58-927709baa361">

### Custom Game Selector
The following is an example of the output of the custom game selector. The user enters a home and away team, and the program shows the probability of each team winning and the probability of them winning by _n_ goals.
<img width="1267" alt="Screenshot 2023-10-24 at 4 29 23 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/6211f913-b9d0-400f-a152-4233815e041e">
