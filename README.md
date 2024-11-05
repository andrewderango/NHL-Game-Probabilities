# Game Outcome Prediction Model

This repository presents a real-time machine learning-based predictive modeling framework designed to estimate the probabilities of game outcomes based on historical scores and team performance metrics for any head-to-head sport. The core of the model revolves around the Iterative Convergence Power Algorithm (ICPA), which performs a convergence-driven iterative refinement process to optimize team strength estimators. These refined team strength estimators are then employed as features in a novel variant of a logistic regression model.

The model effectively predicts various game outcomes, including the probabilities of each team winning, as well as the likelihood of specific goal spreads and margins of victory or defeat. By integrating advanced statistical algorithms with unique machine learning adaptations and real-time data processing, this framework provides an elegant and robust solution for analyzing various match outcomes and the competitive landscape of the league.

## Iterative Convergence Power Algorithm

The Iterative Convergence Power Algorithm (ICPA) is an elegant but powerful method designed to refine team strength estimators through a convergence-based iterative process. It operates by evaluating each team's historical performance in relation to their opponents, adjusting the power values assigned to teams based on their game outcomes. The algorithm iteratively updates these estimators, utilizing feedback from previous iterations to enhance accuracy and stability. This continuous refinement allows the model to dynamically adapt to the changing competitive landscape, ensuring that the team strengths reflect the most current performance data. By serving as robust features for a novel variant of logistic regression, the ICPA plays a crucial role in predicting game outcomes, enabling precise estimations of win probabilities, goal spreads, and margins of victory or defeat. Its ability to converge upon stable and reliable team strength assessments makes it a key component of the predictive modeling framework.

## Scope

While primarily focused on the NHL and NBA, the model also includes native support for additional leagues, such as the KHL, AHL, and CFB, allowing for versatile applications across various competitive contexts. Users can seamlessly integrate their own league data by inputting a spreadsheet of game scores, enabling the model to adapt to new datasets and making it a flexible tool for analyzing different leagues and competitions. By harnessing the power of historical performance data and employing sophisticated statistical techniques, this model serves as an advanced tool for sports analysts, data scientists, and enthusiasts looking to gain insights into game outcome probabilities across multiple leagues.

## Specifications

### Feature Availability

| Feature                                            | NHL     | NBA     | CFB     | KHL     | AHL     | Custom Leagues |
|----------------------------------------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:--------------:|
| Team Power Rankings                                | &check; | &check; | &check; | &check; | &check; |     &check;    |
| Game Probabilities for Today's Games               | &check; | &check; |         |         |         |                |
| Live Scores for Today's Games                      | &check; |         |         |         |         |                |
| Game Probabilities a Game Between any 2 Teams      | &check; | &check; | &check; | &check; | &check; |     &check;    |
| View Biggest Single-Game Upsets                    | &check; | &check; | &check; | &check; | &check; |     &check;    |
| View Best Single-Game Team Performances            | &check; | &check; | &check; | &check; | &check; |     &check;    |
| Most Consistent Teams                              | &check; | &check; | &check; | &check; | &check; |     &check;    |
| Team Game Logs                                     | &check; | &check; | &check; | &check; | &check; |     &check;    |
| Win Probability for a Team Against all Other Teams | &check; | &check; | &check; | &check; | &check; |     &check;    |
| View Model Accuracy                                | &check; | &check; | &check; | &check; | &check; |     &check;    |
| Download CSV's                                     | &check; | &check; | &check; | &check; | &check; |     &check;    |

### Data Sources

| League         | Data Source(s) |
|----------------|-------------------------------------------------------------------------------|
| NHL            | [NHL API](https://api-web.nhle.com/v1/schedule/now) |
| NBA            | [Basketball Reference](https://www.basketball-reference.com/) |
| CFB            | [FlashScore](https://www.flashscore.ca/football/usa/ncaa/results/) |
| KHL            | [FlashScore](https://www.flashscore.ca/hockey/russia/khl/results/) |
| AHL            | [FlashScore](https://www.flashscore.ca/hockey/usa/ahl/results/) |
| Custom Leagues | User-uploaded CSV |


### Model Accuracy

Here are the log loss values for the model over the past 3 seasons for the NHL and NBA. 
| Season         | NHL   | NBA   |
|----------------|-------|-------|
| 2023           | 0.639 | 0.631 |
| 2022           | 0.633 | 0.623 |
| 2021           | 0.638 | 0.639 |

## Featues
The following is a brief explanation of some of the core features. Note that the CSV's outputted by the program are included in the files for each of the leagues in this repo.

### Live Scraping 
The program scrapes data from the NHL API every time it is run, so the data is updated automatically. The time it takes to scrape all game data is up to 3 seconds and depends on how late in the season it is.

### Power Rankings
The following is an image of the final power rankings for the 2022-23 NHL season. Shows the team's POWER ranking, record, goal differential, strength of schedule, etc.
<img width="1261" alt="Screenshot 2023-10-24 at 4 24 35 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/9fb2d900-09f7-4d23-8045-83edb306dd5c">

### Today's Games
The following is an image of the game probabilities for all the games being played today. The home and away scores are updated live.
<img width="1261" alt="Screenshot 2023-10-24 at 4 26 17 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/b0ce4b27-30bd-4b5d-8b58-927709baa361">

### Custom Game Selector
The following is an example of the output of the custom game selector. The user enters a home and away team, and the program shows the probability of each team winning and the probability of them winning by _n_ goals based on the model.
<img width="1267" alt="Screenshot 2023-10-24 at 4 29 23 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/6211f913-b9d0-400f-a152-4233815e041e">

### Biggest Upsets
This returns a CSV of every game played in the season, sorted from the biggest upset to the smallest. The quantification of an upset is done by subtracting the actual goal differential of the game by the expected goal differential, based on the model. Below, it seems that the biggest upset of the 2022-23 season was the Colorado Avalanche's 7-0 victory over the Ottawa Senators on January 14th, 2023. They were expected to win by 0.69 goals but won by 7.
<img width="1261" alt="Screenshot 2023-10-24 at 4 36 10 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/b77c5c12-f025-4890-8389-39306e4ba9e5">

### Best Performances
This returns a CSV of every team's games, their opponent, the score, and the team's performance. Their performance is determined by the goal differential of the game, adjusting for the strength of their opponent in that game. Below, it appears that the Boston Bruins' 7-0 victory over the Buffalo Sabres on March 19th, 2023 was the best team performance of the season while Montreal's 9-2 loss to Washington 
on New Year's Eve 2022 was the worst.
<img width="1250" alt="Screenshot 2023-10-24 at 5 16 26 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/f9d8c839-562c-4d7a-9bcb-c35279c83187">

### Team Consistency Ratings
The option shows how consistent teams were. This is defined as how much their actual performances differed from their expected performances, or how good the model was at predicting scores for their games. From the image below, it seems that the Minnesota Wild were the most consistent team, and the Detroit Red Wings were the least.
<img width="1259" alt="Screenshot 2023-10-24 at 4 38 19 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/544c1552-328b-46d0-95a2-20a2aacc0d57">

### Team Game Logs
This option shows every game that a team has played, and how it affected their POWER score. This value can also be seen as how well they performed in that game, hence the column title "Performance". The program prompts the user for which team's game log to display.
<img width="1257" alt="Screenshot 2023-10-24 at 4 42 16 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/96445af4-ee43-49fa-bfb4-b4b63ef9d154">

### Team Probability Big Board
Entering this option prompts the user to enter a team. Then, a table is returned where each row is a different team, and the table shows the probability that the entered team will beat the team in the given row and the probability of the team betting the team in the row by _n_ goals. For the example below, the program suggests that the Toronto Maple Leafs have a 75.33% chance of beating the Montreal Canadiens, and that the probability of TOR beating MTL by 5+ goals is 10.64%.
<img width="1257" alt="Screenshot 2023-10-24 at 4 44 09 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/4817b99d-7e65-4525-a378-cc19ea846a7d">

### View Model Performance
Returns a Matplotlib plot encapsulating the model's predictions and accuracy. The grey dots represent true instances of games, while the black line is the model's prediction of the probability that the home team will win. The function representing the sigmoid is then printed to the terminal, which can be useful to employ the model elsewhere and in debugging. This is the result for the 2022-23 NHL season:
<img width="639" alt="Screenshot 2023-10-24 at 4 52 25 PM" src="https://github.com/andrewderango/NHL-Game-Probabilities/assets/93727693/9a7e5019-8e37-488e-acaa-7ff910e31e8c">

## Update Log
**February 1, 2023**: Addition of NHL Game Probability Model<br>
**February 3, 2023**: Addition of NBA Game Probability Model<br>
**February 3, 2023**: Addition of Custom League Probability Model<br>
**April 14, 2023**: Pandas deprecated ```pd.DataFrame.append()```, replaced with ```concat()```<br>
**November 10, 2023**: NHL API was shut down and replaced. The API lost an endpoint with all games in a season, so now the program has to loop through each team's schedule and make 30x more API calls which increased runtime from ~1s to ~15s.<br>
**November 4, 2024**: Added custom league support for the KHL, AHL, and CFB.