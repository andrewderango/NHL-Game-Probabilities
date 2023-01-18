import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import time

class Team():
    def __init__(self, name):
        self.name = name
        self.team_game_list = []
        self.agd = 0
        self.opponent_power = []
        self.schedule = 0
        self.power = 0
        self.prev_power = 0

    def read_games(self):
        print(f'--{self.name.upper()} GAME LIST--')
        for game in self.team_game_list:
            print(f'{game.home_team.name}  {game.home_score}-{game.away_score}  {game.away_team.name}')

    def calc_agd(self):
        goal_differential = 0
        for game in self.team_game_list:
            if self == game.home_team: 
                goal_differential += game.home_score - game.away_score
            else: 
                goal_differential += game.away_score - game.home_score
        agd = goal_differential / len(self.team_game_list)

        return agd

    def calc_sched(self):
        self.opponent_power = []
        for game in self.team_game_list:
            if self == game.home_team:
                self.opponent_power.append(game.away_team.prev_power)
            else:
                self.opponent_power.append(game.home_team.prev_power)

        return sum(self.opponent_power) / len(self.opponent_power)

    def calc_power(self):
        return self.calc_sched() + self.agd

class Game():
    def __init__(self, home_team, away_team, home_score, away_score):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score

def game_team_object_creation(df):
    total_game_list = []
    team_list = []

    for index, row in df.iterrows():
        team_in_list = False
        for team in team_list:
            if team.name == row['Home Team']:
                team_in_list = True
                home_team_obj = team
        if team_in_list == False: 
            home_team_obj = Team(row['Home Team'])
            team_list.append(home_team_obj)

        team_in_list = False
        for team in team_list:
            if team.name == row['Away Team']:
                team_in_list = True
                away_team_obj = team
        if team_in_list == False: 
            away_team_obj = Team(row['Away Team'])
            team_list.append(away_team_obj)

        game_obj = Game(home_team_obj, away_team_obj, row['Home Goals'], row['Away Goals'])

        home_team_obj.team_game_list.append(game_obj)
        away_team_obj.team_game_list.append(game_obj)
        total_game_list.append(game_obj)

    return team_list, total_game_list

def assign_power(team_list, iterations):
    for team in team_list:
        team.agd = team.calc_agd()

    for iteration in range(iterations):
        # print(f'ITERATION {iteration+1}')
        for team in team_list:
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
            # print(f'{team.name}\t\tAGD: {team.calc_agd():.2f}\tSCHEDULE: {team.schedule:.2f}\t\tPOWER: {team.power:.2f}')
        for team in team_list:
            team.prev_power = team.power

def logistic_regression(total_game_list, display_results):
    xpoints = [] # Rating differential (Home - Away)
    ypoints = [] # Home Win/Loss Boolean (Win = 1, Tie = 0.5, Loss = 0)

    for game in total_game_list:
        xpoints.append(game.home_team.power - game.away_team.power)

        if game.home_score > game.away_score:
            ypoints.append(1)
        elif game.home_score < game.away_score:
            ypoints.append(0)
        else:
            ypoints.append(0.5)

    parameters, covariates = curve_fit(lambda t, param: 1/(1+np.exp((t)/param)), [-x for x in xpoints], ypoints) # Regression only works if parameter is positive.
    param = -parameters[0]

    if display_results == True:
        x_fitted = np.linspace(np.min(xpoints), np.max(xpoints), 100)
        y_fitted = 1/(1+np.exp((x_fitted)/param))

        r, p = pearsonr(xpoints, ypoints)

        if p > 0.1: significance = 'No' 
        elif p > 0.05: significance = 'Weak'
        elif p > 0.05: significance = 'Weak'
        elif p > 0.01: significance = 'Moderate'
        elif p > 0.005: significance = 'Good'
        elif p > 0.001: significance = 'Strong'
        else: significance = 'Very strong'

        print(f'\nPearson Correlation of Independent and Dependent Variables: {r:.3f}')
        print(f'Significance of Correlation (p-value): {p:.5f}\t({significance} evidence against the null hypothesis)')
        print(f'R² of Regressed Sigmoid Function: {r2_score(ypoints, 1/(1+np.exp((xpoints)/param))):.3f} | 1/(1+exp((x)/{param:.3f}))')

        plt.plot(xpoints, ypoints, 'o', color='grey')
        plt.plot(x_fitted, y_fitted, color='black', alpha=1, label=f'Sigmoid (R² = {r2_score(ypoints, 1/(1+np.exp((xpoints)/param))):.3f})')
        plt.legend()
        plt.title('Logistic Regression of Team Rating Difference vs Game Result')
        plt.xlabel('Rating Difference')
        plt.ylabel('Game Result (Binary)')
        plt.show()
    
    return param

def calc_prob(home_team, away_team, param):
    return 1/(1+np.exp((home_team.power-away_team.power)/param))

def main():
    start_time = time.time()
    filename = 'game_data.csv'
    df = pd.read_csv(f'/Users/andrewderango/Documents/Programming Files/Game Probabilities/{filename}')
    iterations = 1000

    team_list, total_game_list = game_team_object_creation(df)
    assign_power(team_list, iterations)

    print('--FINAL RESULTS--')
    for team in team_list:
        print(f'{team.name}\t\tAGD: {team.calc_agd():.2f}\tSCHEDULE: {team.schedule:.2f}\t\tPOWER: {team.power:.2f}')

    param = logistic_regression(total_game_list, True)

    print(f'Time Elapsed: {time.time()-start_time:.2f} seconds')

if __name__ == '__main__':
    main()