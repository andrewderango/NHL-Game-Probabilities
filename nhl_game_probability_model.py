import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import log_loss
import urllib.request, json
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
        self.goals_for = 0
        self.goals_against = 0
        self.record = '0-0-0'
        self.pct = 0

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

    def calc_pct(self):
        wins = int(self.record[:self.record.find('-')])
        losses = int(self.record[len(str(wins))+1:][:self.record[len(str(wins))+1:].find('-')])
        otl = int(self.record[len(str(losses))+len(str(wins))+2:])
        point_percentage = (wins*2+otl)/(len(self.team_game_list)*2)
        return point_percentage


class Game():
    def __init__(self, home_team, away_team, home_score, away_score):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score

def game_team_object_creation(games_metadf):
    total_game_list = []
    team_list = []

    for index, row in games_metadf.iterrows():
        try:
            row['Home Goals'] = float(row['Home Goals'])
            row['Away Goals'] = float(row['Away Goals'])

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
            home_team_obj.goals_for += game_obj.home_score
            away_team_obj.goals_against += game_obj.home_score
            home_team_obj.goals_against += game_obj.away_score
            away_team_obj.goals_for += game_obj.away_score
            home_team_obj.record = row['Home Record']
            away_team_obj.record = row['Away Record']
            total_game_list.append(game_obj)
        except ValueError: 
            pass

    return team_list, total_game_list

def scrape_nhl_data():
    scraped_df = pd.DataFrame(columns = ['GameID', 'Date', 'Home Team', 'Home Goals', 'Away Goals', 'Away Team', 'Home Record', 'Away Record'])

    with urllib.request.urlopen("https://statsapi.web.nhl.com/api/v1/schedule?season=20222023&gameType=R") as url:
        schedule_metadata = json.load(url)

    for dates in schedule_metadata['dates']:
        for games in dates['games']:
            if games['status']['abstractGameState'] == 'Final': #Completed games only
                scraped_df = scraped_df.append({'GameID':games['gamePk'], 'Date':dates['date'], 'Home Team':games['teams']['home']['team']['name'], 'Home Goals':games['teams']['home']['score'], 'Away Goals':games['teams']['away']['score'],'Away Team':games['teams']['away']['team']['name'], 'Home Record':f"{games['teams']['home']['leagueRecord']['wins']}-{games['teams']['home']['leagueRecord']['losses']}-{games['teams']['home']['leagueRecord']['ot']}", 'Away Record':f"{games['teams']['away']['leagueRecord']['wins']}-{games['teams']['away']['leagueRecord']['losses']}-{games['teams']['away']['leagueRecord']['ot']}"}, ignore_index = True)

    return scraped_df

def assign_power(team_list, iterations):
    for team in team_list:
        team.agd = team.calc_agd()
        team.pct = team.calc_pct()

    for iteration in range(iterations):
        # print(f'ITERATION {iteration+1}')
        for team in team_list:
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
            # print(f'{team.name}\t\tAGD: {team.calc_agd():.2f}\tSCHEDULE: {team.schedule:.2f}\t\tPOWER: {team.power:.2f}')
        for team in team_list:
            team.prev_power = team.power

def prepare_power_rankings(team_list):
    power_df = pd.DataFrame()
    for team in team_list:
        power_df = power_df.append({'Team':team.name, 'POWER':round(team.power,2), 'Record':team.record, 'PCT':f"{team.calc_pct():.3f}",'Avg Goal Differential':round(team.calc_agd(),2), 'GF/Game':f"{team.goals_for/len(team.team_game_list):.2f}", 'GA/Game':f"{team.goals_against/len(team.team_game_list):.2f}", 'Strength of Schedule':f"{team.schedule:.3f}"}, ignore_index=True)
    power_df.sort_values(by=['POWER'], inplace=True, ascending=False)
    power_df = power_df.reset_index(drop=True)
    power_df.index += 1 

    return power_df

def logistic_regression(total_game_list):
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
    
    return xpoints, ypoints, param

def model_performance(xpoints, ypoints, param):
    x_fitted = np.linspace(np.min(xpoints), np.max(xpoints), 100)
    y_fitted = 1/(1+np.exp((x_fitted)/param))

    r, p = pearsonr(xpoints, ypoints)
    print(f'Pearson Correlation of Independent and Dependent Variables: {r:.3f}')
    print(f'Log Loss of the Cumulative Distribution Function (CDF): {log_loss(ypoints, 1/(1+np.exp((xpoints)/param))):.3f}')
    print(f'Regressed Sigmoid: 1/(1+exp((x)/{param:.3f}))')
    print(f'Precise Parameter: {param}')

    plt.plot(xpoints, ypoints, 'o', color='grey')
    # plt.plot(x_fitted, y_fitted, color='black', alpha=1)
    plt.plot(x_fitted, y_fitted, color='black', alpha=1, label=f'CDF (Log Loss = {log_loss(ypoints, 1/(1+np.exp((xpoints)/param))):.3f})')
    plt.legend()
    plt.title('Logistic Regression of Team Rating Difference vs Game Result')
    plt.xlabel('Rating Difference')
    plt.ylabel('Win Probability')
    plt.show()

def calc_prob(team, opponent, param):
    return 1/(1+np.exp((team.power-opponent.power)/param))

def calc_spread(team, opponent, param, lower_bound_spread, upper_bound_spread):
    if lower_bound_spread == '-inf': 
        return 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))
    elif upper_bound_spread == 'inf': 
        return 1 - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))
    else: 
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param)) - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))

def get_todays_games(param, team_list):
    with urllib.request.urlopen("https://statsapi.web.nhl.com/api/v1/schedule?expand=schedule.linescore") as url:
        today_schedule = json.load(url)

    today_games_df = pd.DataFrame(columns = ['GameID', 'Game State', 'Home Team', 'Home Goals', 'Away Goals', 'Away Team', 'Pre-Game Home Win Probability', 'Pre-Game Away Win Probability', 'Home Record', 'Away Record'])

    date = today_schedule['dates'][0]['date']
    for games in today_schedule['dates'][0]['games']:
        for team in team_list:
            if team.name == games['teams']['home']['team']['name']:
                home_team_obj = team
            elif team.name == games['teams']['away']['team']['name']:
                away_team_obj = team

        home_win_prob = calc_prob(home_team_obj, away_team_obj, param)
        away_win_prob = 1-home_win_prob

        if games['status']['abstractGameState'] == 'Live':
            today_games_df = today_games_df.append({'GameID':games['gamePk'], 'Game State':f"{games['linescore']['currentPeriodTimeRemaining']} {games['linescore']['currentPeriodOrdinal']}", 'Home Team':games['teams']['home']['team']['name'], 'Home Goals':games['teams']['home']['score'], 'Away Goals':games['teams']['away']['score'],'Away Team':games['teams']['away']['team']['name'], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':f"{games['teams']['home']['leagueRecord']['wins']}-{games['teams']['home']['leagueRecord']['losses']}-{games['teams']['home']['leagueRecord']['ot']}", 'Away Record':f"{games['teams']['away']['leagueRecord']['wins']}-{games['teams']['away']['leagueRecord']['losses']}-{games['teams']['away']['leagueRecord']['ot']}"}, ignore_index = True)
        elif games['status']['abstractGameState'] == 'Final':
            today_games_df = today_games_df.append({'GameID':games['gamePk'], 'Game State':'Final', 'Home Team':games['teams']['home']['team']['name'], 'Home Goals':games['teams']['home']['score'], 'Away Goals':games['teams']['away']['score'],'Away Team':games['teams']['away']['team']['name'], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':f"{games['teams']['home']['leagueRecord']['wins']}-{games['teams']['home']['leagueRecord']['losses']}-{games['teams']['home']['leagueRecord']['ot']}", 'Away Record':f"{games['teams']['away']['leagueRecord']['wins']}-{games['teams']['away']['leagueRecord']['losses']}-{games['teams']['away']['leagueRecord']['ot']}"}, ignore_index = True)
        else:
            today_games_df = today_games_df.append({'GameID':games['gamePk'], 'Game State':'Pre-Game', 'Home Team':games['teams']['home']['team']['name'], 'Home Goals':games['teams']['home']['score'], 'Away Goals':games['teams']['away']['score'],'Away Team':games['teams']['away']['team']['name'], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':f"{games['teams']['home']['leagueRecord']['wins']}-{games['teams']['home']['leagueRecord']['losses']}-{games['teams']['home']['leagueRecord']['ot']}", 'Away Record':f"{games['teams']['away']['leagueRecord']['wins']}-{games['teams']['away']['leagueRecord']['losses']}-{games['teams']['away']['leagueRecord']['ot']}"}, ignore_index = True)

    return date, today_games_df

def custom_game_selector(param, team_list):
    valid = False
    while valid == False:
        home_team_input = input('Enter the home team: ')
        for team in team_list:
            if home_team_input.strip().lower() == team.name.lower().replace('é','e'):
                home_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    valid = False
    while valid == False:
        away_team_input = input('Enter the away team: ')
        for team in team_list:
            if away_team_input.strip().lower() == team.name.lower().replace('é','e'):
                away_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_probability_df = pd.DataFrame(columns = ['', home_team.name, away_team.name])

    game_probability_df = game_probability_df.append({'':'Rating', home_team.name:f'{home_team.power:.3f}', away_team.name:f'{away_team.power:.3f}'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Record', home_team.name:f'{home_team.record}', away_team.name:f'{away_team.record}'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Point PCT', home_team.name:f'{home_team.pct:.3f}', away_team.name:f'{away_team.pct:.3f}'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win Probability', home_team.name:f'{calc_prob(home_team, away_team, param)*100:.2f}%', away_team.name:f'{(calc_prob(away_team, home_team, param))*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win by 1 Goal', home_team.name:f'{calc_spread(home_team, away_team, param, 0, 1.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 0, 1.5)*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win by 2 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 1.5, 2.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 1.5, 2.5)*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win by 3 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 2.5, 3.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 2.5, 3.5)*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win by 4 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 3.5, 4.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 3.5, 4.5)*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.append({'':'Win by 5+ Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 4.5, "inf")*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 4.5, "inf")*100:.2f}%'}, ignore_index = True)
    game_probability_df = game_probability_df.set_index('')
    print()
    print(game_probability_df)

def menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list):
    while True:
        print("""--MAIN MENU--
    1. View Power Rankings
    2. View Today's Games
    3. Custom Game Selector
    4. View Model Performance
    5. View Program Performance
    6. Quit""")
    # Most/Least Consistent Teams ?
    # Individual Team Game Log (best/worst games)
    # biggest upsets
    # probability big board
    # Give users option to download csv's

        valid = False
        while valid == False:
            user_option = input('Enter a menu option: ')
            try:
                user_option = int(user_option)
                if user_option >= 1 and user_option <= 6:
                    print()
                    valid = True
                else:
                    raise ValueError
            except ValueError:
                print(f'Your option "{user_option}" is invalid.', end=' ')

        if user_option == 1:
            print(power_df)
        elif user_option == 2:
            print(today_games_df)
        elif user_option == 3:
            custom_game_selector(param, team_list)
        elif user_option == 4:
            model_performance(xpoints, ypoints, param)
        elif user_option == 5:
            print(f'Computation Time: {computation_time:.2f} seconds')
            print(f'Games Scraped: {len(total_game_list)}')
            print(f'Rate: {len(total_game_list)/computation_time:.1f} games/second')
        elif user_option == 6:
            return

        input('Press ENTER to continue\t\t')
        print()

def main():
    start_time = time.time()

    games_metadf = scrape_nhl_data()
    iterations = 10
    team_list, total_game_list = game_team_object_creation(games_metadf)
    assign_power(team_list, iterations)
    power_df = prepare_power_rankings(team_list)
    xpoints, ypoints, param = logistic_regression(total_game_list)
    date, today_games_df = get_todays_games(param, team_list)

    computation_time = time.time()-start_time
    menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list)

if __name__ == '__main__':
    main()
