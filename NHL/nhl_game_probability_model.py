import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import log_loss
from tqdm import tqdm
import requests
import json
import os
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

    def calc_consistency(self):
        performance_list = []
        for game in self.team_game_list:
            if self == game.home_team:
                performance_list.append(game.away_team.power + game.home_score - game.away_score)
            else:
                performance_list.append(game.away_team.power + game.home_score - game.away_score)
        
        variance = np.var(performance_list)
        return variance

class Game():
    def __init__(self, home_team, away_team, home_score, away_score, date):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.date = date

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

            game_obj = Game(home_team_obj, away_team_obj, row['Home Goals'], row['Away Goals'], row['Date'])

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
    data = []
    team_id_dict = {}

    team_metadata = requests.get("https://api.nhle.com/stats/rest/en/team").json()
    for team in tqdm(team_metadata['data'], desc='Scraping Games', dynamic_ncols=True, colour='Green', bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):

        if team['fullName'] in ['Atlanta Thrashers', 'Hartford Whalers', 'Minnesota North Stars', 'Quebec Nordiques', 'Winnipeg Jets (1979)', 'Colorado Rockies', 'Ottawa Senators (1917)', 'Hamilton Tigers', 'Pittsburgh Pirates', 'Philadelphia Quakers', 'Detroit Cougars', 'Montreal Wanderers', 'Quebec Bulldogs', 'Montreal Maroons', 'New York Americans', 'St. Louis Eagles', 'Oakland Seals', 'Atlanta Flames', 'Kansas City Scouts', 'Cleveland Barons', 'Detroit Falcons', 'Brooklyn Americans', 'California Golden Seals', 'Toronto Arenas', 'Toronto St. Patricks', 'NHL']:
            continue

        team_id_dict[team['id']] = team['fullName']

        game_metadata = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{team['triCode']}/20232024").json()

        for game in game_metadata['games']:
            if game['gameType'] == 2 and game['gameState'] == 'OFF':
                data.append({'GameID':game['id'], 'Date':game['gameDate'], 'Home Team':game['homeTeam']['id'], 'Home Goals':game['homeTeam']['score'], 'Away Goals':game['awayTeam']['score'], 'Away Team':game['awayTeam']['id'], "FinalState":game['gameOutcome']['lastPeriodType']})

    scraped_df = pd.DataFrame(data)
    scraped_df['Home Team'] = scraped_df['Home Team'].replace(team_id_dict)
    scraped_df['Away Team'] = scraped_df['Away Team'].replace(team_id_dict)
    scraped_df = scraped_df.drop_duplicates(subset='GameID')
    scraped_df = scraped_df.sort_values(by=['GameID'])
    scraped_df = calculate_records(scraped_df) # Adds home and away record columns
    return scraped_df, team_id_dict

def calculate_records(df):
    records = {team: {'wins': 0, 'losses': 0, 'ot_losses': 0} for team in df['Home Team'].unique()}

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = row['Home Goals']
        away_goals = row['Away Goals']
        final_state = row['FinalState']

        if home_goals > away_goals:
            records[home_team]['wins'] += 1
            if final_state == 'REG':
                records[away_team]['losses'] += 1
            else:
                records[away_team]['ot_losses'] += 1
        elif home_goals < away_goals:
            if final_state == 'REG':
                records[home_team]['losses'] += 1
            else:
                records[home_team]['ot_losses'] += 1
            records[away_team]['wins'] += 1
        else:
            print(f'Critical Error: Found Tie | Infomation: {home_team} {home_goals}-{away_goals} {away_team}') # should never happen
            return

        df.loc[index, 'Home Record'] = f"{records[home_team]['wins']}-{records[home_team]['losses']}-{records[home_team]['ot_losses']}"
        df.loc[index, 'Away Record'] = f"{records[away_team]['wins']}-{records[away_team]['losses']}-{records[away_team]['ot_losses']}"

    return df

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
        power_df = pd.concat([power_df, pd.DataFrame.from_dict([{'Team':team.name, 'POWER':round(team.power,2), 'Record':team.record, 'PCT':f"{team.calc_pct():.3f}",'Avg Goal Differential':round(team.calc_agd(),2), 'GF/Game':f"{team.goals_for/len(team.team_game_list):.2f}", 'GA/Game':f"{team.goals_against/len(team.team_game_list):.2f}", 'Strength of Schedule':f"{team.schedule:.3f}"}])], ignore_index=True)
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
    x_fitted = np.linspace(np.min(xpoints)*1.25, np.max(xpoints)*1.25, 100)
    y_fitted = 1/(1+np.exp((x_fitted)/param))

    r, p = pearsonr(xpoints, ypoints)
    print(f'Pearson Correlation of Independent and Dependent Variables: {r:.3f}')
    print(f'Log Loss of the Cumulative Distribution Function (CDF): {log_loss(ypoints, 1/(1+np.exp((xpoints)/param))):.3f}')
    print(f'Regressed Sigmoid: 1/(1+exp((x)/{param:.3f}))')
    print(f'Precise Parameter: {param}')

    plt.plot(xpoints, ypoints, 'o', color='grey')
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
        if upper_bound_spread == 'inf':
            return 1
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param))
    elif upper_bound_spread == 'inf': 
        return 1 - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))
    else: 
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param)) - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))

def download_csv_option(df, filename):
    valid = False
    while valid == False:
        user_input = input('Would you like to download this as a CSV? (Y/N): ')
        if user_input.lower() in ['y', 'yes', 'y.', 'yes.']:
            valid = True
        elif user_input.lower() in ['n', 'no', 'n.', 'no.']:
            return
        else:
            print(f'Sorry, I could not understand "{user_input}". Please enter Y or N: ')

    if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
        os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
    df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
    print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')
    return


def get_todays_games(param, team_list, team_id_dict):
    today_schedule = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

    today_games_df = pd.DataFrame(columns = ['GameID', 'Game State', 'Home Team', 'Home Goals', 'Away Goals', 'Away Team', 'Pre-Game Home Win Probability', 'Pre-Game Away Win Probability', 'Home Record', 'Away Record'])

    try:
        date = today_schedule['gameWeek'][0]['date']
        for games in today_schedule['gameWeek'][0]['games']:
            for team in team_list:
                if team.name == team_id_dict[games['homeTeam']['id']]:
                    home_team_obj = team
                elif team.name == team_id_dict[games['awayTeam']['id']]:
                    away_team_obj = team

            home_win_prob = calc_prob(home_team_obj, away_team_obj, param)
            away_win_prob = 1-home_win_prob

            if games['gameState'] == 'OFF': # final
                today_games_df = pd.concat([today_games_df, pd.DataFrame.from_dict([{'GameID':games['id'], 'Game State':'Final', 'Home Team':team_id_dict[games['homeTeam']['id']], 'Home Goals':games['homeTeam']['score'], 'Away Goals':games['awayTeam']['score'], 'Away Team':team_id_dict[games['awayTeam']['id']], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':home_team_obj.record, 'Away Record':away_team_obj.record}])], ignore_index=True)
            elif games['gameState'] == 'FUT': # pre-game
                today_games_df = pd.concat([today_games_df, pd.DataFrame.from_dict([{'GameID':games['id'], 'Game State':'Pre-Game', 'Home Team':team_id_dict[games['homeTeam']['id']], 'Home Goals':0, 'Away Goals':0, 'Away Team':team_id_dict[games['awayTeam']['id']], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':home_team_obj.record, 'Away Record':away_team_obj.record}])], ignore_index=True)
            else: # in progress
                try:
                    today_games_df = pd.concat([today_games_df, pd.DataFrame.from_dict([{'GameID':games['id'], 'Game State':f"Period {games['periodDescriptor']['number']}", 'Home Team':team_id_dict[games['homeTeam']['id']], 'Home Goals':games['homeTeam']['score'], 'Away Goals':games['awayTeam']['score'], 'Away Team':team_id_dict[games['awayTeam']['id']], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':home_team_obj.record, 'Away Record':away_team_obj.record}])], ignore_index=True)
                except KeyError:
                    today_games_df = pd.concat([today_games_df, pd.DataFrame.from_dict([{'GameID':games['id'], 'Game State':f"Period {games['periodDescriptor']['number']}", 'Home Team':team_id_dict[games['homeTeam']['id']], 'Home Goals':0, 'Away Goals':0, 'Away Team':team_id_dict[games['awayTeam']['id']], 'Pre-Game Home Win Probability':f'{home_win_prob*100:.2f}%', 'Pre-Game Away Win Probability':f'{away_win_prob*100:.2f}%', 'Home Record':home_team_obj.record, 'Away Record':away_team_obj.record}])], ignore_index=True)

        today_games_df.index += 1 

    except IndexError:
        today_games_df = None
        date = None

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

    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Rating', home_team.name:f'{home_team.power:.3f}', away_team.name:f'{away_team.power:.3f}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Record', home_team.name:f'{home_team.record}', away_team.name:f'{away_team.record}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Point PCT', home_team.name:f'{home_team.pct:.3f}', away_team.name:f'{away_team.pct:.3f}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win Probability', home_team.name:f'{calc_prob(home_team, away_team, param)*100:.2f}%', away_team.name:f'{(calc_prob(away_team, home_team, param))*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 1 Goal', home_team.name:f'{calc_spread(home_team, away_team, param, 0, 1.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 0, 1.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 2 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 1.5, 2.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 1.5, 2.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 3 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 2.5, 3.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 2.5, 3.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 4 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 3.5, 4.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 3.5, 4.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 5+ Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 4.5, "inf")*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 4.5, "inf")*100:.2f}%'}])], ignore_index=True)
    game_probability_df = game_probability_df.set_index('')

    return home_team, away_team, game_probability_df

def get_upsets(total_game_list):
    upset_df = pd.DataFrame(columns = ['Home Team', 'Home Goals', 'Away Goals', 'Away Team', 'Date', 'xGD', 'GD', 'Upset Rating'])

    for game in total_game_list:
        expected_score_diff = game.home_team.power - game.away_team.power #home - away
        actaul_score_diff = game.home_score - game.away_score
        upset_rating = actaul_score_diff - expected_score_diff #Positive score is an upset by the home team. Negative scores are upsets by the visiting team.

        upset_df = pd.concat([upset_df, pd.DataFrame.from_dict([{'Home Team':game.home_team.name, 'Home Goals':int(game.home_score), 'Away Goals':int(game.away_score), 'Away Team':game.away_team.name, 'Date':game.date,'xGD':f'{expected_score_diff:.2f}', 'GD':int(actaul_score_diff), 'Upset Rating':f'{abs(upset_rating):.2f}'}])], ignore_index=True)

    upset_df = upset_df.sort_values(by=['Upset Rating'], ascending=False)
    upset_df = upset_df.reset_index(drop=True)
    upset_df.index += 1
    return upset_df

def get_best_performances(total_game_list):
    performance_df = pd.DataFrame(columns = ['Team', 'Opponent', 'GF', 'GA', 'Date', 'xGD', 'Performance'])

    for game in total_game_list:
        performance_df = pd.concat([performance_df, pd.DataFrame.from_dict([{'Team':game.home_team.name, 'Opponent':game.away_team.name, 'GF':int(game.home_score), 'GA':int(game.away_score), 'Date':game.date, 'xGD':f'{game.home_team.power-game.away_team.power:.2f}', 'Performance':round(game.away_team.power+game.home_score-game.away_score,2)}])], ignore_index = True)
        performance_df = pd.concat([performance_df, pd.DataFrame.from_dict([{'Team':game.away_team.name, 'Opponent':game.home_team.name, 'GF':int(game.away_score), 'GA':int(game.home_score), 'Date':game.date, 'xGD':f'{game.away_team.power-game.home_team.power:.2f}', 'Performance':round(game.home_team.power+game.away_score-game.home_score,2)}])], ignore_index = True)

    performance_df = performance_df.sort_values(by=['Performance'], ascending=False)
    performance_df = performance_df.reset_index(drop=True)
    performance_df.index += 1
    return performance_df

def get_team_consistency(team_list):
    consistency_df = pd.DataFrame(columns = ['Team', 'Rating', 'Consistency (z-Score)'])

    for team in team_list:
        consistency_df = pd.concat([consistency_df, pd.DataFrame.from_dict([{'Team':team.name, 'Rating':f'{team.power:.2f}', 'Consistency (z-Score)':team.calc_consistency()}])], ignore_index = True)

    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: (x-consistency_df['Consistency (z-Score)'].mean())/-consistency_df['Consistency (z-Score)'].std())

    consistency_df = consistency_df.sort_values(by=['Consistency (z-Score)'], ascending=False)
    consistency_df = consistency_df.reset_index(drop=True)
    consistency_df.index += 1
    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: f'{x:.2f}')
    return consistency_df

def team_game_log(team_list):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_log_df = pd.DataFrame(columns = ['Date', 'Opponent', 'GF', 'GA', 'Performance'])
    for game in team.team_game_list:
        if team == game.home_team:
            goals_for = game.home_score
            opponent = game.away_team
            goals_against = game.away_score
        else:
            goals_for = game.away_score
            opponent = game.home_team
            goals_against = game.home_score

        game_log_df = pd.concat([game_log_df, pd.DataFrame.from_dict([{'Date':game.date, 'Opponent':opponent.name, 'GF':int(goals_for), 'GA':int(goals_against), 'Performance':round(opponent.power + goals_for - goals_against,2)}])], ignore_index = True)
            
    game_log_df.index += 1 
    return team, game_log_df

def get_team_prob_breakdown(team_list, param):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    prob_breakdown_df = pd.DataFrame(columns = ['Opponent', 'Record', 'PCT', 'Win Probability', 'Lose by 5+', 'Lose by 4', 'Lose by 3', 'Lose by 2', 'Lose by 1', 'Win by 1', 'Win by 2', 'Win by 3', 'Win by 4', 'Win by 5+'])
    for opp_team in team_list:
        if opp_team is not team:
            prob_breakdown_df = pd.concat([prob_breakdown_df, pd.DataFrame.from_dict([{'Opponent': opp_team.name, 
            'Record': opp_team.record,
            'PCT': f'{opp_team.calc_pct():.3f}',
            'Win Probability':f'{calc_prob(team, opp_team, param)*100:.2f}%', 
            'Lose by 5+': f'{calc_spread(team, opp_team, param, "-inf", -4.5)*100:.2f}%',
            'Lose by 4': f'{calc_spread(team, opp_team, param, -4.5, -3.5)*100:.2f}%', 
            'Lose by 3': f'{calc_spread(team, opp_team, param, -3.5, -2.5)*100:.2f}%', 
            'Lose by 2': f'{calc_spread(team, opp_team, param, -2.5, -1.5)*100:.2f}%', 
            'Lose by 1': f'{calc_spread(team, opp_team, param, -1.5, 0)*100:.2f}%', 
            'Win by 1': f'{calc_spread(team, opp_team, param, 0, 1.5)*100:.2f}%', 
            'Win by 2': f'{calc_spread(team, opp_team, param, 1.5, 2.5)*100:.2f}%', 
            'Win by 3': f'{calc_spread(team, opp_team, param, 2.5, 3.5)*100:.2f}%', 
            'Win by 4': f'{calc_spread(team, opp_team, param, 3.5, 4.5)*100:.2f}%',
            'Win by 5+': f'{calc_spread(team, opp_team, param, 4.5, "inf")*100:.2f}%'}])], ignore_index = True)

    prob_breakdown_df = prob_breakdown_df.set_index('Opponent')
    prob_breakdown_df = prob_breakdown_df.sort_values(by=['PCT'], ascending=False)
    return team, prob_breakdown_df

def extra_menu(total_game_list, team_list, param):
    while True:
        print("""--EXTRAS MENU--
    1. Biggest Upsets
    2. Best Performances
    3. Most Consistent Teams
    4. Team Game Logs
    5. Team Probability Big Board
    6. Exit to Main Menu""")

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
            upsets = get_upsets(total_game_list)
            print(upsets)
            download_csv_option(upsets, 'biggest_upsets')
        elif user_option == 2:
            performances = get_best_performances(total_game_list)
            print(performances)
            download_csv_option(performances, 'best_performances')
        elif user_option == 3:
            consistency = get_team_consistency(team_list)
            print(consistency)
            download_csv_option(consistency, 'most_consistent_teams')
        elif user_option == 4:
            team, game_log = team_game_log(team_list)
            print(game_log)
            download_csv_option(game_log, f'{team.name.replace(" ", "_").lower()}_game_log')
        elif user_option == 5:
            team, team_probabilities = get_team_prob_breakdown(team_list, param)
            print(team_probabilities)
            download_csv_option(team_probabilities, f'{team.name.replace(" ", "_").lower()}_prob_breakdown')
        elif user_option == 6:
            pass

        return

def menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date):
    while True:
        print("""--MAIN MENU--
    1. View Power Rankings
    2. View Today's Games
    3. Custom Game Selector
    4. View Model Performance
    5. View Program Performance
    6. Extra Options
    7. Quit""")

        valid = False
        while valid == False:
            user_option = input('Enter a menu option: ')
            try:
                user_option = int(user_option)
                if user_option >= 1 and user_option <= 7:
                    print()
                    valid = True
                else:
                    raise ValueError
            except ValueError:
                print(f'Your option "{user_option}" is invalid.', end=' ')

        if user_option == 1:
            print(power_df)
            download_csv_option(power_df, 'power_rankings')
        elif user_option == 2:
            if today_games_df is not None:
                print(today_games_df)
                download_csv_option(today_games_df, f'{date}_games')
            else:
                print('There are no games today!')
        elif user_option == 3:
            home_team, away_team, custom_game_df = custom_game_selector(param, team_list)
            print(custom_game_df)
            download_csv_option(custom_game_df, f'{home_team.name.replace(" ", "_").lower()}_vs_{away_team.name.replace(" ", "_").lower()}_game_probabilities')
        elif user_option == 4:
            model_performance(xpoints, ypoints, param)
        elif user_option == 5:
            print(f'Computation Time: {computation_time:.2f} seconds')
            print(f'Games Scraped: {len(total_game_list)}')
            print(f'Rate: {len(total_game_list)/computation_time:.1f} games/second')
        elif user_option == 6:
            extra_menu(total_game_list, team_list, param)
        elif user_option == 7:
            return

        input('Press ENTER to continue\t\t')
        print()

def main():
    start_time = time.time()

    games_metadf, team_id_dict = scrape_nhl_data()
    iterations = 10
    team_list, total_game_list = game_team_object_creation(games_metadf)
    assign_power(team_list, iterations)
    power_df = prepare_power_rankings(team_list)
    xpoints, ypoints, param = logistic_regression(total_game_list)
    date, today_games_df = get_todays_games(param, team_list, team_id_dict)

    computation_time = time.time()-start_time
    menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date)

if __name__ == '__main__':
    main()
