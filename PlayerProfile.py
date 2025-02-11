
import gc

# # Get a list of all global variables
# globals_list = list(globals().keys())


# # Display in Streamlit

# # Delete everything except built-in variables
# for name in globals_list:
#     if name[0] != "_" and name not in ['st','gc', 'memory_usage']:  # Avoid deleting built-in variables
#         #print(name)
#         del globals()[name]
import time
import os



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager

from statsbombpy import sb
import plotly.express as px
import plotly.graph_objects as go
# from streamlit_plotly_events import plotly_events
from PIL import Image

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


print('')
print('new run')
print('')
creds = {"user": "rdell@racingloufc.com", "passwd": "8CStqFOa"}

import gc

from PIL import Image, ImageOps
import io
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

from matplotlib import rcParams

import psutil

# Display in Streamlit

    
#from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.font_manager as fm

# fontManager.addfont(font_path)
# sen_font = FontProperties(fname=font_path).get_name()
# rcParams['font.family'] = sen_font
rcParams['text.color'] = 'white' 



# file_name = 'InternationalWomensData.parquet'
# df = pd.read_parquet(file_name)
st.set_page_config( 
    page_title="Player Profile",
    page_icon=":checkered_flag:",
    layout="centered",
    initial_sidebar_state="expanded"   
)


import os
regular_font_path = 'Montserrat-Regular.ttf'
bold_font_path = 'Montserrat-Bold.ttf'

process = psutil.Process(os.getpid())
cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU percentage
memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
disk_usage = psutil.disk_usage('/').used / (1024**3)  # Convert to GB


bold = fm.FontProperties(fname=bold_font_path)
regular = fm.FontProperties(fname=regular_font_path)


custom_css = f"""
<style>
video {{
    width: 100%;
    height: auto;
}}
@font-face {{
    font-family: 'Montserrat';
    src: url('file://{regular_font_path}') format('truetype');
    font-weight: normal;
}}
@font-face {{
    font-family: 'Montserrat';
    src: url('file://{bold_font_path}') format('truetype');
    font-weight: bold;
}}
html, body, [class*="css"] {{
    font-family: 'Montserrat', sans-serif;
    background-color: #400179;
    color: #ffffff;
}}
.sidebar .sidebar-content {{
    background-color: #400179;
}}
img[data-testid="stLogo"] {{
            height: 3rem;
}}



</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

start_graphic = False
pos_list = ['CBs', 'WBs', 'CMs', 'AMs', 'Ws', 'STs', 'GKs']

league_list = ['NWSL', 'USL', 'England', 'Spain', 'Germany', 'France', 'Sweden', 'Brazil', 'Mexico', 'MLS', 'MLS Next Pro']
 

# img = Image.open(image_path)


# st.image(img, caption="600x3000 Image", use_column_width=True)

# dpi = 100  # Dots per inch
# width, height = 800 / dpi, 1800 / dpi 

# # Create a figure
# fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

# ax.set_facecolor("black")
# ax.axis("off")
#st.pyplot(fig)

#gc.collect()

raw_cols = ['Player',
 'pos_group',
 'Team',
 'Minutes',
 'Number',
 'Foot',
 'player_id',
 'Position',
 'Detailed Position',
 'Key Passes',
 'Big Chances Created',
 'Pass OBV',
 'Passes into Final Third',
 'Passes into Box',
 'Progressive Passes',
 'Through Passes',
 '% of Passes Progressive',
 '% of Passes Forward',
 'xA',
 'Assists',
 'Long Passes Completed',
 'Long Pass %',
 'Crosses Completed into Box',
 'Cross into Box %',
 'Cross Shot Assists',
 'Cross Assists',
 'Dribble OBV',
 'Carries',
 'Take Ons',
 'Dribble %',
 'Progressive Carries',
 'Progressive Carries in Final Third',
 'Carries into Final Third',
 'Carries in Final Third',
 'Final Third Take Ons',
 'Turnovers Leading to Shot',
 'Forward Pass %',
 'Backward Pass %',
 '% of Passes Backward',
 'Left Pass %',
 '% of Passes Left',
 'Right Pass %',
 '% of Passes Right',
 'Short Pass %',
 'Ball Retention %',
 'Pressured Pass %',
 'Fouls Won',
 'Passes Completed',
 'Passes Received',
 'Passes Received Under Pressure',
 'Final Third Receptions',
 'Zone 14 Receptions',
 'Box Receptions',
 'Six Yard Box Receptions',
 'Goals',
 'Shots',
 'xG',
 'xG/Shot',
 'Goal Conversion',
 'xGOT',
 'xGOT per xG',
 'Goals per xG',
 'Big Chances',
 'Big Chance Conversion',
 'Blocks',
 'Tackle %',
 'Tackles Won',
 'Goals Conceded after Defensive Action',
 'Dribbled Past',
 'Defensive OBV',
 'Ball Recoveries',
 'Pressures',
 'Counterpressures',
 'Defensive Third Blocks',
 'Defensive Third Tackle %',
 'Defensive Third Tackles Won',
 'Defensive Third Dribbled Past',
 'Defensive Third Tackle to Dribbled Past Ratio',
 'Defensive Third Defensive OBV',
 'Defensive Third Ball Recoveries',
 'Defensive Third Pressures',
 'Defensive Third Counterpressures',
 'Attacking Third Pressures',
 'Attacking Third Counterpressures',
 'Interceptions',
 'Attacking Third Tackles',
 'Attacking Third Interceptions',
 'Average Defensive Action Distance',
 'Attacking Half Pressures',
 'Attacking Half Counterpressures',
 'Attacking Half Tackles',
 'Attacking Half Interceptions',
 'Attacking Half Pressure Regains',
 'Pressure Regains Leading to Shots',
 'Aerial Wins',
 'Aerial %',
 'Attacking SP Aerial Wins',
 'Defensive SP Aerial Wins',
 'Attacking SP Aerial %',
 'Defending SP Aerial %',
 'Defensive Third Interceptions',
 'GK Shots Faced',
 'GK Shots on Target Faced',
 'GK Goals Conceded',
 'GK Saves',
 'GK xG Against',
 'GK xGOT Against',
 'GK 1v1s Save Rate',
 'GK Avg. Distance',
 'GK Save %',
 'Goals Prevented',
 'Big Chances Faced',
 'Big Chances Saved',
 'Big Chances Save %',
 'PSV-99',
 'PSV-85',
 'Distance',
 'HI Running Distance',
 'Sprinting Distance',
 'Count High Acceleration',
 'Count High Deceleration']


metric_replacements = {
    'Average Defensive Action Distance': 'Avg. Def. Distance',
    'Passes into Final Third': 'Final 1/3 Entries',
    'Progressive Passes': 'Prog Passes',
    'Prog Carries': 'Prog Carries',
    'Cross into Box %': 'Cross %',
    'Attacking Third Pressures': 'Att. 1/3 Pressures',
    'Attacking Half Pressure Regains': 'Pressure Regains',
    'Big Chance Conversion': 'Big Chance Conv.',
    'Big Chances Created': 'Chances Created',
    'PSV-99': 'Top Speed',
    'PSV-85': '2nd Top Speed',
    'GK Save %': 'Save %',
    'GK Avg. Distance': 'Avg. Def. Dist.',
    'GK Shots on Target Faced': 'Shots OT Faced',
    'Big Chances Save %': 'Big Chance Save %',
    'GK 1v1s Save Rate': '1v1 Save %',
    'Crosses Completed into Box': 'Crosses Completed',
    'Progressive Carries in Final Third': 'Att. 1/3 Carries',
    'Passes Received Under Pressure': 'Pressured Receptions',
    'Defensive Third Blocks': 'Def. 1/3 Blocks',
    'Defensive Third Interceptions': 'Def. 1/3 Ints',
    'Defensive Third Tackles Won': 'Def. 1/3 Tackles',
    'Defensive Third Tackle to Dribbled Past Ratio': 'Def. 1/3 Tackle %',
    'Defensive Third Ball Recoveries': 'Def. 1/3 Ball Recovs.',
    'Defensive Third Pressures': 'Def. 1/3 Pressures',
    'Defensive Third Counterpressures': 'Def. 1/3 Counterpressures',
    'Attacking Third Pressures': 'Att. 1/3 Pressures',
    'Attacking Third Counterpressures': 'Att. 1/3 Counterpressures',
    'Attacking Third Tackles': 'Att. 1/3 Tackles',
    'Attacking Third Interceptions': 'Att. 1/3 Ints',
    'Attacking Half Pressures': 'Att. 1/2 Pressures',
    'Attacking Half Counterpressures': 'Att. 1/2 Counterpressures',
    'Count High Acceleration': 'High Accels',
    'Count High Deceleration': 'High Decels'

    



}
team_replacements = {
    "El Paso Locomotive": "El Paso",
    "San Diego Loyal": "San Diego",
    'Tampa Bay Rowdies': 'Tampa Bay',
    'Sporting Kansas City II': 'Sporting KC II',
    'Loudoun United': 'Loudoun',
    'Memphis 901': 'Memphis',
    'Tacoma Defiance': 'Tacoma',
    'Hartford Athletic': 'Hartford',
    #'Miami FC'
    'Birmingham Legion': 'Birmingham',
    #'Real Monarchs',
    'Pittsburgh Riverhounds': 'Pittsburgh',
    'Atlanta United II':'Atlanta II',
    #'San Antonio',
    'Orange County SC': 'Orange County',
    'Louisville City': 'LouCity',
    'New York RB II': 'NYRB II',
    'Philadelphia Union II': 'Union II',
    #'LA Galaxy II',
    'Phoenix Rising': 'Phoenix',
    #'Austin Bold',
    'Rio Grande Valley': 'RGV',
    'Las Vegas Lights': 'LV Lights',
    #'Saint Louis',
    #'Reno 1868',
    #'North Carolina',
    'New Mexico United': 'New Mexico',
    'Portland Timbers II': 'Portland II',
    #'OKC Energy',
    'Charlotte Independence': 'Charlotte Ind.',
    'Sacramento Republic': 'Sacramento',
    'Charleston Battery': 'Charleston',
    #'FC Tulsa',
    'VfL Wolfsburg WFC': 'Wolfsburg',
    'TSG 1899 Hoffenheim': 'Hoffenheim',
    'Eintracht Frankfurt': 'Frankfurt',
    'TSV Bayer 04 Leverkusen': 'Bayer 04',
    'NJ/NY Gotham FC': 'Gotham',
    'North Carolina Courage': 'NC Courage',
    'Seattle Reign': 'Seattle',
    'Orlando Pride': 'Orlando',
    'Washington Spirit': 'Washington',
    #'Kansas City',
    #'Houston Dash',
    'Chicago Red Stars': 'Chicago',
    'Portland Thorns': 'Portland',
    'Racing Louisville FC': 'Racing',
    'San Diego Wave': 'San Diego',
    'Utah Royals': 'Utah'
}
import re 

def safe_div(x,y):
    if y == 0: return 0
    else: return x/y
def replace_team_names(text, replacements):
    if not isinstance(text, str):
        return text  # Return non-string values unchanged
    
    # Sort replacements by length (longest first) to avoid partial matches
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Create a regular expression pattern
    pattern = '|'.join(re.escape(key) for key, _ in sorted_replacements)
    
    # Replace function
    def replace_func(match):
        matched_text = match.group()
        for key, value in sorted_replacements:
            if re.search(re.escape(key), matched_text, re.IGNORECASE):
                return value
        return matched_text  # If no replacement found, return the original text

    # Perform the replacement
    return re.sub(pattern, replace_func, text, flags=re.IGNORECASE)


def calculate_age(birthdate: str) -> float:
    # Convert the birthdate string into a datetime object
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
    
    # Get the current date
    today = datetime.today()
    
    # Calculate the difference in years
    age = (today - birthdate).days / 365.25  # 365.25 accounts for leap years
    
    # Return the age rounded to 1 decimal point
    return round(age, 1)

def cm_to_feet_inches(cm: float) -> str:
    if np.isnan(cm): return 'NA'
    inches_total = round(cm / 2.54)  # Convert cm to total inches
    feet = inches_total // 12  # Get whole feet
    inches = inches_total % 12  # Get remaining inches
    return f"{feet}'{inches}"

def concat_unique(series):
    return ', '.join(series.astype(str).unique())

def concat_season(series):
    unique_series = series.astype(str).unique()
    if len(unique_series) > 1:
        return f"{unique_series[0]} - {unique_series[-1]}"
    else:
        return unique_series[0]


def concat_not_unique(series):
    return ', '.join(series.astype(str))

def normalize(series):
    series = pd.to_numeric(series, errors='coerce')  # Convert to numeric, forcing non-numeric values to NaN
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(1, index=series.index)  # All values are the same or all NaN
    return (series - min_val) / (max_val - min_val)

def cosine_sim(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


position_mapping = {
            "Goalkeeper": 'GK',
            "Left Back": 'LB',
            "Left Wing Back": 'LWB',
            "Right Back": 'RB',
            "Right Wing Back": 'RWB',
            "Center Back": 'CB',
            "Right Center Back": 'RCB',
            "Left Center Back": 'LCB',
            "Center Defensive Midfield": 'CDM',
            "Left Defensive Midfield": 'LDM',
            "Right Defensive Midfield": 'RDM',
            "Left Center Midfield": 'LCM',
            "Right Center Midfield": 'RCM',
            "Left Midfield": 'LM',
            "Left Attacking Midfield": 'LAM',
            "Left Wing": 'LW',
            "Right Midfield": 'RM',
            "Right Attacking Midfield": 'RAM',
            "Right Wing": 'RW',
            "Center Attacking Midfield": 'CAM',
            "Center Forward": 'CF',
            "Left Center Forward": 'LCF',
            "Right Center Forward": 'RCF',
            "Striker": 'CF'
        }

season_mapping = {
    '2021': '2021',
    '2022': '2022',
    '2023': '2023',
    '2024': '2024',
    '2025': '2025',
    '20/21': '2021',
    '21/22': '2022',
    '22/23': '2023',
    '23/24': '2024',
    '24/25': '2025'
}



poss_seasons = ['2021','2022','2023','2024','2025','20/21','21/22','22/23','23/24','24/25']



file_name = 'CombinedAppData.parquet'

import pandas as pd
import streamlit as st

#@st.cache_data
def load_data():
    return pd.read_parquet(file_name)

data1 = load_data()  # Cached, so it doesn’t reload every time


#data1 = pd.read_parquet(file_name)
data_copy = data1.copy(deep=True)

#data1['file_season'] = data1['Season'].apply(lambda x: season_mapping.get(x, x))
data1 = data1[data1['Competition'].isin(league_list)]
data1 = data1[data1['Season'].isin(poss_seasons)]


with st.sidebar:
    #st.image(img_path, width=90)  # Adjust the width as needed
    league = st.selectbox(
            'Select League',league_list)
    
    league_index = league_list.index(league)
    
    
    data1 = data1[data1['Competition'] == league].sort_values(by = 'Player')
    player = st.selectbox(
            'Select Player', data1['Player'].unique())
    
    data1 = data1[data1['Player'] == player]
    season_options = sorted(data1['Season'].unique())


    player_id = data1.iloc[0]['offline_player_id']
    del data1
    

    seasons = st.pills("Select Seasons", season_options, selection_mode = "multi", default = season_options[-1])
    mapped_seasons = list(map(lambda x: season_mapping.get(x, x), seasons))

    # print(seasons)
    # print(mapped_seasons)

    

    player_data = pd.DataFrame()
    comp_data = pd.DataFrame()

    events = pd.DataFrame()
    max_poss_matches = 0

    if len(seasons) > 0:
        max_mapped_season = max(mapped_seasons)
        for s in mapped_seasons:
            p_events = pd.read_parquet(f"{league}{s}-AppLeagueEvents.parquet")

            teams = p_events[p_events['player_id'] == player_id]['team'].unique()
            
            max_poss_matches += p_events[p_events['team'].isin(teams)].groupby('player_id')['match_id'].nunique().max()
            #print(max_poss_matches)
            p_events = p_events[p_events['player_id'] == player_id]
            s_filename = f"{league}{s}-AppPlayerSeasonPercentiles.parquet"
            sdata = pd.read_parquet(s_filename)
            sdata['Season'] = s
            scomp_data = sdata.copy(deep=True)
            #sdata = sdata[sdata['Player'] == player]
            sdata = sdata[sdata['player_id'] == player_id]
            if s == max_mapped_season:
                recent_player_data = sdata.copy(deep = True)
            player_data = pd.concat([player_data, sdata])
            comp_data = pd.concat([comp_data, scomp_data])

            events = pd.concat([events, p_events])

            
            #data1['file_season'] = data1['Season'].apply(lambda x: season_mapping.get(x, x))
        
        pos_map = {
            1: 'GK',
            3: 'FB/WB',
            4: 'CB',
            6: 'CM',
            10: 'AM',
            7: 'W',
            9: 'ST'

        }
        player_data['Position Group'] = player_data['pos_group'].apply(lambda x: pos_map.get(x, x))
        comp_data['Position Group'] = comp_data['pos_group'].apply(lambda x: pos_map.get(x, x))
        recent_player_data['Position Group'] = recent_player_data['pos_group'].apply(lambda x: pos_map.get(x, x))  
        data_copy['Position Group'] = data_copy['pos_group'].apply(lambda x: pos_map.get(x, x))

        

        #print(len(events))

        
        

 

        # data1 = data1[data1['Season'].isin(seasons)]

        position_minutes = player_data.groupby('Position Group')['Minutes'].sum().to_dict()
        #position_labels = [f"{position} ({position_minutes[position]} mins)" for position in sorted(player_data['Position Group'].unique())]
        position_labels = [
                f"{position} ({position_minutes[position]} mins)" 
                for position in sorted(player_data['Position Group'].unique(), key=lambda pos: position_minutes[pos], reverse=True)
        ]



        positions = st.pills("Select Positions", 
                             position_labels, #sorted(player_data['Position Group'].unique()), 
                             selection_mode = "multi", default = position_labels[0])
        
        positions = [label.split(' ')[0] for label in positions]


        

        
     
        if len(positions) > 0:
            start_graphic = True

            player_data = player_data[player_data['Position Group'].isin(positions)]
            comp_data = comp_data[comp_data['Position Group'].isin(positions)]
            median_minutes = np.median(comp_data['Minutes'])
            comp_data = comp_data[(comp_data['Minutes'] > median_minutes) & (comp_data['player_id'] != player_id)]

            
            

            comp_data = pd.concat([comp_data, player_data])

            events = events[events['Position Group'].isin(positions)]
            # comp_data['Team'] = comp_data['Team'].apply(lambda x: replace_team_names(x, team_replacements))
            # #print(comp_data.loc[comp_data['player_id'] == player_id, 'Team'])
            # team_name = comp_data.loc[comp_data['player_id'] == player_id, 'Team'].values[0]


            comp_data = comp_data.sort_values(by = ['Season', 'Minutes'], ascending = [False, False])

            comp_data['Shortened Position'] = comp_data['Detailed Position'].apply(lambda x: position_mapping.get(x, x))  

       
            recent_player_data = recent_player_data[recent_player_data['Position Group'].isin(positions)]

            max_mapped_season = max(mapped_seasons)

            #player_map_file = f'/Users/malekshafei/Desktop/Louisville/Player Mapping/player-mapping{league}{max_mapped_season}.json'
            player_map_file = f'Player Mapping/player-mapping{league}{max_mapped_season}.json'
            player_mapping = pd.read_json(player_map_file)
            #print(player_mapping)

            
            player_nickname = player_mapping[player_mapping['offline_player_id'] == player_id].iloc[0]['player_nickname']
            if player_nickname == None: player_nickname = player

            player_birthdate = player_mapping[player_mapping['offline_player_id'] == player_id].iloc[0]['player_birth_date']
            player_age = calculate_age(player_birthdate)
            

            player_raw_height = player_mapping[player_mapping['offline_player_id'] == player_id].iloc[0]['player_height']
            player_height = cm_to_feet_inches(player_raw_height)


            #comp_data['Tackle to Dribbled Past Ratio'] = round(safe_div(comp_data['Tackle to Dribbled Past Ratio'], comp_data['Tackle to Dribbled Past Ratio'] + 1) * 100,1)
            #comp_data['Tackle to Dribbled Past Ratio'] = round(comp_data['Tackle to Dribbled Past Ratio'] / (comp_data['Tackle to Dribbled Past Ratio'] + 1) * 100, 1)
            
            comp_data.drop(['Tackle %'], axis = 1)
            comp_data['Tackle %'] = round(comp_data['Tackle to Dribbled Past Ratio'] / (comp_data['Tackle to Dribbled Past Ratio'] + 1) * 100, 1)
            comp_data.drop(['Tackle to Dribbled Past Ratio'], axis = 1)
            


            #comp_data['Distance'] = comp_data['Distance'].fillna(0).astype(int)
            comp_data['PSV-99'] = comp_data['PSV-99'].fillna(comp_data.groupby('player_id')['PSV-99'].transform('mean'))
            comp_data['PSV-85'] = comp_data['PSV-85'].fillna(comp_data.groupby('player_id')['PSV-85'].transform('mean'))
            comp_data['Distance'] = comp_data['Distance'].fillna(comp_data.groupby('player_id')['Distance'].transform('mean'))
            comp_data['HI Running Distance'] = comp_data['HI Running Distance'].fillna(comp_data.groupby('player_id')['HI Running Distance'].transform('mean'))
            comp_data['Sprinting Distance'] = comp_data['Sprinting Distance'].fillna(comp_data.groupby('player_id')['Sprinting Distance'].transform('mean'))
            comp_data['Count High Acceleration'] = comp_data['Count High Acceleration'].fillna(comp_data.groupby('player_id')['Count High Acceleration'].transform('mean'))
            comp_data['Count High Deceleration'] = comp_data['Count High Deceleration'].fillna(comp_data.groupby('player_id')['Count High Deceleration'].transform('mean'))

            
            


            card_options = ['Shot Map', 'Key Passes', 'Ball Carrying', 'Progressive Actions', 'Touch Map', 'Pressure Map', 'Radar']

            selected_card = st.pills("Selected Visuals",
                                      card_options, default = 'Radar'
                                      
                                      )
            
            

            special_cols = ['Player', 'pos_group', 'Team', 'Minutes', 'Number', 'Foot', 'player_id', 'Position', 'Detailed Position', 'Shortened Position','Season']
            ratio_cols = ['% of Passes Progressive', '% of Passes Forward', 'Long Pass %', 'Cross into Box %', 'Dribble %', 
                          'Forward Pass %', 'Backward Pass %','% of Passes Backward','Left Pass %','% of Passes Left','Right Pass %','% of Passes Right','Short Pass %','Ball Retention %','Pressured Pass %',
                          'xG/Shot','Goal Conversion','xGOT per xG','Goals per xG','Big Chance Conversion','Tackle %','Tackle %','Defensive Third Tackle %','Average Defensive Action Distance','Aerial %','Attacking SP Aerial %','Defending SP Aerial %','GK 1v1s Save Rate','GK Avg. Distance','GK Save %','Big Chances Save %','PSV-99','PSV-85',]
            
            

            print(comp_data[comp_data['player_id'] == player_id][['Player', 'Season', 'Position Group', 'Minutes', 'PSV-99']])
            for col in raw_cols:
                if col not in special_cols:
                    comp_data[col] = comp_data[col] * comp_data['Minutes']


            aggs = {col: 'sum' for col in raw_cols if col not in special_cols}
            aggs['Position Group'] = concat_unique
            aggs['pos_group'] = concat_unique
            aggs['Team'] = 'first'
            aggs['Minutes'] = 'sum'
            aggs['Number'] = 'first'
            aggs['Foot'] = 'first'
            aggs['player_id'] = 'first'
            aggs['Position'] = 'first'
            aggs['Detailed Position'] = 'first'
            aggs['Shortened Position'] = concat_unique
            aggs['Season'] = concat_season
            # aggs['PSV-99'] = 'max'
            # aggs['PSV-85'] = 'max'
            # aggs['Distance'] = 'max'
            # aggs['HI Running Distance'] = 'max'
            # aggs['Sprinting Distance'] = 'max'
            # aggs['Count High Acceleration'] = 'max'
            # aggs['Count High Deceleration'] = 'max'

            #aggs['Matches'] = 'size'
            
            comp_data = comp_data.groupby('Player').agg(aggs).reset_index()

            for col in raw_cols:
                if col not in special_cols:
                    comp_data[col] = comp_data[col] / comp_data['Minutes']
                    comp_data[f'pct{col}'] = round(comp_data[col].rank(pct=True) * 100,2)

            #print('after')
            #print(comp_data[comp_data['player_id'] == player_id][['Player', 'Season', 'Position Group', 'Minutes', 'PSV-99']])
            

            
            #comp_data['GK_Difficulty_Faced_Raw'] = (0.3 * comp_data['pctGK Shots on Target Faced']) + (0.2 * comp_data['pctGK xG Against']) + (0.2 * comp_data['pctGK xGOT Against']) + (0.3 * comp_data['pctBig Chances Faced'])
            comp_data['Shot Stopping'] = (0.05 * comp_data['pctGK Shots on Target Faced']) + (0.1 * comp_data['pctBig Chances Save %']) + (0.7 * comp_data['pctGoals Prevented']) + (0.15 * comp_data['pctGK Save %'])
            comp_data['Short Distribution'] = (0.15 * comp_data['pctForward Pass %']) + (0.2 * comp_data['pctPressured Pass %']) + (0.65 * comp_data['pctShort Pass %'])
            comp_data['Long Distribution'] = (0.2 * comp_data['pctProgressive Passes']) + (0.1 * comp_data['pctPasses into Final Third']) +  (0.35 * comp_data['pctLong Passes Completed']) + (0.1 * comp_data['pctPass OBV']) + (0.25 * comp_data['pctLong Pass %'])
            comp_data['Stepping Out'] = (0.1 * comp_data['pctGK Avg. Distance'])
            comp_data['Saving Big Chances'] = (0.25 * comp_data['pctBig Chances Faced']) + (0.75 * comp_data['pctBig Chances Save %'] )
            comp_data['1v1 Saving'] = (1 * comp_data['pctGK 1v1s Save Rate'])
            
            comp_data['Chance Creation'] = (0.3 * comp_data['pctxA']) + (0.15 * comp_data['pctKey Passes']) + (0.25 * comp_data['pctBig Chances Created']) + (0.1 * comp_data['pctPass OBV']) + (0.2 * comp_data['pctAssists'])
            comp_data['Ball Progression'] = (0.2 * comp_data['pctPass OBV']) + (0.15 * comp_data['pctPasses into Final Third']) + (0.25 * comp_data['pctProgressive Carries']) + (0.3 * comp_data['pctProgressive Passes']) + (0.1 * comp_data['pctLong Passes Completed'])
            comp_data['Ball Retention'] = (0.15 * comp_data['pctForward Pass %']) + (0.55 * comp_data['pctBall Retention %']) + (0.2 * comp_data['pctPressured Pass %']) + (0.1 * comp_data['pctShort Pass %'])
            comp_data['Verticality'] = (0.25 * comp_data['pct% of Passes Progressive']) + (0.6 * comp_data['pct% of Passes Forward']) + (0.15 * (1- comp_data['pct% of Passes Backward']))
            comp_data['Carrying'] = (0.25 * comp_data['pctTake Ons']) + (0.1 * comp_data['pctCarries']) + (0.5 * comp_data['pctProgressive Carries']) + (0.15 * comp_data['pctDribble %'])
            comp_data['Poaching'] = (0.55 * comp_data['pctxG']) + (0.2 * comp_data['pctxG/Shot']) + (0.2 * comp_data['pctBox Receptions']) + (0.05 * comp_data['pctSix Yard Box Receptions'])
            comp_data['Finishing'] = (0.25 * comp_data['pctGoals per xG']) + (0.15 * comp_data['pctxGOT per xG']) + (0.45 * comp_data['pctGoal Conversion']) + (0.15 * comp_data['pctGoals'])
            comp_data['Goal Threat'] = (0.3 * comp_data['pctxG']) + (0.3 * comp_data['pctGoals']) + (0.2 * comp_data['pctBox Receptions']) + (0.1 * comp_data['pctGoal Conversion']) + (0.1 * comp_data['pctxGOT per xG'])
            comp_data['Crossing'] = (0.3 * comp_data['pctCrosses Completed into Box']) + (0.2 * comp_data['pctCross into Box %']) + (0.25 * comp_data['pctCross Shot Assists']) + (0.25 * comp_data['pctCross Assists'])
            comp_data['Heading'] = (0.7 * comp_data['pctAerial %']) + (0.3 * comp_data['pctAerial Wins'])
            comp_data['Set Piece Threat'] = (0.75 * comp_data['pctAttacking SP Aerial Wins']) + (0.25 * comp_data['pctAttacking SP Aerial %'])

            comp_data['Speed'] = (0.4 * comp_data['pctPSV-99']) + (0.45 * comp_data['pctPSV-85']) + (0.15 * comp_data['pctSprinting Distance'])
            comp_data['HSR Distance'] = (0.4 * comp_data['pctDistance']) + (0.6 * comp_data['pctHI Running Distance'])

            comp_data['High Pressing'] = (0.25 * comp_data['pctAttacking Half Pressures']) + (0.15 * comp_data['pctAttacking Third Pressures']) + (0.2 * comp_data['pctAttacking Half Pressure Regains']) + (0.1 * comp_data['pctPressure Regains Leading to Shots']) + (0.3 * comp_data['pctAverage Defensive Action Distance'])
            comp_data['Defending High'] = (0.2 * comp_data['pctAttacking Half Pressures']) + (0.05 * comp_data['pctAttacking Half Pressure Regains']) + (0.75 * comp_data['pctAverage Defensive Action Distance'])
            comp_data['Tackle Accuracy'] = (0.2 * (100 - comp_data['pctDribbled Past'])) + (0.65 * comp_data['pctTackle %']) + (0.15 * comp_data['pctTackles Won'])
            comp_data['Defensive Output'] = (0.1 * comp_data['pctBlocks']) + (0.3 * comp_data['pctTackles Won']) + (0.4 * comp_data['pctBall Recoveries']) + (0.2 * comp_data['pctInterceptions'])
            comp_data['Receiving Forward'] = (0.6 * comp_data['pctFinal Third Receptions']) + (0.15 * comp_data['pctBox Receptions']) + (0.15 * comp_data['pctShots']) + (0.1 * comp_data['pctxG'])

            rating_cols = ['Chance Creation', 'Crossing','Ball Progression', 'Ball Retention', 'Verticality', 'Carrying', 'Poaching', 'Finishing', 'Goal Threat', 'Heading', 'Set Piece Threat', 'Speed', 'HSR Distance', 'High Pressing', 'Defending High', 'Tackle Accuracy', 'Defensive Output', 'Receiving Forward']

            for x in rating_cols:
                comp_data[x] = round(comp_data[x].rank(pct=True) * 100,2)


            important_metrics = []
            selected_metrics = []

            if 'GK' in positions:
                important_metrics.append(['Shot Stopping', 'Short Distribution', 'Long Distribution', 'Stepping Out', 'Saving Big Chances', '1v1 Saving'])
                selected_metrics.append(['Goals Prevented', 'GK Save %',
                                         'GK Avg. Distance', 'GK Shots on Target Faced',
                                         'Big Chances Save %', 'GK 1v1s Save Rate',
                                         'Short Pass %', 'Forward Pass %',
                                         'Long Pass %', 'Pressured Pass %',
                                         'Progressive Passes', 'Passes into Final Third'])

            if 'CB' in positions:
                #print('CB')
                important_metrics.append(['Speed','Tackle Accuracy', 'Defending High', 'Defensive Output', 'Heading', 'Set Piece Threat', 'Ball Retention', 'Ball Progression', 'Verticality'])
                selected_metrics.append(['PSV-99', 'Average Defensive Action Distance', 
                                         'Tackle %', 'Tackles Won',
                                         'Interceptions', 'Blocks',
                                         'Aerial Wins', 'Aerial %',
                                         'Progressive Passes', 'Passes into Final Third', 
                                         'Progressive Carries', 'Ball Retention %'])
            if 'FB/WB' in positions:
                #print('FB/WB')
                important_metrics.append(['Crossing','Chance Creation','Receiving Forward','Speed', 'High Pressing', 'Tackle Accuracy', 'Defensive Output',  'Ball Retention', 'Ball Progression', 'Carrying', 'HSR Distance', 'Goal Threat', 'Verticality', 'Heading',])
                selected_metrics.append(['PSV-99', 'Distance',
                                         'Key Passes', 'Big Chances Created',
                                         'xA', 'Assists',
                                         'Cross Shot Assists', 'Cross into Box %',
                                         'Tackle %', 'Tackles Won',
                                         'Attacking Third Pressures', 'Attacking Half Pressure Regains'
                                         ])
            if 'CM' in positions:
                #print('CM')
                important_metrics.append(['Tackle Accuracy', 'Defensive Output', 'High Pressing', 'Heading', 'Set Piece Threat', 'Ball Retention', 'Ball Progression', 'Carrying','Receiving Forward','Chance Creation','Speed', 'HSR Distance', 'Goal Threat', 'Verticality'])
                selected_metrics.append(['PSV-99', 'Distance',
                                         'xA', 'Assists',
                                         'xG', 'Goals',
                                         'Progressive Passes', 'Progressive Carries',
                                         'Tackle %', 'Ball Recoveries',
                                         'Attacking Third Pressures', 'Attacking Half Pressure Regains'
                                         ])
                

            if 'AM' in positions:
                #print('AM')
                important_metrics.append(['Chance Creation', 'Carrying', 'Speed', 'Goal Threat', 'Defensive Output', 'High Pressing', 'Chance Creation', 'Ball Progression', 'Ball Retention', 'Crossing',  'HSR Distance', 'Poaching', 'Finishing'])
                selected_metrics.append(['PSV-99', 'Distance',
                                        'Goals', 'xG',
                                        'Box Receptions', 'Goal Conversion',
                                        'Assists', 'xA',
                                        'Progressive Carries', 'Dribble %',
                                        'Attacking Third Pressures', 'Ball Recoveries'])
            if 'W' in positions:
                #print('W')
                important_metrics.append(['Chance Creation', 'Carrying', 'Speed', 'Goal Threat', 'Defensive Output', 'High Pressing', 'Chance Creation', 'Ball Progression', 'Ball Retention', 'Crossing',  'HSR Distance', 'Poaching', 'Finishing'])
                selected_metrics.append(['PSV-99', 'Distance',
                                        'Goals', 'xG',
                                        'Box Receptions', 'Goal Conversion',
                                        'Assists', 'xA',
                                        'Progressive Carries', 'Dribble %',
                                        'Attacking Third Pressures', 'Ball Recoveries'])
                
            if 'ST' in positions:
                #print('ST')
                important_metrics.append(['Finishing', 'Poaching', 'High Pressing', 'Speed', 'HSR Distance', 'Defensive Output', 'Chance Creation', 'Ball Retention', 'Carrying',  'Goal Threat', 'Heading', 'Set Piece Threat'])
                selected_metrics.append(['Goals', 'xG',
                                        'Shots', 'Box Receptions',
                                        'Goal Conversion', 'Big Chance Conversion',
                                        'Key Passes', 'Ball Retention %',
                                        'Attacking Third Pressures', 'Ball Recoveries',
                                        'PSV-99', 'Distance'])

            #important_metrics = list(set(item for sublist in important_metrics for item in sublist))

            all_ratings = ['Speed', 'HSR Distance', 'Defensive Output', 'Defending High', 'High Pressing', 'Tackle Accuracy', 'Heading', 'Ball Retention', 'Ball Progression', 'Verticality', 'Carrying', 'Chance Creation', 'Crossing', 'Poaching', 'Finishing', 'Goal Threat', 'Receiving Forward', 'Set Piece Threat','Shot Stopping', 'Short Distribution', 'Long Distribution', 'Stepping Out', 'Saving Big Chances', '1v1 Saving']
            #all_metrics = [col for col in comp_data.columns[11:] if col not in all_ratings and col[:3] != 'pct']
            all_metrics = ['Key Passes','Big Chances Created','Pass OBV','Passes into Final Third','Passes into Box','Progressive Passes','Through Passes','% of Passes Progressive','% of Passes Forward','xA','Assists','Long Passes Completed','Long Pass %','Crosses Completed into Box','Cross into Box %','Cross Shot Assists','Cross Assists','Dribble OBV','Carries','Take Ons','Dribble %','Progressive Carries','Progressive Carries in Final Third','Carries into Final Third','Carries in Final Third','Final Third Take Ons','Forward Pass %','Backward Pass %','% of Passes Backward','% of Passes Left','% of Passes Right','Short Pass %','Ball Retention %','Pressured Pass %','Fouls Won','Passes Completed','Passes Received','Passes Received Under Pressure','Final Third Receptions','Zone 14 Receptions','Box Receptions','Six Yard Box Receptions','Goals','Shots','xG','xG/Shot','Goal Conversion','xGOT','xGOT per xG','Goals per xG','Big Chances','Big Chance Conversion','Blocks','Tackles Won','Tackle %','Defensive OBV','Ball Recoveries','Pressures','Counterpressures','Defensive Third Blocks','Defensive Third Tackles Won','Defensive Third Tackle to Dribbled Past Ratio','Defensive Third Defensive OBV','Defensive Third Ball Recoveries','Defensive Third Pressures','Defensive Third Counterpressures','Attacking Third Pressures','Attacking Third Counterpressures','Interceptions','Attacking Third Tackles','Attacking Third Interceptions','Average Defensive Action Distance','Attacking Half Pressures','Attacking Half Counterpressures','Attacking Half Tackles','Attacking Half Interceptions','Attacking Half Pressure Regains','Pressure Regains Leading to Shots','Aerial Wins','Aerial %','Attacking SP Aerial Wins','Defensive SP Aerial Wins','Attacking SP Aerial %','Defending SP Aerial %','Defensive Third Interceptions','GK Shots Faced','GK Shots on Target Faced','GK Goals Conceded','GK Saves','GK xG Against','GK xGOT Against','GK 1v1s Save Rate','GK Avg. Distance','GK Save %','Goals Prevented','Big Chances Faced','Big Chances Saved','Big Chances Save %','PSV-99','PSV-85','Distance','HI Running Distance','Sprinting Distance','Count High Acceleration','Count High Deceleration']
            
            


            important_metrics_unique = []
            seen = set()
            for sublist in important_metrics:
                for item in sublist:
                    if item not in seen:
                        seen.add(item)
                        important_metrics_unique.append(item)

            #selected_metrics = list(set(item for sublist in selected_metrics for item in sublist))
            selected_metrics_unique = []
            seen = set()
            for sublist in selected_metrics:
                for item in sublist:
                    if item not in seen:
                        seen.add(item)
                        selected_metrics_unique.append(item)

            important_metrics = important_metrics_unique
            selected_metrics = selected_metrics_unique
            selected_metrics = selected_metrics[:12]

            strengths_list = []
            weaknesses_list = []

            pdata = comp_data.loc[comp_data['player_id'] == player_id, important_metrics]

            strengths = {col: pdata[col].values[0] for col in important_metrics if pdata[col].values[0] > 80}
            weaknesses = {col: pdata[col].values[0] for col in important_metrics if pdata[col].values[0] < 35}

            sorted_strengths = sorted(strengths.items(), key=lambda item: item[1], reverse=True)
            sorted_weaknesses = sorted(weaknesses.items(), key=lambda item: item[1])

            max_total = 9
            half_limit = max_total // 2
            strengths_count = min(len(sorted_strengths), half_limit)
            weaknesses_count = min(len(sorted_weaknesses), half_limit)
            remaining_slots = max_total - (strengths_count + weaknesses_count)
            if remaining_slots > 0:
                if strengths_count < weaknesses_count:
                    strengths_count += min(remaining_slots, len(sorted_strengths) - strengths_count)
                else:
                    weaknesses_count += min(remaining_slots, len(sorted_weaknesses) - weaknesses_count)

            
            strengths_list = [col for col, _ in sorted_strengths[:strengths_count]]
            weaknesses_list = [col for col, _ in sorted_weaknesses[:weaknesses_count]]


            if selected_card == 'Radar':
                compare = st.radio('Compare with another player?', ["No", "Yes"])

                

                all_options = all_ratings + all_metrics

                
                if compare == 'Yes':
                    print('')
                    col1, col2 = st.columns([2,0.6])
                    with col1: league2 = st.selectbox('Competition',league_list, index = league_index)
                    data_copy = data_copy[(data_copy['Competition'] == league2) & (data_copy['Position Group'].isin(positions))].sort_values(by = 'Player')
                    col1, col2 = st.columns([2,0.6])
                    with col1: comp_player = st.selectbox('Player', data_copy[data_copy['Player'] != player]['Player'].unique())
                    season_options2 = sorted(data_copy[data_copy['Player'] == comp_player]['Season'].unique())
                    with col2: seasons2 = st.pills('Season', season_options2, selection_mode='multi', default = season_options2[-1], key = 'season2')


                    data_copy = data_copy[data_copy['Player'] == comp_player]

                    player_id2 = data_copy.iloc[0]['offline_player_id']
                    mapped_seasons2 = list(map(lambda x: season_mapping.get(x, x), seasons2))
                    player_data2 = pd.DataFrame()
                    comp_data2 = pd.DataFrame()

                    if len(seasons2) > 0:
                        max_mapped_season2 = max(mapped_seasons2)
                        for s2 in mapped_seasons2:
                            
                            #print(max_poss_matches)
                            s_filename2 = f"{league2}{s2}-AppPlayerSeasonPercentiles.parquet"
                            print(s_filename2)
                            sdata2 = pd.read_parquet(s_filename2)
                            sdata2['Season'] = s2
                            scomp_data2 = sdata2.copy(deep=True)
                            sdata2 = sdata2[sdata2['Player'] == comp_player]
                            #sdata2 = sdata2[sdata2['player_id'] == player_id2]
                            player_data2 = pd.concat([player_data2, sdata2])
                            comp_data2 = pd.concat([comp_data2, scomp_data2])

                    #print(sorted(comp_data2['Player'].unique()))
                    #print(comp_data2[comp_data2['Team'] == 'Orlando Pride'][['Player', 'player_id']])
                    

                    
                    
                    
                    
                    comp_data2['Position Group'] = comp_data2['pos_group'].apply(lambda x: pos_map.get(x, x))

                    comp_data2 = comp_data2[comp_data2['Position Group'].isin(positions)]
                    comp_data2['Shortened Position'] = comp_data2['Detailed Position'].apply(lambda x: position_mapping.get(x, x))  
                    median_minutes2 = np.median(comp_data2['Minutes'])
                    comp_data2 = comp_data2[(comp_data2['Minutes'] > median_minutes2) & (comp_data2['player_id'] != player_id2)]
                    comp_data2 = comp_data2[(comp_data2['player_id'] != player_id2)]

                    comp_data2 = pd.concat([comp_data2, player_data2])

                    comp_data2 = comp_data2.sort_values(by = ['Season', 'Minutes'], ascending = [False, False])

                    

                    comp_data2.drop(['Tackle %'], axis = 1)
                    comp_data2['Tackle %'] = round(comp_data2['Tackle to Dribbled Past Ratio'] / (comp_data2['Tackle to Dribbled Past Ratio'] + 1) * 100, 1)
                    comp_data2.drop(['Tackle to Dribbled Past Ratio'], axis = 1)
                    


                    #comp_data['Distance'] = comp_data['Distance'].fillna(0).astype(int)
                    comp_data2['PSV-99'] = comp_data2['PSV-99'].fillna(comp_data2.groupby('player_id')['PSV-99'].transform('mean'))
                    comp_data2['PSV-85'] = comp_data2['PSV-85'].fillna(comp_data2.groupby('player_id')['PSV-85'].transform('mean'))
                    comp_data2['Distance'] = comp_data2['Distance'].fillna(comp_data2.groupby('player_id')['Distance'].transform('mean'))
                    comp_data2['HI Running Distance'] = comp_data2['HI Running Distance'].fillna(comp_data2.groupby('player_id')['HI Running Distance'].transform('mean'))
                    comp_data2['Sprinting Distance'] = comp_data2['Sprinting Distance'].fillna(comp_data2.groupby('player_id')['Sprinting Distance'].transform('mean'))
                    comp_data2['Count High Acceleration'] = comp_data2['Count High Acceleration'].fillna(comp_data2.groupby('player_id')['Count High Acceleration'].transform('mean'))
                    comp_data2['Count High Deceleration'] = comp_data2['Count High Deceleration'].fillna(comp_data2.groupby('player_id')['Count High Deceleration'].transform('mean'))

                    for col in raw_cols:
                        if col not in special_cols:
                            comp_data2[col] = comp_data2[col] * comp_data2['Minutes']


                    #print(player_id2)
                    #print(comp_data2[comp_data2[['player_id'] == player_id2]][['Player', 'Minutes']])
                    comp_data2 = comp_data2.groupby('Player').agg(aggs).reset_index()

                    for col in raw_cols:
                        if col not in special_cols:
                            comp_data2[col] = comp_data2[col] / comp_data2['Minutes']
                            comp_data2[f'pct{col}'] = round(comp_data2[col].rank(pct=True) * 100,2)
                    #print(comp_data2[comp_data2[['player_id'] == player_id2]][['Player', 'Minutes']])
                    mins2 = comp_data2.loc[comp_data2['player_id'] == player_id2,'Minutes'].values[0]
                    #print(mins2)

                    
                    
                    #comp_data2['GK_Difficulty_Faced_Raw'] = (0.3 * comp_data2['pctGK Shots on Target Faced']) + (0.2 * comp_data2['pctGK xG Against']) + (0.2 * comp_data2['pctGK xGOT Against']) + (0.3 * comp_data2['pctBig Chances Faced'])
                    comp_data2['Shot Stopping'] = (0.05 * comp_data2['pctGK Shots on Target Faced']) + (0.2 * comp_data2['pctBig Chances Save %']) + (0.6 * comp_data2['pctGoals Prevented']) + (0.15 * comp_data2['pctGK Save %'])
                    comp_data2['Short Distribution'] = (0.15 * comp_data2['pctForward Pass %']) + (0.2 * comp_data2['pctPressured Pass %']) + (0.65 * comp_data2['pctShort Pass %'])
                    comp_data2['Long Distribution'] = (0.2 * comp_data2['pctProgressive Passes']) + (0.1 * comp_data2['pctPasses into Final Third']) +  (0.35 * comp_data2['pctLong Passes Completed']) + (0.1 * comp_data2['pctPass OBV']) + (0.25 * comp_data2['pctLong Pass %'])
                    comp_data2['Stepping Out'] = (0.1 * comp_data2['pctGK Avg. Distance'])
                    comp_data2['Saving Big Chances'] = (0.25 * comp_data2['pctBig Chances Faced']) + (0.75 * comp_data2['pctBig Chances Save %'] )
                    comp_data2['1v1 Saving'] = (1 * comp_data2['pctGK 1v1s Save Rate'])
                    
                    comp_data2['Chance Creation'] = (0.3 * comp_data2['pctxA']) + (0.15 * comp_data2['pctKey Passes']) + (0.25 * comp_data2['pctBig Chances Created']) + (0.1 * comp_data2['pctPass OBV']) + (0.2 * comp_data2['pctAssists'])
                    comp_data2['Ball Progression'] = (0.2 * comp_data2['pctPass OBV']) + (0.15 * comp_data2['pctPasses into Final Third']) + (0.25 * comp_data2['pctProgressive Carries']) + (0.3 * comp_data2['pctProgressive Passes']) + (0.1 * comp_data2['pctLong Passes Completed'])
                    comp_data2['Ball Retention'] = (0.15 * comp_data2['pctForward Pass %']) + (0.55 * comp_data2['pctBall Retention %']) + (0.2 * comp_data2['pctPressured Pass %']) + (0.1 * comp_data2['pctShort Pass %'])
                    comp_data2['Verticality'] = (0.25 * comp_data2['pct% of Passes Progressive']) + (0.6 * comp_data2['pct% of Passes Forward']) + (0.15 * (1- comp_data2['pct% of Passes Backward']))
                    comp_data2['Carrying'] = (0.25 * comp_data2['pctTake Ons']) + (0.1 * comp_data2['pctCarries']) + (0.5 * comp_data2['pctProgressive Carries']) + (0.15 * comp_data2['pctDribble %'])
                    comp_data2['Poaching'] = (0.55 * comp_data2['pctxG']) + (0.2 * comp_data2['pctxG/Shot']) + (0.2 * comp_data2['pctBox Receptions']) + (0.05 * comp_data2['pctSix Yard Box Receptions'])
                    comp_data2['Finishing'] = (0.25 * comp_data2['pctGoals per xG']) + (0.15 * comp_data2['pctxGOT per xG']) + (0.45 * comp_data2['pctGoal Conversion']) + (0.15 * comp_data2['pctGoals'])
                    comp_data2['Goal Threat'] = (0.3 * comp_data2['pctxG']) + (0.3 * comp_data2['pctGoals']) + (0.2 * comp_data2['pctBox Receptions']) + (0.1 * comp_data2['pctGoal Conversion']) + (0.1 * comp_data2['pctxGOT per xG'])
                    comp_data2['Crossing'] = (0.3 * comp_data2['pctCrosses Completed into Box']) + (0.2 * comp_data2['pctCross into Box %']) + (0.25 * comp_data2['pctCross Shot Assists']) + (0.25 * comp_data2['pctCross Assists'])
                    comp_data2['Heading'] = (0.7 * comp_data2['pctAerial %']) + (0.3 * comp_data2['pctAerial Wins'])
                    comp_data2['Set Piece Threat'] = (0.75 * comp_data2['pctAttacking SP Aerial Wins']) + (0.25 * comp_data2['pctAttacking SP Aerial %'])

                    comp_data2['Speed'] = (0.4 * comp_data2['pctPSV-99']) + (0.45 * comp_data2['pctPSV-85']) + (0.15 * comp_data2['pctSprinting Distance'])
                    comp_data2['HSR Distance'] = (0.4 * comp_data2['pctDistance']) + (0.6 * comp_data2['pctHI Running Distance'])

                    comp_data2['High Pressing'] = (0.25 * comp_data2['pctAttacking Half Pressures']) + (0.15 * comp_data2['pctAttacking Third Pressures']) + (0.2 * comp_data2['pctAttacking Half Pressure Regains']) + (0.1 * comp_data2['pctPressure Regains Leading to Shots']) + (0.3 * comp_data2['pctAverage Defensive Action Distance'])
                    comp_data2['Defending High'] = (0.2 * comp_data2['pctAttacking Half Pressures']) + (0.05 * comp_data2['pctAttacking Half Pressure Regains']) + (0.75 * comp_data2['pctAverage Defensive Action Distance'])
                    comp_data2['Tackle Accuracy'] = (0.2 * (100 - comp_data2['pctDribbled Past'])) + (0.65 * comp_data2['pctTackle %']) + (0.15 * comp_data2['pctTackles Won'])
                    comp_data2['Defensive Output'] = (0.1 * comp_data2['pctBlocks']) + (0.3 * comp_data2['pctTackles Won']) + (0.4 * comp_data2['pctBall Recoveries']) + (0.2 * comp_data2['pctInterceptions'])
                    comp_data2['Receiving Forward'] = (0.6 * comp_data2['pctFinal Third Receptions']) + (0.15 * comp_data2['pctBox Receptions']) + (0.15 * comp_data2['pctShots']) + (0.1 * comp_data2['pctxG'])

                    rating_cols = ['Chance Creation', 'Crossing','Ball Progression', 'Ball Retention', 'Verticality', 'Carrying', 'Poaching', 'Finishing', 'Goal Threat', 'Heading', 'Set Piece Threat', 'Speed', 'HSR Distance', 'High Pressing', 'Defending High', 'Tackle Accuracy', 'Defensive Output', 'Receiving Forward']

                    for x in rating_cols:
                        comp_data2[x] = round(comp_data2[x].rank(pct=True) * 100,2)



                selected_radars = important_metrics[:8]
                selected_radars = st.multiselect("Customize Radar",
                                                 all_options, default = important_metrics[:8])
                



            # #print(positions)
            # #print(important_metrics)
            # for col in important_metrics:
            #     #(comp_data.loc[comp_data['player_id'] == player_id,col].values[0])
            #     #print(col, int(comp_data.loc[comp_data['player_id'] == player_id,col].values[0]))
            #     if comp_data.loc[comp_data['player_id'] == player_id,col].values[0] > 80:
            #         strengths_list.append(col)
            #     elif comp_data.loc[comp_data['player_id'] == player_id,col].values[0] < 35:
            #         weaknesses_list.append(col)
                
            # if len(strengths_list) + len(weaknesses_list) > 8:
            #     while len(strengths_list) + len(weaknesses_list) > 8:
            #         if len(strengths_list) > len(weaknesses_list)

                

            
            
            

            
        

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"[{stage}] Memory Usage: {mem_usage:.2f} MB")

log_memory_usage("Start") 
            
# output_dir = './saved_images'  # Save it inside your Streamlit app folder
# os.makedirs(output_dir, exist_ok=True)



if start_graphic:
    #Start Graphic
    dpi = 100  # Dots per inch
    #width, height = 800 / dpi, 1800 / dpi
    width, height = 1275 / dpi, 1650 / dpi

    # Create a figure
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("black")  # Set the figure background to black

    ax.set_facecolor("black")
    ax.axis("off")

    comp_data['Team'] = comp_data['Team'].replace({
                'NJ/NY Gotham FC': 'NJ NY Gotham FC',
                'SGS Essen 19/68': 'SGS Essen 19 68'
            })
    team_name = comp_data.loc[comp_data['player_id'] == player_id, 'Team'].values[0]
   


    #club_image_path = f"/Users/malekshafei/Desktop/Louisville/Club Logos/{team_name}.webp"
    club_image_path = f"Club Logos/{team_name}.webp"
    with Image.open(club_image_path) as img:
        width, height = img.size
        #print(width,height)
        if height > width * 2:
            width_factor = (1200/width) * 0.65
        else:
            width_factor = 1200/width
        
    
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
 
    # def add_image(ax, image_path, xy, zoom=0.2):
    
    #     img = mpimg.imread(image_path)  # Load image
    #     imagebox = OffsetImage(img, zoom=zoom)  # Set resize factor
    #     ab = AnnotationBbox(imagebox, xy, xycoords="data", frameon=False)  # Keep fixed in data coordinates
    #     ax.add_artist(ab)  # Add image without modifying axis settings

    def add_real_image(ax, image_path, xy, zoom=0.2):
        # Open image with PIL (supports .webp and all formats)
        img = Image.open(image_path)
        
        # Convert PIL image to NumPy array (Matplotlib expects this format)
        img_array = np.array(img) 

        # Create OffsetImage with the image array
        imagebox = OffsetImage(img_array, zoom=zoom)
        
        # Attach the image to the given coordinates
        ab = AnnotationBbox(imagebox, xy, xycoords="data", frameon=False)
        ax.add_artist(ab)  # Add image to plot

    def add_image(ax, img, xy, zoom=0.2):
        # Open image with PIL (supports .webp and all formats)
        #img = Image.open(image_path)
        
        # Convert PIL image to NumPy array (Matplotlib expects this format)
        img_array = np.array(img) 

        # Create OffsetImage with the image array
        imagebox = OffsetImage(img_array, zoom=zoom)
        
        # Attach the image to the given coordinates
        ab = AnnotationBbox(imagebox, xy, xycoords="data", frameon=False)
        ax.add_artist(ab)  # Add image to plot
    
    if league in ['England', 'France']: add_real_image(ax, club_image_path, xy=(0.837, .972), zoom=0.05 * width_factor)
    else: add_real_image(ax, club_image_path, xy=(0.825, .972), zoom=0.05 * width_factor) 

    #league_image_path = f"/Users/malekshafei/Desktop/Louisville/League Logos/{league}.webp"
    league_image_path = f"League Logos/{league}.webp"
    with Image.open(league_image_path) as img:
        width, height = img.size
        #print("wayy",width,height)
        if height > width * 1.4:
            width_factor = (400/width) * 0.65
        else:
            width_factor = 320/width
    add_real_image(ax, league_image_path, xy=(0.95, .972), zoom=0.25 * width_factor)

    name_size = 50
    if len(player_nickname) > 15: name_size = 45
    
    ax.text(
        0.053, 0.997, player_nickname,  # Coordinates: (0, 1) for top-left in normalized figure coordinates
        color="white",  # Text color
        fontsize=name_size,    # Text size
        ha="left", va="top",  # Align text to the top-left corner
        transform=ax.transAxes,  # Use figure-relative coordinates
        fontproperties=bold
        )

    #player_bio_text = f"{team_name} - {player_age} - {player_height}"

    ax.text(
        0.05, 0.937, 'Data Report',  # Coordinates: (0, 1) for top-left in normalized figure coordinates
        color="white",  # Text color
        fontsize=30,    # Text size
        ha="left", va="top",  # Align text to the top-left corner
        transform=ax.transAxes,  # Use figure-relative coordinates
        fontproperties=regular
        )

    import matplotlib.patches as patches



    #Player Bio

    rect1 = patches.Rectangle( 
        (0.05, 0.64), #bottom left coords
        0.34, 0.25,  # Width, Height
        linewidth=0.25,  # Outline thickness
        edgecolor="purple",  # Outline color
        facecolor="purple",  # Fill color
        alpha=0.25  # Optional: Transparency
        )
    ax.add_patch(rect1)

    #detailed_position = comp_data.loc[comp_data['player_id'] == player_id, 'Shortened Position'].values[0]

    detailed_position = ','.join(comp_data.loc[comp_data['player_id'] == player_id, 'Shortened Position'].values[0].split(',')[:2])
    #print(detailed_position)
    kit_number = comp_data.loc[comp_data['player_id'] == player_id, 'Number'].values[0]
    foot = comp_data.loc[comp_data['player_id'] == player_id,'Foot'].values[0]

    minutes = comp_data.loc[comp_data['player_id'] == player_id,'Minutes'].values[0]

    matches = len(events['match_id'].unique())
    pct_mins = min(int((minutes / (max_poss_matches * 98)) * 100), 100)

    goals = int(comp_data.loc[comp_data['player_id'] == player_id,'Goals'].values[0] * (minutes/90))
    assists = int(comp_data.loc[comp_data['player_id'] == player_id,'Assists'].values[0] * (minutes/90))
    gap90 = round(comp_data.loc[comp_data['player_id'] == player_id,'Goals'].values[0] + comp_data.loc[comp_data['player_id'] == player_id,'Assists'].values[0],2)



    def add_basic_text(x,y, text, fontsize = 13, ha = 'left', color = 'white', fontproperties = regular, ax = ax):
        ax.text(
            x, y, text,
            color=color,  # Text color
            fontsize=fontsize,    # Text size
            ha=ha, va="top",  
            transform=ax.transAxes,  # Use figure-relative coordinates
            fontproperties=fontproperties
        )

    # comp_data['Team'] = comp_data['Team'].apply(lambda x: replace_team_names(x, team_replacements))
    # #print(comp_data.loc[comp_data['player_id'] == player_id, 'Team'])
    # team_name = comp_data.loc[comp_data['player_id'] == player_id, 'Team'].values[0]
    # add_basic_text(0.075, 0.87, 'Player Info', fontsize= 20, fontproperties=bold)


    # add_basic_text(0.075, 0.84, 'Club', fontsize= 14.5, color = 'gray')
    # add_basic_text(0.075, 0.822, team_name, fontsize= 14.5)

    add_basic_text(0.23, 0.84, 'Age', fontsize= 14.5, color = 'gray')
    add_basic_text(0.23, 0.822, player_age, fontsize= 14.5)

    add_basic_text(0.305, 0.84, 'Height', fontsize= 13, color = 'gray')
    add_basic_text(0.305, 0.822, player_height, fontsize= 14.5)


    add_basic_text(0.075, 0.79, 'Position(s)', fontsize= 14.5, color = 'gray')
    add_basic_text(0.075, 0.772, detailed_position, fontsize= 14.5)

    add_basic_text(0.23, 0.79, '#', fontsize= 14.5, color = 'gray')
    add_basic_text(0.23, 0.772, kit_number, fontsize= 14.5)

    add_basic_text(0.305, 0.79, 'Foot', fontsize= 14.5, color = 'gray')
    add_basic_text(0.305, 0.772, foot, fontsize= 14.5)


    add_basic_text(0.075, 0.74, '% Mins Played', fontsize= 14.5, color = 'gray')
    add_basic_text(0.075, 0.722, f'{pct_mins}%', fontsize= 14.5)

    add_basic_text(0.23, 0.74, 'Mins', fontsize= 14.5, color = 'gray')
    add_basic_text(0.23, 0.722, minutes, fontsize= 14.5)

    add_basic_text(0.305, 0.74, 'Games', fontsize= 14, color = 'gray')
    add_basic_text(0.305, 0.722, matches, fontsize= 14.5)


    add_basic_text(0.075, 0.69, 'G+A p90', fontsize= 14.5, color = 'gray')
    add_basic_text(0.075, 0.672, gap90, fontsize= 14.5)

    add_basic_text(0.23, 0.69, 'Goals', fontsize= 14.5, color = 'gray')
    add_basic_text(0.23, 0.672, goals, fontsize= 14.5)

    add_basic_text(0.305, 0.69, 'Assists', fontsize= 14.5, color = 'gray')
    add_basic_text(0.305, 0.672, assists, fontsize= 14.5)

    
    strwek_list_len = len(strengths_list) + len(weaknesses_list)
    if len(strengths_list) == 0:
        strwek_list_len += 1
    elif len(weaknesses_list) == 0:
        strwek_list_len += 1
    elif len(strengths_list) == 0 and len(weaknesses_list) == 0:
        strwek_list_len += 2
    
    


    add_basic_text(0.07, 0.60, 'Strengths', fontsize= 18, fontproperties=bold)
    

    start_y = 0.575
    if len(strengths_list) == 0:
        add_basic_text(0.08, start_y, f'None', fontsize= 16.5, color= 'red')
        start_y -= 0.025
    else:
        for x in strengths_list:
            add_basic_text(0.08, start_y, f'+ {x}', fontsize= 16.5, color= 'green')
            start_y -= 0.025


    start_y -= .01
    add_basic_text(0.07, start_y, 'Weaknesses', fontsize= 18,fontproperties=bold)
    if len(weaknesses_list) == 0:
        add_basic_text(0.08, start_y - 0.025, f'None', fontsize= 16.5, color= 'green')
        start_y -= 0.025
    else:
        for x in weaknesses_list:
            start_y -= 0.025
            add_basic_text(0.08, start_y, f'-{x}', fontsize= 16.5, color='red')

    start_y -= 0.025


    rect_2_height = 0.61 - start_y

    rect2 = patches.Rectangle( 
        (0.05, start_y),#0.58), #bottom left coords
        0.34, rect_2_height,#0.18,  # Width, Height
        linewidth=0.25,  # Outline thickness
        edgecolor="purple",  # Outline color
        facecolor="purple",  # Fill color
        alpha=0.25  # Optional: Transparency
        )
    ax.add_patch(rect2)



    

    def calculate_similarity(player1, player2):
        columns = important_metrics
        if not columns:
            return 0
        values1 = player1[columns].values
        values2 = player2[columns].values
        values1_norm = normalize(pd.Series(values1))
        values2_norm = normalize(pd.Series(values2))
        
        return cosine_sim(values1_norm, values2_norm)[0][0]

    
    # print(comp_data['Team'].unique())
    def get_most_similar_players(n=5):
        player_rows = comp_data[comp_data['player_id'] == player_id]
        # if player_rows.empty:
        #     st.error(f"Player {player_name} not found in the dataset.")
        #     return pd.DataFrame()
        
        player = player_rows.iloc[0]
        
        # Check for NAs in the player's data
        #na_columns = player[player['columns_to_compare']].isna().sum()
        # if na_columns > 0:
        #     st.warning(f"Player {player_name} has {na_columns} NA values in their data.")
        
        similarities = comp_data.apply(lambda x: calculate_similarity(player, x), axis=1)
        similar_indices = similarities.sort_values(ascending=False).index[1:n+1]  # Exclude the player itself
        similar_players = comp_data.loc[similar_indices]
        return pd.DataFrame({
            'Player': similar_players['Player'],
            'Similarity': similarities[similar_indices],
            'Team': similar_players['Team'],
            #'Age': similar_players['Age'],
            'Detailed Position': similar_players['Detailed Position'],
            'Minutes': similar_players['Minutes']
        })
    

    import matplotlib.patches as patches
    def add_progress_circle(ax, percentage, x, y, size):
        if percentage > 75:
            color = "green"
        elif percentage > 50:
            color = "yellow"
        elif percentage > 25:
            color = "orange"
        else:
            color = "red"

        # Create inset axes for the progress circle (relative size)
        inset_ax = ax.inset_axes([x, y, size, size], transform=ax.transAxes)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_frame_on(False)
        inset_ax.set_aspect('equal')
        inset_ax.figure.set_dpi(300)

        # Draw background circle (gray)
        #inset_ax.add_patch(patches.Circle((0, 0), 1, fill = False, ec="black", lw=2))
        inset_ax.add_patch(patches.Circle((0, 0), 1, fill=False, edgecolor="lightgray", lw=2))


        theta1 = 90  # Start at 12 o'clock
        theta2 = 90 - (360 * (percentage / 100))
        
        #arc = patches.Wedge((0, 0), 2, 2, angle = 0, theta2 = 90, color=color, lw=4)
        arc = patches.Arc((0, 0), 2, 2, angle=0, theta1=theta2, theta2=theta1, color=color, lw=5)


        inset_ax.add_patch(arc)

        # Add percentage text in the middle
        inset_ax.text(0, 0, f"{percentage:.0f}%", fontsize=150*size,
                    ha="center", va="center", color="white", fontproperties=regular)

        # Formatting to keep it circular
        inset_ax.set_xlim(-1.1, 1.1)
        inset_ax.set_ylim(-1.1, 1.1)
        inset_ax.margins(0.5)

        if size < 0.1:
            inset_ax.set_xlim(-1.3, 1.3)
            inset_ax.set_ylim(-1.3, 1.3)
        

        

    #add_progress_circle(ax, 83, x=0.75, y=0.75, size=0.1)

    def split_name(name):
        name_parts = name.strip().split()  # Split the name by spaces
        
        if len(name_parts) > 3:
            first_part = name_parts[0]
            last_word = name_parts[-1]
        elif len(name_parts) > 1:
            first_part = " ".join(name_parts[:-1])  # All parts except the last
            last_word = name_parts[-1]  # The last word


        else:
            first_part = ""
            last_word = name_parts[0]
        
        return first_part, last_word

        

    start_y -= 0.04
    add_basic_text(0.075, start_y, 'Similar Players', fontsize= 20,fontproperties=bold)

    start_y -= 0.05
    similar_players = get_most_similar_players()
    
    for i, (_, row) in enumerate(similar_players.iterrows(), 1):
        sim_pct = round(row['Similarity'] * 100, 2)

        add_basic_text(0.07, start_y, f"{i}.", fontsize= 15)

        

        ##
        tt = row['Team']

        #club_path = f"/Users/malekshafei/Desktop/Louisville/Club Logos/{tt}.webp"
        club_path = f"Club Logos/{tt}.webp"
        with Image.open(club_path) as img:
            width, height = img.size
            #print(width,height)
            width_factor = 1200/width
            
        
        #league_image_path = "/Users/malekshafei/Downloads/USL Logo.webp" 

        ##
        add_real_image(ax, club_path, xy=(0.115, start_y-0.005), zoom=0.019 * width_factor)

        name1, name2 = split_name(row['Player'])
        if name1 == '':
            add_basic_text(0.145, start_y, name2, fontsize= 15, fontproperties=bold)
        else:
            add_basic_text(0.145, start_y+0.01, name1, fontsize= 15)
            add_basic_text(0.145, start_y-0.01, name2, fontsize= 15, fontproperties=bold)



        #add_basic_text(0.145, start_y, row['Player'], fontsize= 15)
        add_progress_circle(ax, sim_pct, x=0.33, y=start_y - 0.03, size=0.05)

        start_y -= 0.05
        
        
        
    
    rect1 = patches.Rectangle( 
        (0.05, start_y), #bottom left coords
        0.34, 0.317,  # Width, Height
        linewidth=0.25,  # Outline thickness
        edgecolor="purple",  # Outline color
        facecolor="purple",  # Fill color
        alpha=0.25  # Optional: Transparency
        )
    ax.add_patch(rect1)

    log_memory_usage("After Left") 
    
    #selected_metrics
    def plot_progress_bar(ax, percentage, x, y, width, height = 0.05):
        
        # Ensure percentage is within 0 to 100 range
        percentage = max(0, min(100, percentage))
        
        # Set the color of the progress bar based on percentage
        if percentage > 75:
            color = 'green'
        elif percentage > 50:
            color = 'yellow'
        elif percentage > 25:
            color = 'orange'
        else:
            color = 'red'
        
        # Draw background bar (gray)
        ax.add_patch(patches.Rectangle((x, y), width, height, color="lightgray"))
        
        # Draw progress bar (colored)
        ax.add_patch(patches.Rectangle((x, y), width * (percentage / 100), height, color=color))
        
        # Add percentage text in the middle of the progress bar
        #ax.text(x + width / 2, y + height / 2, f"{percentage}%", ha='center', va='center', fontsize=10, color="black")

        # Set aspect and axis limits
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')  # Hide axis for better visual

    start_y = 0.87
    add_basic_text(0.48, start_y, 'Key Metrics', fontsize= 20,fontproperties=bold)
    add_basic_text(0.665, start_y - 0.0049, '(Data per 90 minutes)', fontsize= 12)
    start_y1 = start_y
    start_y2 = start_y
    
    start_y1 -= 0.05
    for metric in selected_metrics[::2]:
        
        if metric in metric_replacements:
            add_basic_text(0.48, start_y1, metric_replacements[metric], fontsize= 12,fontproperties=bold)
        else:
            add_basic_text(0.48, start_y1, metric, fontsize= 12,fontproperties=bold)


        raw_metric = comp_data.loc[comp_data['player_id'] == player_id,metric].values[0]
        pct_metric = comp_data.loc[comp_data['player_id'] == player_id,f'pct{metric}'].values[0]
        

        plot_progress_bar(ax, percentage=pct_metric, x=0.481, y=start_y1-0.022, width=0.21, height=0.005)
        add_basic_text(0.69, start_y1, f"{round(raw_metric,2)}", fontsize= 12, ha='right')
        start_y1 -= 0.04

    start_y2 -= 0.05
    for metric in selected_metrics[1::2]:
        #add_basic_text(0.74, start_y2, metric, fontsize= 12,fontproperties=bold)

        if metric in metric_replacements:
            add_basic_text(0.74, start_y2, metric_replacements[metric], fontsize= 12,fontproperties=bold)
        else:
            add_basic_text(0.74, start_y2, metric, fontsize= 12,fontproperties=bold)

        raw_metric = comp_data.loc[comp_data['player_id'] == player_id,metric].values[0]
        pct_metric = comp_data.loc[comp_data['player_id'] == player_id,f'pct{metric}'].values[0]
        

        plot_progress_bar(ax, percentage=pct_metric, x=0.741, y=start_y2-0.022, width=0.21, height=0.005)
        add_basic_text(0.95, start_y2, f"{round(raw_metric,2)}", fontsize= 12, ha='right')
        start_y2 -= 0.04

    rect1 = patches.Rectangle( 
        (0.442, 0.57), #bottom left coords
        0.542, 0.32,  # Width, Height
        linewidth=0.25,  # Outline thickness
        edgecolor="purple",  # Outline color
        facecolor="purple",  # Fill color
        alpha=0.25  # Optional: Transparency
        )
    ax.add_patch(rect1)

    log_memory_usage("After Right 1") 

   
    if selected_card == 'Shot Map':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, 'Shot Map', fontsize= 20,fontproperties=bold)
        add_basic_text(0.64, 0.51 - 0.0049, '(Penalties Excluded)', fontsize= 12)
        
        shots = events[(events['type'] == 'Shot') & (events['shot_type'] != 'Penalty')]
        goals_scored = len(events[events['shot_outcome'] == 'Goal'])
        xg_total = round(np.nansum(events['shot_statsbomb_xg']),2)
        shots_taken = len(shots)
        xg_per_shot = round(safe_div(xg_total, shots_taken),2)
        goal_conversion = int(safe_div(goals_scored, shots_taken) * 100)
        pens_taken = len(events[(events['shot_type'] == 'Penalty')])
        pens_scored = len(events[(events['shot_outcome'] == 'Goal') & (events['shot_type'] == 'Penalty')])

        transition_xg = round(np.nansum(events[(events['pressure_in_prev_15s'] == True) | (events['counter_shot'] == True)]['shot_statsbomb_xg']),2)
        sp_xg = round(np.nansum(events[(events['shot_from_corner'] == True) | (events['shot_from_fk'] == True)]['shot_statsbomb_xg']),2)
        regular_xg = round(xg_total - sp_xg - transition_xg,1)


        add_basic_text(0.485, 0.475, f'Goals: {goals_scored}', fontsize= 16)
        add_basic_text(0.485, 0.44, f'xG: {xg_total}', fontsize= 16)

        add_basic_text(0.7, 0.475, f'Shots: {shots_taken}', fontsize= 16, ha = 'center')
        add_basic_text(0.7, 0.44, f'xG / Shot: {xg_per_shot}', fontsize= 16, ha = 'center')

        add_basic_text(0.96, 0.475, f'Conversion: {goal_conversion}%', fontsize= 16, ha = 'right')
        add_basic_text(0.96, 0.44, f'Penalties: {pens_scored}/{pens_taken}', fontsize= 16, ha = 'right')


        add_basic_text(0.50, 0.405, f'Transition xG: {transition_xg}', fontsize= 16, ha = 'left')
        add_basic_text(0.915, 0.405, f'Set Piece xG: {sp_xg}', fontsize= 16, ha = 'right')

        
 

        #add_basic_text(0.53, 0.475, f'{goals_scored}', fontsize= 14)


        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#200020', line_color='#c7d5cc',
                      half=True, pad_top=6, corner_arcs=True,)
        
        fig2,ax2 = pitch.draw(figsize=(8,12))
        #fig2.set_facecolor('purple')
        
        shots = events[events['type'] == 'Shot']
        for _, row in shots.iterrows():
            x = row['x']
            y = row['y']
            outcome = row['shot_outcome']
            xg = row['shot_statsbomb_xg']


            color = 'red' 
            if outcome == 'Goal': color = 'green'
            elif outcome == 'Saved': color = 'yellow'
            elif outcome == 'Saved to Post': color = 'yellow'
            #elif outcome == 'Blocked': color = 'red'
            # else: color == 'red'
            
            
            pitch.scatter(x, y, ax=ax2, color=color, marker='.', s=xg*2050)

        pitch.scatter(85, 15, ax=ax2, color='green', marker='.', s=1000)
        pitch.scatter(85, 27.2, ax=ax2, color='yellow', marker='.', s=1000)
        pitch.scatter(85, 40.2, ax=ax2, color='red', marker='.', s=1000)
        


        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        cropped_img = img.crop((130, 50, 660, 400)) #(left, upper, right, lower)
        #visual_file_name = "os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)



        add_image(ax, cropped_img, xy=(0.715, .17), zoom=0.7285)
        add_basic_text(0.51, 0.031, f'Goal', fontsize= 14, ha = 'left')
        add_basic_text(0.62, 0.031, f'Saved', fontsize= 14, ha = 'left')
        add_basic_text(0.735, 0.031, f'Off Target / Blocked', fontsize= 14, ha = 'left')

        log_memory_usage("After Shot") 

    if selected_card == 'Key Passes':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, 'Key Passes', fontsize= 20,fontproperties=bold)
        #add_basic_text(0.64, 0.51 - 0.0049, '(Set Piees Excluded)', fontsize= 12)
        
        kps = events[(events['type'] == 'Pass') & ((events['pass_shot_assist'] == True) | (events['pass_goal_assist'] == True))]
        assists = len(events[events['pass_goal_assist'] == True])
        xa_total = round(np.nansum(events['xA']),2)
        kps_num = len(kps)

        big_chances_created = len(events[events['xA'] > 0.1])
        sp_kps = len(events[(events['pass_type'].isin(['Free Kick', 'Corner'])) & ((events['pass_shot_assist'] == True) | (events['pass_goal_assist'] == True))])
        cross_att = len(events[events['pass_cross'] == True])
        cross_succ = len(events[(events['completed_pass'] == True) & (events['pass_cross'] == True)])
        cross_shot_assists = len(events[((events['pass_shot_assist'] == True)  | (events['pass_goal_assist'] == True)) & (events['pass_cross'] == True)])




        add_basic_text(0.485, 0.475, f'Assists: {assists}', fontsize= 16)
        add_basic_text(0.485, 0.44, f'Big Chances: {big_chances_created}', fontsize= 16)

        add_basic_text(0.68, 0.475, f'xA: {xa_total}', fontsize= 16, ha = 'center')
        
        add_basic_text(0.485, 0.405, f'Crosses: {cross_succ}/{cross_att}', fontsize= 16, ha = 'left')
        add_basic_text(0.8, 0.405, f'Cross Shot Assists: {cross_shot_assists}', fontsize= 16, ha = 'center')

        add_basic_text(0.93, 0.475, f'Key Passes: {kps_num}', fontsize= 16, ha = 'right')
        add_basic_text(0.93, 0.44, f'(From SPs: {sp_kps})', fontsize= 16, ha = 'right')


        # add_basic_text(0.50, 0.405, f'Transition xG: {transition_xg}', fontsize= 16, ha = 'left')
        # add_basic_text(0.915, 0.405, f'Set Piece xG: {sp_xg}', fontsize= 16, ha = 'right')

        
 

        #add_basic_text(0.53, 0.475, f'{goals_scored}', fontsize= 14)


        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#200020', line_color='#c7d5cc',
                      half=True, pad_top=6, corner_arcs=True,)
        
        fig2,ax2 = pitch.draw(figsize=(8,12))
        #fig2.set_facecolor('purple')
        
        
        for _, row in kps.iterrows():
            x1 = row['x']
            y1 = row['y']
            x2 = row['pass_end_x']
            y2 = row['pass_end_y']
            outcome = row['pass_goal_assist']
            xa = row['xA']


            color = 'orange' 
            if outcome == True: color = 'green'
            else: color = 'orange'
            
            #elif outcome == 'Blocked': color = 'red'
            # else: color == 'red'
            
            
            pitch.scatter(x1, y1, ax=ax2, color='white', marker='.', s=80)
            pitch.scatter(x2, y2, ax=ax2, color=color, marker='.', s=250)
            pitch.lines(linewidth = 3, xstart = x1, ystart = y1, xend=x2,yend=y2,comet = True,ax=ax2,color=color)

        # pitch.scatter(85, 15, ax=ax2, color='green', marker='.', s=1000)
        # pitch.scatter(85, 27.2, ax=ax2, color='orange', marker='.', s=1000)
        # #pitch.scatter(85, 40.2, ax=ax2, color='red', marker='.', s=1000)
        


        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        #cropped_img = img.crop((130, 50, 660, 400)) #(left, upper, right, lower)
        cropped_img = img.crop((30, 50, 760, 470)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2) 



        #add_image(ax, visual_file_name, xy=(0.715, .17), zoom=0.7285)
        add_image(ax, cropped_img, xy=(0.715, .2), zoom=0.5285)
        # add_basic_text(0.51, 0.081, f'Goal Assist', fontsize= 14, ha = 'left')
        # add_basic_text(0.62, 0.031, f'Shot Assist', fontsize= 14, ha = 'left')
        #add_basic_text(0.735, 0.031, f'Off Target / Blocked', fontsize= 14, ha = 'left')

        log_memory_usage("After KP") 


    if selected_card == 'Ball Carrying':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, '1v1 Dribbling & Carrying', fontsize= 20,fontproperties=bold)
        #add_basic_text(0.64, 0.51 - 0.0049, '(Set Piees Excluded)', fontsize= 12)

        dribbles = events[(events['type'].isin(['Carry', 'Dribble']))]
        take_on_att = len(events[(events['type'] == 'Dribble')])
        take_on_succ = len(events[(events['type'] == 'Dribble') & (events['dribble_outcome'] == 'Complete')])

        box_take_on_att = len(events[(events['type'] == 'Dribble') & (events['x'] > 102) & (events['y'] > 17) & (events['y'] < 62)])
        box_take_on_succ = len(events[(events['type'] == 'Dribble') & (events['dribble_outcome'] == 'Complete') & (events['x'] > 102) & (events['y'] > 17) & (events['y'] < 62)])

        dribble_succ = int(safe_div(take_on_succ, take_on_att) * 100)
        box_dribble_succ = int(safe_div(box_take_on_succ, box_take_on_att) * 100)
        prog_carries = len(events[events['is_progressive_carry'] == True])
        

        f3_take_ons = len(events[(events['type'] == 'Dribble') & (events['x'] > 80) & (events['dribble_outcome'] == 'Complete')])
        f3_carries = len(events[(events['is_progressive_carry'] == True) & (events['carry_end_x'] > 80)])
        carries_into_box = len(events[(events['type'] == 'Carry') & (events['is_box_entry'] == True)])


        



        add_basic_text(0.485, 0.475, f'Take Ons: {take_on_succ}/{take_on_att} ({dribble_succ}%)', fontsize= 16)
        add_basic_text(0.485, 0.44, f'Progressive Carries: {prog_carries}', fontsize= 16)

        add_basic_text(0.78, 0.475, f'(Inside Box: {box_take_on_succ}/{box_take_on_att})', fontsize= 16)
        add_basic_text(0.78, 0.44, f'Box Entries: {carries_into_box}', fontsize= 16)



        pitch = Pitch(pitch_type='statsbomb', pitch_color='#200020', line_color='#c7d5cc',
                       pad_top=6,  corner_arcs=True,)
        
        fig2,ax2 = pitch.draw(figsize=(8,12))
        #fig2.set_facecolor('purple')
        
        
        for _, row in dribbles.iterrows():

            if row['is_progressive_carry'] == True:
                x1 = row['x']
                y1 = row['y']
                x2 = row['carry_end_x']
                y2 = row['carry_end_y']

                #print(x1, y1)
                #print(x2,y2)

                #print('')
                
                color = 'orange' 

                #pitch.scatter(x1, y1, ax=ax2, color=color, marker='.', s=80)
                #pitch.scatter(x2, y2, ax=ax2, color=color, marker='.', s=150)
                pitch.lines(transparent= True , linewidth = 2, comet = True, xstart = x1, ystart = y1, xend=x2,yend=y2,ax=ax2,color=color)


            if row['type'] == 'Dribble':
                x1 = row['x']
                y1 = row['y']
                if row['dribble_outcome'] == 'Complete': color = 'green'
                else: color = 'red'

                pitch.scatter(x1, y1, ax=ax2, color=color, marker='.', s=250)


        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        #cropped_img = img.crop((130, 50, 660, 400)) #(left, upper, right, lower)
        cropped_img = img.crop((150, 30, 760, 530)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)



        #add_image(ax, visual_file_name, xy=(0.715, .17), zoom=0.7285)
        add_image(ax, cropped_img, xy=(0.715, .2), zoom=0.6285)

        log_memory_usage("After Drib") 

    if selected_card == 'Progressive Actions':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, 'Progressive', fontsize= 20,fontproperties=bold)
        add_basic_text(0.66, 0.51, 'Passes', fontsize= 20,fontproperties=bold, color='orange')
        add_basic_text(0.77, 0.51, '&', fontsize= 20,fontproperties=bold)
        add_basic_text(0.80, 0.51, 'Carries', fontsize= 20,fontproperties=bold, color = 'magenta')
        #add_basic_text(0.64, 0.51 - 0.0049, '(Set Piees Excluded)', fontsize= 12)

        prog_actions = events[(events['is_progressive'] == True) | (events['is_progressive_carry'] == True)]
        succ_prog_passes = len(events[(events['type'] == 'Pass') & (events['is_progressive'] == True) & (events['completed_pass'] == True)] )
        att_prog_passes = len(events[(events['type'] == 'Pass') & (events['is_progressive'] == True)] )
        prog_carries = len(events[(events['type'] == 'Carry') & (events['is_progressive_carry'] == True)] )
        
        prog_pass_rate = int(safe_div(succ_prog_passes, att_prog_passes) * 100)
        total_actions = len(events[events['type'].isin(['Pass', 'Carry'])])
        pct_prog = int(safe_div(len(prog_actions), total_actions) * 100)
        
        



        add_basic_text(0.485, 0.475, f'Passes: {succ_prog_passes}/{att_prog_passes} ({prog_pass_rate}%)', fontsize= 16)
        add_basic_text(0.78, 0.475, f'Carries: {prog_carries}', fontsize= 16)

        add_basic_text(0.485, 0.44, f'% of Actions Progressive: {pct_prog}%', fontsize= 16)



        pitch = Pitch(pitch_type='statsbomb', pitch_color='#200020', line_color='#c7d5cc',
                       pad_top=6,  corner_arcs=True,)
        
        fig2,ax2 = pitch.draw(figsize=(8,12))
        #fig2.set_facecolor('purple')
        
        
        for _, row in prog_actions.iterrows():

            if row['type'] == 'Carry':
                x1 = row['x']
                y1 = row['y']
                x2 = row['carry_end_x']
                y2 = row['carry_end_y']

                
                color = 'magenta' 

                #pitch.scatter(x1, y1, ax=ax2, color=color, marker='.', s=80)
                #pitch.scatter(x2, y2, ax=ax2, color=color, marker='.', s=150)
                pitch.lines(transparent= True , linewidth = 2, comet = True, xstart = x1, ystart = y1, xend=x2,yend=y2,ax=ax2,color=color)


            if row['type'] == 'Pass' and row['completed_pass'] == True:
                x1 = row['x']
                y1 = row['y']
                x2 = row['pass_end_x']
                y2 = row['pass_end_y']
                
                
                color = 'orange'

                pitch.lines(transparent= True , linewidth = 2, comet = True, xstart = x1, ystart = y1, xend=x2,yend=y2,ax=ax2,color=color)


        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        #cropped_img = img.crop((130, 50, 660, 400)) #(left, upper, right, lower)
        cropped_img = img.crop((150, 30, 760, 530)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)



        #add_image(ax, visual_file_name, xy=(0.715, .17), zoom=0.7285)
        add_image(ax, cropped_img, xy=(0.715, .2), zoom=0.6285)

        log_memory_usage("After Prog") 

    if selected_card == 'Touch Map':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, 'Touch Map', fontsize= 20,fontproperties=bold)
        #add_basic_text(0.64, 0.51 - 0.0049, '(Set Piees Excluded)', fontsize= 12)

        from matplotlib.colors import LinearSegmentedColormap

        df_touches = events.loc[events.type.isin(['Pass', 'Ball Receipt*', 'Shot']), ['x', 'y']]
        flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 10 colors",
                                                  ['#e3aca7', '#c03a1d'], N=10)
        
        pitch = Pitch(line_color='white', line_zorder=2, pitch_color='#200020')
        fig2, ax2 = pitch.draw(figsize=(12, 8))
        hexmap = pitch.hexbin(df_touches.x, df_touches.y, ax=ax2, edgecolors='#f4f4f4',
                            gridsize=(12, 6), cmap=flamingo_cmap, mincnt=3)
                




        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        
        cropped_img = img.crop((10, 10, 1125, 770)) #(left, upper, right, lower)
        #cropped_img = img.crop((147, 20, 760, 550)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #img.save(visual_file_name)
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)



        
        add_image(ax, cropped_img, xy=(0.715, .285), zoom=0.345)
        # add_basic_text(0.51, 0.081, f'Goal Assist', fontsize= 14, ha = 'left')
        # add_basic_text(0.62, 0.031, f'Shot Assist', fontsize= 14, ha = 'left')
        #add_basic_text(0.735, 0.031, f'Off Target / Blocked', fontsize= 14, ha = 'left')

        log_memory_usage("After Touch") 

    if selected_card == 'Pressure Map':

        rect1 = patches.Rectangle( 
           (0.442, 0.33), #bottom left coords
            0.542, 0.2,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, 'Pressure Map', fontsize= 20,fontproperties=bold)
        #add_basic_text(0.64, 0.51 - 0.0049, '(Set Piees Excluded)', fontsize= 12)

        from matplotlib.colors import LinearSegmentedColormap

        df_touches = events.loc[events.type.isin(['Pressure']), ['x', 'y']]
        flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 10 colors",
                                                  ['#e3aca7', '#c03a1d'], N=10)
        
        pitch = Pitch(line_color='white', line_zorder=2, pitch_color='#200020')
        fig2, ax2 = pitch.draw(figsize=(12, 8))
        hexmap = pitch.hexbin(df_touches.x, df_touches.y, ax=ax2, edgecolors='#f4f4f4',
                            gridsize=(12, 6), cmap=flamingo_cmap, mincnt=3)
                




        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)

        # output_dir = './saved_images'  # Save it inside your Streamlit app folder
        # os.makedirs(output_dir, exist_ok=True)
        
        cropped_img = img.crop((10, 10, 1125, 770)) #(left, upper, right, lower)
        #cropped_img = img.crop((147, 20, 760, 550)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'
        #img.save(visual_file_name)
        #cropped_img.save(visual_file_name)


        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)



        
        add_image(ax, cropped_img, xy=(0.715, .285), zoom=0.345)
        # add_basic_text(0.51, 0.081, f'Goal Assist', fontsize= 14, ha = 'left')
        # add_basic_text(0.62, 0.031, f'Shot Assist', fontsize= 14, ha = 'left')
        #add_basic_text(0.735, 0.031, f'Off Target / Blocked', fontsize= 14, ha = 'left')

        log_memory_usage("After Pressure") 


    if selected_card == 'Radar':

        rect1 = patches.Rectangle( 
           (0.442, 0.05), #bottom left coords
            0.542, 0.48,  # Width, Height
            linewidth=0.25,  # Outline thickness
            edgecolor="purple",  # Outline color
            facecolor="purple",  # Fill color
            alpha=0.25  # Optional: Transparency
            )
        ax.add_patch(rect1)

        add_basic_text(0.48, 0.51, f'{player_nickname} Radar', fontsize= 20,fontproperties=bold, color = 'white')
        add_basic_text(0.965, 0.51, f'{minutes} mins', fontsize= 15,fontproperties=bold, color = 'white', ha = 'right')


        metrics = selected_radars

        metrics = [f"pct{metric}" if metric not in all_ratings else metric for metric in metrics]


        data1 = [comp_data.loc[comp_data['player_id'] == player_id,metric].values[0] for metric in metrics]
        

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        data1 += data1[:1]  # Repeat the first value to close the polygon
        angles += angles[:1]  # Repeat the first angle to close the polygon
        #print(angles)


        fig2, ax2 = plt.subplots(figsize=(16, 9), subplot_kw=dict(polar=True, facecolor='#200020'))

        label_radius = 110
        ax2.set_rorigin(-2)
        max_metric_length = 0
        for angle, metric in zip(angles[:-1], metrics):
            x = label_radius * np.cos(angle)
            y = label_radius * np.sin(angle)


            rotation = np.degrees(angle)
            if rotation > 90 and rotation < 270:
                rotation -= 180
                ha = 'right'
            else:
                ha = 'left'

            # ha = 'left' if -np.pi/2 <= angle <= np.pi/2 else 'right'
            # ax2.text(angle, label_radius, metric, ha=ha, va='center', fontsize=14, color='white')


            if metric[:3] == 'pct':
                metric = metric[3:]

            if metric in metric_replacements:
                metric = metric_replacements[metric]

            if len(metric) > max_metric_length:
                max_metric_length = len(metric)
            
            if ' ' in metric: 
                metric = metric.replace(' ', '\n')

            #print(metric, angle)
            
            if angle == 1.5707963267948966:
                # ax2.text(angle + 0.3, label_radius + 10, metric,
                #     ha=ha, va='center', fontsize=22, color='white',fontproperties=bold)
                ax2.text(angle, label_radius, metric,
                    ha='center', va='center', fontsize=22, color='white',fontproperties=bold)
                
            elif angle == 4.71238898038469:
                ax2.text(angle, label_radius, metric,
                    ha='center', va='center', fontsize=22, color='white',fontproperties=bold)

            else:
                ax2.text(angle, label_radius, metric,
                    ha=ha, va='center', fontsize=22, color='white',fontproperties=bold)
            
            


        fig2.patch.set_facecolor('#200020')
        fig2.set_facecolor('#200020')

        ax2.set_facecolor('#200020')


        ax2.spines['polar'].set_visible(False)

        ax2.plot(angles, [100] * len(angles), color='white', linewidth=2.25, linestyle='-')
        ax2.plot(angles, [75] * len(angles), color='white', linewidth=0.7, linestyle='-')
        ax2.plot(angles, [50] * len(angles), color='white', linewidth=0.7, linestyle='-')
        ax2.plot(angles, [25] * len(angles), color='white', linewidth=0.7, linestyle='-')

        if compare == 'No': 
            ax2.plot(angles, data1, color='green', linewidth=0.4, linestyle='-', marker='o', markersize=3)
            ax2.fill(angles, data1, color='green', alpha=0.95)

        if compare == 'Yes':
            add_basic_text(0.48, 0.48, f'vs {comp_player}', fontsize= 20,fontproperties=bold, color = 'red')
            add_basic_text(0.965, 0.48, f'{mins2} mins', fontsize= 15,fontproperties=bold, color = 'red', ha = 'right')
            data2 = [comp_data2.loc[comp_data2['player_id'] == player_id2,metric].values[0] for metric in metrics]
            data2 += data2[:1]
            ax2.plot(angles, data1, color='white', linewidth=2.5, linestyle='-', marker='o', markersize=3)
            ax2.fill(angles, data1, color='white', alpha=0.7)

            ax2.plot(angles, data2, color='red', linewidth=2.5, linestyle='-', marker='o', markersize=3)
            ax2.fill(angles, data2, color='red', alpha=0.55)


        ax2.set_xticks(angles[:-1])
        metrics = ["" for i in range(len(metrics))]
        ax2.set_xticklabels(metrics)

        ax2.set_yticks([])
        ax2.set_ylim(0, 100)

        ax2.plot(0, 0, 'ko', markersize=4, color='#200020')
        #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        #fig.subplots_adjust(left=0.25, right=0.75, top=0.75, bottom=0.25)
        fig2.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)



    



        buf = io.BytesIO()
        fig2.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        width, height = img.size 
        #print(img.size)
        
         # Save it inside your Streamlit app folder
        #os.makedirs(output_dir, exist_ok=True)

        cropped_img = img#img.crop((10, 10, 1125, 770)) #(left, upper, right, lower)
        #cropped_img = img.crop((147, 20, 760, 550)) #(left, upper, right, lower)
        #visual_file_name = os.path.join(output_dir, f"ShotMap.webp")
        visual_file_name = 'saved_images/ShotMap.webp'

        #f"/Users/malekshafei/Documents/ShotMap.webp"
        #img.save(visual_file_name)
        #cropped_img.save(visual_file_name)
        
        #fig2.savefig(visual_file_name)
        plt.close(fig2)

        zoom = 0.37
        
        # if compare == 'Yes': add_image(ax, visual_file_name, xy=(0.715, .25), zoom=zoom)
        # if compare == 'No': add_image(ax, visual_file_name, xy=(0.715, .275), zoom=zoom)
        if compare == 'Yes': add_image(ax, cropped_img, xy=(0.715, .25), zoom=zoom)
        if compare == 'No': add_image(ax, cropped_img, xy=(0.715, .275), zoom=zoom)

        log_memory_usage("After Radar") 


    comp_data['Team'] = comp_data['Team'].apply(lambda x: replace_team_names(x, team_replacements))
    #print(comp_data.loc[comp_data['player_id'] == player_id, 'Team'])
    team_name = comp_data.loc[comp_data['player_id'] == player_id, 'Team'].values[0]
    add_basic_text(0.075, 0.87, 'Player Info', fontsize= 20, fontproperties=bold)


    add_basic_text(0.075, 0.84, 'Club', fontsize= 14.5, color = 'gray')
    add_basic_text(0.075, 0.822, team_name, fontsize= 14.5)

  # Force memory cleanup


        


        
        

        # fig,ax = pitch.draw(figsize=(6,8))
        # fig.set_facecolor('#400179')


    st.pyplot(fig)

    log_memory_usage("After Final") 

del events
del comp_data
del player_data
if selected_card == 'Radar':
   if compare == 'Yes': del player_data2
del recent_player_data
del player_map_file

# if memory_usage > 1100:
#     print('>1300')
#     for key in list(st.session_state.keys()):
#         print(key)
#         del st.session_state[key]
#     log_memory_usage("After Everything") 
    
#     gc.collect()
#     log_memory_usage("After Everything") 



import gc
gc.collect()
st.sidebar.header("Resource Usage")
# st.sidebar.write(f"**CPU Usage:** {cpu_usage:.2f}%")
st.cache_data.clear()
st.sidebar.write(f"**Memory Usage:** {memory_usage:.2f} MB")


if st.button("Hard Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    
    st.cache_resource.clear()
    gc.collect()
    st.rerun()
    
# st.sidebar.write(f"**Disk Usage:** {disk_usage:.2f} GB")

#log_memory_usage("After Everything") 


