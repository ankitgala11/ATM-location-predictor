import requests
import json
import time
import pprint
import folium
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim 
from pandas import json_normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
from io import BytesIO
import base64

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from geopy.extra.rate_limiter import RateLimiter

api="hYwNTC2Rvr9Tqq-lrldgamYG54WvXxiqrANplFwt69g"
base_url_discover = "https://discover.search.hereapi.com/v1/discover?apiKey="+api+"&"
base_url_geocode = "https://geocode.search.hereapi.com/v1/geocode?apiKey="+api+"&"


def build_map(curr_map, curr_df, color, fill):
    for lat, lng, label in zip(curr_df.Latitude, curr_df.Longitude, curr_df.Category+" - "+curr_df.Name):
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            color= color,
            popup=label,
            fill = True,
            fill_color= fill,
            fill_opacity=0.6
        ).add_to(curr_map)
    
def get_df(param, loctn):
    # param = ""
    endpoint_url = "in=circle:"+str(loctn['lat'])+","+str(loctn['lng'])+";r=1000&limit=100&q="+param
    endpoint_url = base_url_discover+endpoint_url
    res = requests.get(endpoint_url, headers={}).json()         
    resDf = pd.DataFrame.from_dict(res['items'])
    if len(res['items']) == 0:
        return [resDf, {}]
    resDf = pd.concat([resDf.drop(['position'], axis=1), resDf['position'].apply(pd.Series)], axis=1)
    resDf = resDf.drop(labels=['id', 'resultType', 'access', 'distance', 'references', 'contacts', 'ontologyId'], axis=1, errors='ignore')
    resDf = pd.concat([resDf.drop(['categories'], axis=1), resDf['categories'].apply(pd.Series)[0].apply(pd.Series)['name']], axis=1)
    resDf = pd.concat([resDf.drop(['address'], axis=1), resDf['address'].apply(pd.Series)[['city', 'district', 'postalCode']]], axis=1)
    resDf.rename(columns = {
        'name':'Category', 
        'lat':'Latitude', 
        'lng':'Longitude', 
        'city':'City', 
        'district': 'District', 
        'postalCode':'Postal Code',
        'title':'Name'
    }, inplace = True)
    
    curr_map = folium.Map(location=[loctn['lat'],loctn['lng']], zoom_start=14)
    build_map(curr_map, resDf, 'blue', 'blue')
    
    return [resDf,curr_map]

def model(location, play=True):  
    queryLocation = location
    # with open('result_json.json') as json_file:
    #     data = json.load(json_file)
    #     lclresult = []
    #     if data:
    #         for idx, c in enumerate(data['data']):
    #             c_data = json.loads(c)
    #             cdf = pd.DataFrame(c_data)
    #             data['data'][idx] = cdf
    #             cl = idx
    #             fl = ((cdf['1st Most Common Venue'] +"_$_"+ cdf['2nd Most Common Venue'] +"_$_"+ cdf['3rd Most Common Venue'] +"_$_"+ cdf['4th Most Common Venue'] +"_$_"+ cdf['5th Most Common Venue']).mode()[0]).split("_$_")
    #             fl.insert(0, cl)
    #             lclresult.append(fl)
    # data['result'] = pd.DataFrame(lclresult, columns=['Cluster Label', '1st MCV', '2nd MCV', '3rd MCV', '4th MCV', '5th MCV'])

    # if play==False and data['location'] and data['location']==location:
    #     return data['result']
    # else:
    #     del data

    print(">>>>> check 1")
    # endpoint_param = "Vancouver, VAN"
    endpoint_param = location
    endpoint_url = base_url_geocode+"q="+endpoint_param
    location = requests.get(endpoint_url, headers={}).json()
    loctn = location['items'][0]['position']

    geolocator = Nominatim(user_agent="van_explorer")
    geocode = RateLimiter(geolocator.reverse,min_delay_seconds = 1, return_value_on_exception = None) 
    big_map = folium.Map(location=[loctn['lat'],loctn['lng']], tiles="cartodbpositron", zoom_start=14)
    poi_map = folium.Map(location=[loctn['lat'],loctn['lng']], tiles="cartodbpositron", zoom_start=14)
    
    [atm1_df, atm_map] = get_df('atm', loctn)
    if not(atm1_df.empty) and  bool(atm_map):
        build_map(big_map, atm1_df, 'red', 'red')
        atm1_df = atm1_df[ (atm1_df['Category'] == 'Bank') | (atm1_df['Category'] == 'ATM')].reset_index(drop=True)
    
    [transport_df, transport_map] = get_df('station', loctn)
    if not(transport_df.empty) and  bool(transport_map):
        build_map(poi_map, transport_df, 'blue', 'blue')  
        build_map(big_map, transport_df, 'blue', 'lightblueblue')  

    [gas_df, gas_map] = get_df('gas', loctn)
    if not(gas_df.empty) and  bool(gas_map):
        build_map(poi_map, gas_df, 'gray', 'gray')
        build_map(big_map, gas_df, 'blue', 'blue')

    [market_df, market_map] = get_df('market', loctn)
    if not(market_df.empty) and  bool(market_map):
        build_map(poi_map, market_df, 'orange', 'orange')
        build_map(big_map, market_df, 'blue', 'blue')

    [mall_df, mall_map] = get_df('mall', loctn)
    if not(mall_df.empty) and  bool(mall_map):
        build_map(poi_map, mall_df, 'pink', 'pink')
        build_map(big_map, mall_df, 'blue', 'blue')
    
    [shop_df, shop_map] = get_df('supermarket', loctn)
    if not(shop_df.empty) and  bool(shop_map):
        build_map(poi_map, shop_df, 'lightgreen', 'lightgreen')
        build_map(big_map, shop_df, 'blue', 'blue')
    
    big_map.save('templates/footmap.html')
    poi_map.save('templates/poimap.html')

    print(">>>>> check 2")
    all_df = pd.concat([transport_df, gas_df, market_df, mall_df])
    all_df = all_df.drop(labels=['id', 'resultType', 'access', 'distance', 'references', 'contacts', 'ontologyId', 'chains', 'openingHours', 'foodTypes', 'index'], axis=1, errors='ignore')
    all_df.reset_index(inplace = True, drop = True)
    for index, row in all_df.iterrows():
        try:
            location = str(geolocator.reverse("{}, {}".format(row['Latitude'], row['Longitude'])))           
            location = location.split(', ')[2]        
            all_df.loc[index,'Neighborhood'] = location
        except:
            all_df.loc[index,'Neighborhood'] = ''                  
    all_df.to_pickle('getNeighbourhoodData.pkl')
    all_df2 = pd.read_pickle('getNeighbourhoodData.pkl')
    for index, row in all_df.iterrows():
        try:        
            if all_df2.iloc[index]['Latitude'] == row['Latitude'] and all_df2.iloc[index]['Longitude'] == row['Longitude']:
                all_df.loc[index,'Neighborhood'] = all_df2.iloc[index]['Neighborhood']
            else:
                all_df.loc[index,'Neighborhood'] = ''        
        except:
            all_df.loc[index,'Neighborhood'] = ''
            pass

    all_df = all_df.drop_duplicates(subset =["Latitude", "Name", "Longitude"],keep = 'first')
    all_df = all_df.drop_duplicates(subset =["Latitude", "Longitude",  "Category"],keep = 'first')
    all_df = all_df.dropna()

    atm_df = atm1_df.drop(labels=['id', 'resultType', 'access', 'distance', 'references', 'contacts', 'ontologyId', 'chains', 'openingHours', 'foodTypes', 'index'], axis=1, errors='ignore')
    atm_df.reset_index(inplace = True, drop = True)

    print(">>>>> check 3")
    geolocator = Nominatim(user_agent = 'ATM_explor')
    for index, row in atm_df.iterrows():
        try:
            location = str(geolocator.reverse("{}, {}".format(row['Latitude'], row['Longitude'])))           
            location = location.split(', ')[2]        
            atm_df.loc[index,'Neighborhood'] = location
        except:
            atm_df.loc[index,'Neighborhood'] = ''                  
    atm_df.to_pickle('getAtmNeighbourhoodData.pkl')
    atm_df2 = pd.read_pickle('getATMNeighbourhoodData.pkl')
    for index, row in atm_df2.iterrows():
        try:        
            if atm_df2.iloc[index]['Latitude'] == row['Latitude'] and atm_df2.iloc[index]['Longitude'] == row['Longitude']:
                atm_df.loc[index,'Neighborhood'] = atm_df2.iloc[index]['Neighborhood']
            else:
                atm_df.loc[index,'Neighborhood'] = ''        
        except:
            atm_df.loc[index,'Neighborhood'] = ''
            pass
    atm_df = atm_df.drop_duplicates(subset =["Latitude", "Name", "Longitude"],keep = 'first')
    atm_df = atm_df.dropna()
    big_df = pd.concat([all_df,atm_df])
    big_df.reset_index(inplace = True, drop = True)
    big_df.groupby('Neighborhood').count()
    mapping = {
        "ATM":"ATM",
        "Bank":"ATM",
        "Public Transport":"Public Transport",
        "Underground Train-Subway":"Public Transport",
        "Bus Stop":"Public Transport",
        "Bus Station":"Public Transport",
        "Bus Rapid Transit":"Public Transport",
        "Commuter Train":"Public Transport",
        "Railway Station":"Public Transport",
        "Water Transit":"Public Transport",
        "Light Rail":"Public Transport",
        "Supermarket":"Supermarket",
        "Specialty Store":"Supermarket",
        "Convenience Store":"Supermarket",
        "Women's Apparel":"Supermarket",
        "Market":"Supermarket",
        "Grocery":"Supermarket",
        "Mall":"Mall",
        "Museum":"Mall",
        "Nightlife-Entertainment":"Mall",
        "Shopping Mall":"Mall",
        "Gas Station":"Gas Station",
    }

    for index, row in big_df.iterrows():
        if big_df.iloc[index]['Category'] in mapping:
            big_df.loc[index,'Category'] = mapping[big_df.iloc[index]['Category']]
    print(">>>>> check 4")
    concat_onehot = pd.get_dummies(big_df[['Category']], prefix="", prefix_sep="")
    concat_onehot['Neighborhood'] = big_df['Neighborhood'] 
    fixed_columns = [concat_onehot.columns[-1]] + list(concat_onehot.columns[:-1])
    concat_onehot = concat_onehot[fixed_columns]
    concat_grouped = concat_onehot.groupby('Neighborhood').mean().reset_index()
    def return_most_common_venues(row, num_top_venues):
        row_categories = row.iloc[1:]
        row_categories_sorted = row_categories.sort_values(ascending=False)
        
        return row_categories_sorted.index.values[0:num_top_venues]
    
    num_top_venues = 5
    indicators = ['st', 'nd', 'rd']

    # create columns according to number of top venues
    columns = ['Neighborhood']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))

    # create a new dataframe
    concat_venues_sorted = pd.DataFrame(columns=columns)
    concat_venues_sorted['Neighborhood'] = concat_grouped['Neighborhood']
    print(">>>>> check 5")
    for ind in np.arange(concat_grouped.shape[0]):
        concat_venues_sorted.iloc[ind, 1:] = return_most_common_venues(concat_grouped.iloc[ind, :], num_top_venues)

    concat_venues_sorted.head()
    k_rng = range(1,10)
    sse = []
    for k in k_rng:
        km = KMeans(n_clusters=k)
        try:
            km.fit(concat_onehot[['ATM', 'Gas Station', 'Mall', 'Supermarket', 'Public Transport']])
        except:
            try:
                km.fit(concat_onehot[['ATM', 'Mall', 'Supermarket', 'Public Transport']])
            except:
                km.fit(concat_onehot[['ATM', 'Supermarket', 'Public Transport']])
        sse.append(km.inertia_)




    # set number of clusters
    kclusters = 5

    concat_grouped_clustering = concat_grouped.drop('Neighborhood', 1)

    # run k-means clustering
    kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(concat_grouped_clustering)
    print(kmeans)
    # check cluster labels generated for each row in the dataframe
    kmeans.labels_[0:10] 
    concat_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

    concat_merged = big_df[['Name', 'Category', 'Latitude', 'Longitude', 'Postal Code', 'City', 'Neighborhood']]

    # merge concat_grouped with concat_venues_sorted to add latitude/longitude for each neighborhood
    concat_merged = concat_merged.join(concat_venues_sorted.set_index('Neighborhood'), on='Neighborhood')    
    print(">>>>> check 6")
    # create map
    map_clusters = folium.Map(location=[loctn['lat'],loctn['lng']], tiles="cartodbpositron", zoom_start=15)

    # set color scheme for the clusters
    x = np.arange(kclusters)
    ys = [i + x + (i*x)**2 for i in range(kclusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]

    # add markers to the map
    markers_colors = []
    for lat, lon, poi, cluster in zip(concat_merged['Latitude'], concat_merged['Longitude'], concat_merged['Neighborhood'], concat_merged['Cluster Labels']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=rainbow[cluster-1],
            fill=True,
            fill_color=rainbow[cluster-1],
            fill_opacity=0.7).add_to(map_clusters)
        
    map_clusters.save('templates/map.html')


    clusters = [
        concat_merged.loc[concat_merged['Cluster Labels'] == 0, concat_merged.columns[[1] + list(range(5, concat_merged.shape[1]))]].reset_index(drop=True),
        concat_merged.loc[concat_merged['Cluster Labels'] == 1, concat_merged.columns[[1] + list(range(5, concat_merged.shape[1]))]].reset_index(drop=True),
        concat_merged.loc[concat_merged['Cluster Labels'] == 2, concat_merged.columns[[1] + list(range(5, concat_merged.shape[1]))]].reset_index(drop=True),
        concat_merged.loc[concat_merged['Cluster Labels'] == 3, concat_merged.columns[[1] + list(range(5, concat_merged.shape[1]))]].reset_index(drop=True),
        concat_merged.loc[concat_merged['Cluster Labels'] == 4, concat_merged.columns[[1] + list(range(5, concat_merged.shape[1]))]].reset_index(drop=True),
    ]
    clusters_json_data = []
    for idx, c in enumerate(clusters):
        clusters_json_data.append(c.to_json())
    
    clusters_json = { 'location' : queryLocation, 'data' : clusters_json_data }    

    print(">>>>> check 7")
    # with open('result_json.json', 'w') as outfile:
    #     json.dump(clusters_json, outfile)
    
    lclresult = []
    for idx, c in enumerate(clusters_json['data']):
        c_data = json.loads(c)
        cdf = pd.DataFrame(c_data)
        clusters_json['data'][idx] = cdf
        cl = c_data['Cluster Labels']['0']
        fl = ((cdf['1st Most Common Venue'] +"_$_"+ cdf['2nd Most Common Venue'] +"_$_"+ cdf['3rd Most Common Venue'] +"_$_"+ cdf['4th Most Common Venue'] +"_$_"+ cdf['5th Most Common Venue']).mode()[0]).split("_$_")
        fl.insert(0, cl)
        lclresult.append(fl)
    clusters_json['result'] = pd.DataFrame(lclresult, columns=['Cluster Label', '1st MCV', '2nd MCV', '3rd MCV', '4th MCV', '5th MCV'])

    return clusters_json['result']