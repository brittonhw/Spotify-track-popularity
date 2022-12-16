

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pylab import savefig





#read csv
df = pd.read_csv('/content/gdrive/My Drive/cs260/dataset.csv')

#drop unused variables
df = df.drop(['Unnamed: 0', 'track_id', 'album_name', 'track_name'], axis = 1)

#copy in case un-processed is needed
cdf = df.copy()


#label encoder -> modifed from kaggle, switch allows for panda dummy mode, but make sure to drop artists
def labelencoder(df, switch = True):
    if switch:
      for c in df.columns:
          if df[c].dtype=='object': 
              df[c] = df[c].fillna('N')
              lbl = LabelEncoder()
              lbl.fit(list(df[c].values))
              df[c] = lbl.transform(df[c].values)
      return df
    else:
      return pd.get_dummies(df, columns=['track_genre'])

df = labelencoder(df, True)
X = df.loc[:,df.columns != 'popularity'].values
y = df['popularity'].values

#found this tactice for splitting on sci-kit train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
print(df.head())

genres = ['rock', 'pop', 'ska', 'hip-hop', 'alternative', 'country', 'jazz']
#possible_genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'honky-tonk', 'house', 'idm', 'indian', 'indie-pop', 'indie', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metalcore', 'minimal-techno', 'mpb', 'new-age', 'opera', 'pagode', 'party', 'piano', 'pop-film', 'pop', 'power-pop', 'progressive-house', 'psych-rock', 'punk-rock', 'punk', 'r-n-b', 'reggae', 'reggaeton', 'rock-n-roll', 'rock', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'world-music']

#plot stripplot for genre popularity on above specified genres
filter = cdf.loc[cdf['track_genre'].isin(genres)]
sns.set_context("paper", font_scale=1.1)
plt.figure(figsize=(10, 10))
sns.stripplot(x="track_genre", y="popularity", data=filter).set(title='Popularity by Genre')
plt.savefig('genres.png')

#plot artist stripplot
artists = ['The Lumineers',  'Troglauer', 'Fetty Wap', 'Dua Lipa', 'Taylor Swift' ]
filter_artist = cdf.loc[cdf['artists'].isin(artists)]
sns.set_context("paper", font_scale=1.1)
plt.figure(figsize=(10, 10))
sns.stripplot(x="artists", y="popularity", data=filter_artist).set(title='Popularity by Artist')
plt.savefig('artists.png')

#ignore unless track genre (or artists) are needed
def unique_track_genre(df):
  track_genre = df['track_genre']
  unique_track_genre = []
  for i in track_genre:
    if i not in unique_track_genre:
      unique_track_genre.append(i)
  return unique_track_genre

#print(unique_track_genre(cdf))

#generate heatmap
sns.set(font_scale = 0.25)
svm = sns.heatmap(df.corr(), annot=True)

figure = svm.get_figure() 
figure.savefig('heatmap1.png', dpi=400)

#sample random 1000 to clear noise 
df3d = df.sample(1000)


#3d plot (unused, too unclear)
xdata = df3d['energy']
ydata = df3d['danceability']
zdata = df3d['popularity']

fig = plt.figure(figsize=(9, 6))
# Create 3D container
ax = plt.axes(projection = '3d')
# Visualize 3D scatter plot
ax.scatter3D(xdata, ydata, zdata)
# Give labels
ax.set_xlabel('energy')
ax.set_ylabel('danceability')
ax.set_zlabel('popularity')

#ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')





#plot energy and dancability 
sns.scatterplot(x="energy", y="danceability", hue="popularity", data=df3d).set(title='Popularity against Energy and Danceability ')
plt.savefig('dancevenergy.png')


#find 500 most popular tracks and sort
dfcount = cdf.sort_values(by=['popularity']).tail(500)

#plot against genre
sns.countplot(x="track_genre", data=dfcount).set(title='500 Most Popular Tracks by Genre')
plt.xticks([])
plt.savefig('genrestop.png')



dfcount = cdf.sort_values(by=['popularity']).tail(100)
#plot against artists
sns.countplot(x="artists", data=dfcount).set(title='100 Most Popular Tracks by Artist')
plt.xticks([])
plt.savefig('topartists.png')
dfcount = cdf.sort_values(by=['popularity']).tail(1000)


#plot side by side explict graph
fig, ax =plt.subplots(1,2)
fig.subplots_adjust(hspace=0.125, wspace=1.125)
sns.countplot(x="explicit", data=dfcount, ax=ax[0]).set(title='Explicit Visualization of Most Popular 1000')
sns.countplot(x="explicit", data=df, ax=ax[1]).set(title='Explicit Visualization of All Data')


plt.savefig('explicitstogether.png')
#plt.show()


#explict individual, not necessary just for playing around
sns.countplot(x="explicit", data=df).set(title='Explicit Visualization of All Data')
plt.savefig('explicittotal.png')
#plt.show()


#here it can be tested to find combinations of simularity between different atributes. I found energy and dancability, might 
#be more but there are a lot of combinations
drops = cdf.drop(['track_genre', 'explicit', 'artists', 'key','mode', 'time_signature', 'duration_ms', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'], axis = 1)
means = drops.mean()
means['popularity']
labels = []
for col in drops.columns:
  if col != 'popularity':
    labels.append(col)
filtered_means = drops.copy()
is_mean = True
for i in labels:
  filtered_means = filtered_means.loc[(filtered_means[i] < (means[i] * 1.25)) & (filtered_means[i] > (means[i]*0.75))]
filtered_means_reduce = filtered_means.sample(1000)
#filters by only keeping similar rows to mean, excluding popularity, than plotting against popularity to see clumping
sns.scatterplot(x="energy", y="danceability", hue="popularity", data=filtered_means_reduce).set(title='Popularity of Similar Songs')
plt.savefig('different_means.png')


#bayesian ridge, scores badly
reg = linear_model.BayesianRidge(verbose = True)
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
print('score: ', score)
reg_predictions = reg.predict(X_test)

print(mean_squared_error(y_test, reg.predict(X_test)))


#linear model, even worse, seems to require increible eta0, unsure of best hyperparameters
sgd = linear_model.SGDRegressor(max_iter=10000, tol=1e-3, power_t=0.001, eta0=0.00000000000001, early_stopping=True)
sgd.fit(X_train, y_train)
sgd_predictions = sgd.predict(X_test)
print(mean_squared_error(y_test, sgd_predictions))
#print(mean_squared_error(test_answers, sgd_predictions))

sgd.score(X_test, y_test)


#seems best model, from XGBR, influenced choice of this by the kaggle, performs best but still not great
xgbr = XGBRegressor(scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.3,
                      subsample = 0.8,
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=10, 
                      gamma=1
                      )
xgbr.fit(X_train, y_train)
xgbr_predictions = xgbr.predict(X_test)

#print('score: ', xgbr.score(tests, test_answers))
print(mean_squared_error(y_test, xgbr_predictions))

xgbr.score(X_test, y_test)

X_test[1:5]