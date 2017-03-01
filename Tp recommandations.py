# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:16:53 2016

@author: Macbook
"""


# coding: utf-8

# 
# # Espace latent et factorisation matricielle
# 
# Le problème dit de recommendation consiste à partir d'un historique d'avis que des utilisateurs ont donné sur  un ensemble d'items de prédire les avis non observés (par exemple des notes sur des films, des achats de produits, ...). Les techniques de factorisation matricelle permettent dans ce cadre (et plus généralement) de mettre en évidence un espace "latent" (non directement observé dans les données) qui explicite les corrélations entre les avis des utilisateurs et les produits (par exemple genre des films ou profil d'utilisateurs).
# 
# Nous allons utiliser dans la suite les données movielens de notes d'utilisateurs sur des films.
# 
# ## Stochastic gradient descent
# 
# Soit un ensemble de $m$ utilisateurs $U$ et un ensemble de $n$ items $I$, et une matrice $R$ de dimension $m\times n $ telle que $r_{u,i}$ représente la note que l'utilisateur $u$ a donné à l'item $i$, 0 si pas de note correspondante (on suppose les notes $>0$). Dans cette partie, l'hypothèse est qu'il existe un espace latent de dimension $d$ "commun" aux utilisateurs et aux items, qui permet d'expliquer par une combinaison linéaires les goùts des utilisateurs. Dans le cadre de la recommendation de films par exemple, cet espace pourrait décrire des genres de films. A chaque utilisateur $u$ correspond un vecteur de pondération $x$ de taille $d$ qui indique les intérêts de l'utilisateur en fonction de chaque genre; à chaque film $i$ correspond un vecteur de pondération $y$ de taille $d$ qui indique la corrélation du film avec chaque genre.
# 
# L'objecitf est ainsi de trouver deux matrices $X$ et $Y$ de tailles $m\times d$ et $d\times n$ telles que $R \approx X Y$.
# On utilise en général les moindres carrés comme fonction de coût pour calculer l'erreur de reconstruction.
# Comme seule une partie des scores est observée lors de l'apprentissage, l'erreur ne doit être comptée que pour ces scores là.
# La fonction à optimiser est $min_{X,Y} \sum_{u,i|r_{u,i}>0} (r_{u,i}-y_{.,i}'x_{u,.})^2+\lambda (||y_{.,i}||^2+||x_{u,.}||^2)$.
# 
# Une manière d'optimiser la fonction est par descente de gradient stochastique, avec les formules de mise-à-jour suivantes :
# 
# * $e_{u,i} =  r_{ui}-y_{.,i}'x_{u,.}$
# 
# * $y_{.,i} = y_{.,i}+\gamma (e_{u,i}x_{u,.} -\lambda q_{.,i})$
# 
# * $x_{u,.} = x_{u,.}+\gamma (e_{u,i}y_{.,i} -\lambda x_{u,.})$
# 
# 
# Le code suivant permet de charger les données movielens (données à charger à partir de http://grouplens.org/datasets/movielens/1m/, le MovieLens 1M Dataset par exemple).

# In[2]:

#import numpy as np
#import matplotlib.pyplot as plt
#def read_movielens():
#    users = dict()
#    movies = dict()
#    with open('users.dat', "r") as f:
#        for l in f:
#            l = l.strip().split("::")
#            users[int(l[0])] = [l[1], int(l[2]), int(l[3]), l[4]]
#    with open('movies.dat', "r") as f:
#        for l in f:
#            l = l.strip().split("::")
#            movies[int(l[0])] = [l[1],l[2].split("|")]
#    ratings = np.zeros((max(users)+1,max(movies)+1))
#    with open('ratings.dat',"r") as f:
#        for l in f:
#            l = l.strip().split("::")
#            ratings[int(l[0]),int(l[1])]=int(l[2])
#    return ratings, users, movies
#
#
#### nb_movies, nb_users : nombre de movies, users
#### ratings : matrice users x movies avec les notes
#### users : dictionnaire des users avec les infos 
#### movies : dictionnaire des films avec les infos
#### genres, genres_id : mapping des genres vers identifiants
#### movies_genres : matrice binaire  movies x genres avec un 1 si film du genre 
#ratings,users,movies = read_movielens()
#nb_movies = max(movies)+1
#nb_users = max(users)+1
#genres =  {0: 'Action', 1: 'Adventure', 2: 'Animation', 3: "Children's", 4: 'Comedy', 
#           5: 'Crime', 6: 'Documentary', 7: 'Drama', 8: 'Fantasy', 9: 'Film-Noir',
#           10: 'Horror', 11: 'Musical', 12: 'Mystery', 13: 'Romance', 14: 'Sci-Fi',
#           15: 'Thriller', 16: 'War', 17: 'Western'}
#genres_id = dict(zip(genres.values(),genres.keys()))
#movies_genres = np.zeros((nb_movies,len(genres)))
#for idx,m in movies.items():
#    for g in m[1]:
#        movies_genres[idx-1][genres_id[g]]=1



# Implémenter l'algorithme. Tester le sur les données movielens et analyser les résultats.

#choix des données d'apprentissage aléatoirement
#on ne choisit qu'une partie des films
#indices_movies = np.array(range(ratings.shape[1]))
#np.random.shuffle(indices_movies)
#indices = indices_movies[0:indices_movies.shape[0]/3*2]
#R = ratings[:, indices]
##
##
###initialisation
#m = R.shape[0]
#n = R.shape[1]
#d = movies_genres.shape[1]
#X = np.ones((m,d))
#Y = movies_genres[indices, :]
#Y = Y.transpose()
gamma = 0.07
lamb = 0.3
max_iter = 900000

#descente stochastique
#verification par la fonction de cout
iteration = 0
loss =0

for t in range(max_iter):
	if (t%80000 == 0):
		iteration = np.append(t,[iteration])
#trouver un élément aléatoirement
	u = np.random.randint(0,m)
	i = np.random.randint(0,n)
	while (R[u,i] == 0):
		u = np.random.randint(0,m)
		i = np.random.randint(0,n)
#	print u,i
	#actualiser les valeurs
	e_ui =  R[u,i]-X[u,:].dot(Y[:,i])
#	print 'eui', e_ui
#	print 'colonne Y.,i', Y[:,i]
#	print 'ligne Xu,.', X[u,:]
	Y[:,i] = Y[:,i] + gamma*(e_ui*X[u,:] -lamb*Y[:,i])
#	print 'new colonne i Y', Y[:,i]
	X[u,:] = X[u,:] + gamma*(e_ui*Y[:,i] - lamb*X[u,:])

	#calculer la fonction de coût
	if (t%80000 == 0):
		hinge_loss = 0
		hinge_loss = np.square(R - X.dot(Y))
		
		indices = (R != 0)
		
#		Y_norm = np.square(np.linalg.norm(Y, axis=0))
#		Y_norm = np.tile(Y_norm, (m, 1))
	#	print 'dim Y', Y_norm.shape[0], Y_norm.shape[1]
#		Y_norm = Y_norm.reshape((m, n))
		
		X_norm = np.square(np.linalg.norm(X, axis=1))
		X_norm = np.tile(X_norm.transpose(), (n,1)).transpose()
	#	print 'dim X', X_norm.shape[0], X_norm.shape[1]
		X_norm = X_norm.reshape((m,n))
		
		hinge_loss = hinge_loss + lamb*Y_norm #lamb*X_norm
		hinge_loss = hinge_loss[indices]
		hinge_loss = np.sum(hinge_loss)
		print 'coût', hinge_loss
		
		loss = np.append(hinge_loss,[loss])
	

#plot erreur
plt.figure()
plt.xlabel("iterations")
plt.ylabel("hinge_loss")


plt.plot(iteration,loss)
plt.show()

#test sur les données restantes
indices_test = indices_movies[indices_movies.shape[0]/3*2+1:]
R_test = ratings[:, indices_test]
Y_test = movies_genres[indices_test, :]

hinge_loss_test = 0
hinge_loss_test = np.square(R_test - X.dot(Y_test.transpose()))

indices_ratings = (R_test != 0)

hinge_loss_test = hinge_loss_test[indices_ratings]
hinge_loss_test = np.sum(hinge_loss_test)
print 'coût sur l ensemble de test', hinge_loss



# Comment choisir un ensemble de test ?
# Observer bien l'effet de la régularisation sur la convergence et sur le sur-apprentissage.
# 
# Bonus : proposer une adaptation de l'algo pour prendre en compte les biais pour les utilisateurs et pour les films.

# <h2>Visualisation : algorithmes MDS et t-SNE</h2>
# Afin de pouvoir visualiser les données, nous avons  besoin d'une réduction de dimension très forte (en 2 ou 3d ...). L'objectif de la plupart des algorithmes dans ce domaine est de préserver les distances locales lors des projections. Deux exemples d'algorithmes sont  "MultiDimensional scaling" (MDS) et t-Stochastic Neighbor Embedding (t-SNE). MDS utilise une approche algébrique par identification de valeurs propres à partir d'une matrice de similarité entre données; t-SNE utilise  une modélisation probabiliste en étudiant la KL-divergence entre la distribution des points dans l'espace originale et dans l'espace projeté.
# 
# Scikit-learn implémente ces deux algos. Etudier les résultats précédents en visualisant les données à partir de la nouvelle représentation et par MDS ou t-SNE. Vous pouvez utiliser pour cela une similarité cosinus : $\frac{u . v}{||u||||v||}$ mieux adapté aux grandes dimensions (pourquoi ?) pour MDS, et les coordonnées des films dans l'espace latent trouvé dans les questions précédentes pour t-SNE. Utiliser l'information sur les genres pour analyser vos résultats.
# 
# documentation : 
# http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds
# 
