

```python
import numpy as np
import pickle
from scipy.stats import binom
import matplotlib.pyplot as plt
import pandas as pd 
import pickle

%matplotlib inline

np.random.seed(0)
```


```python
data = pd.read_table('../data/data_nam.txt', sep=' ',header=0)
```


```python
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndivID</th>
      <th>PopID</th>
      <th>Pop</th>
      <th>Country</th>
      <th>Continent</th>
      <th>sex</th>
      <th>lat</th>
      <th>long</th>
      <th>L1.125</th>
      <th>L1.130</th>
      <th>...</th>
      <th>L677.255.553287981859</th>
      <th>L677.259</th>
      <th>L677.263</th>
      <th>L677.267</th>
      <th>L678.202</th>
      <th>L678.206</th>
      <th>L678.209.848101265823</th>
      <th>L678.210</th>
      <th>L678.214</th>
      <th>L678.218</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chipewyan29</th>
      <td>2012</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan31</th>
      <td>2156</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan33</th>
      <td>2381</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan35</th>
      <td>2382</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan37</th>
      <td>2383</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5717 columns</p>
</div>



# Analysis

Names of the different populations.


```python
print('Names of the different populations:\n', np.unique(data['Pop']), '\n')
print('Number of populations: ', len(np.unique(data['Pop'])))
```

    Names of the different populations:
     ['Ache' 'Arhuaco' 'Aymara' 'Cabecar' 'Chipewyan' 'Cree' 'Embera' 'Guarani'
     'Guaymi' 'Huilliche' 'Inga' 'Kaingang' 'Kaqchikel' 'Karitiana' 'Kogi'
     'Maya' 'Mixe' 'Mixtec' 'Ojibwa' 'Piapoco' 'Pima' 'Quechua' 'Surui'
     'TicunaArara' 'Waunana' 'Wayuu' 'Zapotec'] 
    
    Number of populations:  27



```python
print('Coordinates of the different populations:\n')
data[['Pop', 'long', 'lat']].set_index('Pop').drop_duplicates()
```

    Coordinates of the different populations:
    

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>long</th>
      <th>lat</th>
    </tr>
    <tr>
      <th>Pop</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chipewyan</th>
      <td>-107.3</td>
      <td>59.55</td>
    </tr>
    <tr>
      <th>Cree</th>
      <td>-102.5</td>
      <td>50.33</td>
    </tr>
    <tr>
      <th>Ojibwa</th>
      <td>-81.0</td>
      <td>46.50</td>
    </tr>
    <tr>
      <th>Kaqchikel</th>
      <td>-91.0</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>Mixtec</th>
      <td>-97.0</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>Mixe</th>
      <td>-96.0</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>Zapotec</th>
      <td>-97.0</td>
      <td>16.00</td>
    </tr>
    <tr>
      <th>Guaymi</th>
      <td>-82.0</td>
      <td>8.50</td>
    </tr>
    <tr>
      <th>Cabecar</th>
      <td>-84.0</td>
      <td>9.50</td>
    </tr>
    <tr>
      <th>Aymara</th>
      <td>-70.0</td>
      <td>-22.00</td>
    </tr>
    <tr>
      <th>Huilliche</th>
      <td>-73.0</td>
      <td>-41.00</td>
    </tr>
    <tr>
      <th>Guarani</th>
      <td>-54.0</td>
      <td>-23.00</td>
    </tr>
    <tr>
      <th>Ache</th>
      <td>-56.0</td>
      <td>-24.00</td>
    </tr>
    <tr>
      <th>Kaingang</th>
      <td>-52.5</td>
      <td>-24.00</td>
    </tr>
    <tr>
      <th>Quechua</th>
      <td>-74.0</td>
      <td>-14.00</td>
    </tr>
    <tr>
      <th>Kogi</th>
      <td>-74.0</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Inga</th>
      <td>-77.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Wayuu</th>
      <td>-73.0</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>TicunaArara</th>
      <td>-70.0</td>
      <td>-4.00</td>
    </tr>
    <tr>
      <th>Embera</th>
      <td>-76.0</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>Waunana</th>
      <td>-77.0</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>Arhuaco</th>
      <td>-73.8</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>Piapoco</th>
      <td>-68.0</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>Karitiana</th>
      <td>-63.0</td>
      <td>-10.00</td>
    </tr>
    <tr>
      <th>Surui</th>
      <td>-62.0</td>
      <td>-11.00</td>
    </tr>
    <tr>
      <th>Maya</th>
      <td>-91.0</td>
      <td>19.00</td>
    </tr>
    <tr>
      <th>Pima</th>
      <td>-108.0</td>
      <td>29.00</td>
    </tr>
  </tbody>
</table>
</div>



## PCA and Regression : Data


```python
data_aux = data.copy()
data_aux = data_aux.drop(data.columns[0:6], axis=1)
x = data_aux.drop(data_aux.columns[0:2], axis=1).values
y_long = data['long'].values
y_lat = data['lat'].values
```


```python
# Save data.
# pickle.dump(x, open('../data/data_nam_clean.matrix', 'wb'), -1)
```


```python
from sklearn.decomposition import PCA

pca = PCA(100).fit(x)
comp = pca.transform(x)
```


```python
from sklearn.linear_model import LinearRegression

reg_long = LinearRegression()
reg_lat = LinearRegression()

reg_long.fit(comp, y_long)
reg_lat.fit(comp, y_lat)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
from sklearn.svm import SVR

reg_long = SVR()
reg_lat = SVR()

reg_long.fit(comp, y_long)
reg_lat.fit(comp, y_lat)
```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
x_long = reg_long.predict(comp)
x_lat = reg_lat.predict(comp)
```

## PCA and Regression : Generated Data


```python
import matplotlib.pyplot as plt 

x_gen = np.load('../generated/synthetic_data_nam.npy')
```


```python
x_gen = np.round(x_gen)
```


```python
from sklearn.decomposition import PCA

pca_gen = PCA(100).fit(x_gen)
comp_gen = pca.transform(x_gen)
```


```python
x_gen_long = reg_long.predict(comp_gen)
x_gen_lat = reg_lat.predict(comp_gen)
```


```python
plt.scatter(x_gen_long, x_gen_lat )
plt.title('Original data coordinates of individuals')
```




![alt text](https://raw.githubusercontent.com/thomasgrsp/workaroundmedgan/master/images/gen.png)



```python
plt.scatter(x_long, x_lat)
plt.title('Generated data coordinates of individuals')
```




![alt text](https://raw.githubusercontent.com/thomasgrsp/workaroundmedgan/master/images/data.png)

