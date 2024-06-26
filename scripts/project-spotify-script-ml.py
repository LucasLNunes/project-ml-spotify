import requests
import base64
import pandas as pd
import time
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
# Configurações do Spotify
client_id = '#Necessário Criar o Aplicativo do Spotify'
client_secret = ' #Após Criar solicitar pegar na URL essa informação'
redirect_uri = 'https://myfirstspotfyappforstudy.com/callback'
scopes = 'user-read-private user-read-email playlist-read-private playlist-read-collaborative user-read-recently-played user-top-read user-library-read'
# URL de autorização
auth_url = f"https://accounts.spotify.com/authorize?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&scope={scopes.replace(' ', '%20')}"
print(f"Vá para o seguinte URL para autorizar o acesso: {auth_url}")
# Código de autorização obtido após o usuário autorizar
authorization_code = input("Insira o código de autorização aqui: ")
# Trocar o código de autorização por um token de acesso
token_url = "https://accounts.spotify.com/api/token"
headers = {
    'Authorization': 'Basic ' + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
    'Content-Type': 'application/x-www-form-urlencoded'
}
data = {
    'grant_type': 'authorization_code',
    'code': authorization_code,
    'redirect_uri': redirect_uri
}

response = requests.post(token_url, headers=headers, data=data)
token_info = response.json()
# Verificar a resposta completa
print("Resposta da solicitação para obter o token:")
print(token_info)
# Verificar se o token foi obtido com sucesso
if 'access_token' in token_info:
    access_token = token_info['access_token']
    print("Token de acesso obtido com sucesso!")
    print(f"Access Token: {access_token}")
else:
    print("Erro ao obter o token de acesso:")
    print(token_info)
# Função para obter o perfil do usuário
def get_user_profile(access_token):
    url = 'https://api.spotify.com/v1/me'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    return response.json()
# Função para obter as playlists do usuário
def get_user_playlists(access_token):
    url = 'https://api.spotify.com/v1/me/playlists'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    return response.json()

# Obter playlists do usuário
playlists_data = get_user_playlists(access_token)
# Função para converter os dados das playlists em um DataFrame
def playlists_to_dataframe(playlists_data):
    playlists = []
    for item in playlists_data['items']:
        playlist_info = {
            'playlist_name': item['name'],
            'playlist_id': item['id'],
            'owner_name': item['owner']['display_name'],
            'total_tracks': item['tracks']['total'],
            'playlist_uri': item['uri']
        }
        playlists.append(playlist_info)
    df = pd.DataFrame(playlists)
    return df

# Converter as playlists em um DataFrame
df_playlists = playlists_to_dataframe(playlists_data)

# Exibir o DataFrame das playlists
df_playlists
# Função para obter todas as faixas de uma playlist com paginação
def get_all_playlist_tracks(access_token, playlist_id):
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    all_tracks = []
    offset = 0
    limit = 100  # Você pode ajustar o limite conforme necessário

    while True:
        params = {
            'offset': offset,
            'limit': limit
        }
        response = requests.get(url, headers=headers, params=params)
        try:
            response_json = response.json()
            if 'items' in response_json:
                all_tracks.extend(response_json['items'])
                if len(response_json['items']) < limit:
                    break
                offset += limit
            else:
                break
        except ValueError:
            print(f"Erro ao decodificar JSON para a playlist {playlist_id}. Resposta: {response.text}")
            break

    return all_tracks
# Função para converter as faixas em um DataFrame
def tracks_to_dataframe(tracks_data):
    tracks = []
    for item in tracks_data:
        track = item['track']
        track_info = {
            'track_name': track['name'],
            'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
            'album_name': track['album']['name'],
            'track_duration_ms': track['duration_ms'],
            'track_popularity': track['popularity'],
            'track_uri': track['uri']
        }
        tracks.append(track_info)
    df = pd.DataFrame(tracks)
    return df

# Iterar sobre o DataFrame das playlists e obter faixas de cada uma
all_tracks_combined = []

for index, row in df_playlists.iterrows():
    playlist_id = row['playlist_id']
    print(f"Obtendo músicas da playlist: {row['playlist_name']} (ID: {playlist_id})")
    
    tracks_data = get_all_playlist_tracks(access_token, playlist_id)
    df_tracks = tracks_to_dataframe(tracks_data)
    
    # Exibir as faixas da playlist atual
    print(f"Faixas da playlist {row['playlist_name']}:")
    print(df_tracks)
    
    all_tracks_combined.append(df_tracks)

# Combinar todos os DataFrames de faixas em um único DataFrame
if all_tracks_combined:
    df_all_tracks = pd.concat(all_tracks_combined, ignore_index=True)
    print("Todas as faixas:")
    print(df_all_tracks)
else:
    print("Nenhuma faixa encontrada nas playlists do usuário.")
df_all_tracks['track_id'] = df_all_tracks['track_uri'].apply(lambda x: x.split(':')[-1])
# Função para buscar características de uma música usando seu track_id
def get_track_features(track_id, access_token):
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    return response.json()
# Função para buscar características de músicas em lote
def get_track_features_batch(track_ids, token):
    url = "https://api.spotify.com/v1/audio-features"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(track_ids)
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()["audio_features"]
# / Lista para armazenar as características das músicas
track_features_list = []

# Iterar sobre os track_ids em lotes
batch_size = 100  # Tamanho do lote pode ser ajustado conforme necessário
for i in range(0, len(df_all_tracks["track_id"]), batch_size):
    batch_ids = df_all_tracks["track_id"][i:i + batch_size].tolist()
    batch_features = get_track_features_batch(batch_ids, access_token)
    track_features_list.extend(batch_features)
    time.sleep(1)  # Pequeno atraso para evitar limites de taxa

    # Imprimir o progresso a cada lote de 100 registros
    print(f"Processados {i + len(batch_ids)} de {len(df_all_tracks['track_id'])} músicas")


# Converter a lista de características em um DataFrame
df_track_features = pd.DataFrame(track_features_list)

# Combinar o DataFrame original com o DataFrame de características
df_combined = pd.merge(df_all_tracks, df_track_features, left_on="track_id", right_on="id")

# Exibir as primeiras linhas do DataFrame combinado
print(df_combined.head())
# Definir o cache dos IDs dos artistas
artist_id_cache = {}
# Definir o cache dos IDs dos artistas
artist_id_cache = {}

# Função para buscar o ID de um artista pelo nome
def get_artist_id(artist_name, access_token, cache):
    if artist_name in cache:
        return cache[artist_name]
    
    if not artist_name.strip():  # Verificar se o nome do artista está vazio
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": artist_name,
        "type": "artist",
        "limit": 1
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()
    if results['artists']['items']:
        artist_id = results['artists']['items'][0]['id']
        cache[artist_name] = artist_id
        return artist_id
    else:
        return None
# Função para buscar informações de artistas em lote
def get_artists_info(artist_ids, access_token):
    url = "https://api.spotify.com/v1/artists"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "ids": ",".join(artist_ids)
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["artists"]
# Extraindo o nome do primeiro artista de cada música
df_combined['first_artist_name'] = df_combined['artist_name'].apply(lambda x: x.split(",")[0])
df_combined
# Obter IDs dos artistas com barra de progresso
artist_ids = []
for name in tqdm(df_combined['first_artist_name'], desc="Obtendo IDs dos artistas"):
    artist_id = get_artist_id(name, access_token, artist_id_cache)
    artist_ids.append(artist_id)

df_combined['artist_id'] = artist_ids
# Lista para armazenar os gêneros musicais
genres_dict = {}

# Iterar sobre os artist_ids em lotes
batch_size = 50  # Tamanho do lote pode ser ajustado conforme necessário
artist_ids = df_combined['artist_id'].dropna().unique().tolist()

# Função para adicionar backoff exponencial
def fetch_with_backoff(batch_ids, access_token, retries=5):
    delay = 1
    for _ in range(retries):
        try:
            return get_artists_info(batch_ids, access_token)
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                print(f"Limite de requisições atingido. Aguardando {delay} segundos...")
                time.sleep(delay)
                delay *= 2  # Aumenta o delay exponencialmente
            else:
                raise e
    raise Exception("Número máximo de tentativas excedido")

# Iterar sobre os artist_ids em lotes com barra de progresso
for i in tqdm(range(0, len(artist_ids), batch_size), desc="Buscando informações dos artistas"):
    batch_ids = artist_ids[i:i + batch_size]
    try:
        batch_artists_info = fetch_with_backoff(batch_ids, access_token)
        for artist_info in batch_artists_info:
            genres_dict[artist_info['id']] = ", ".join(artist_info.get('genres', []))
    except Exception as e:
        print(f"Erro ao buscar informações dos artistas: {e}")
    time.sleep(5)  # Atraso fixo de 5 segundos entre lotes para evitar limites de taxa

# Adicionar os gêneros ao DataFrame
df_combined['genres'] = df_combined['artist_id'].map(genres_dict)
# Função para obter novos lançamentos com paginação
def get_new_releases(access_token, country='BR', limit=50, max_results=500):
    url = "https://api.spotify.com/v1/browse/new-releases"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    all_releases = []
    offset = 0
    
    while len(all_releases) < max_results:
        params = {
            "country": country,
            "limit": limit,
            "offset": offset
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        new_releases = response.json()
        all_releases.extend(new_releases['albums']['items'])
        if len(new_releases['albums']['items']) < limit:
            break
        offset += limit
    
    return all_releases
# Função para obter as faixas de um álbum
def get_album_tracks(album_id, access_token):
    url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
# Função para filtrar as faixas dos novos lançamentos
def filter_new_tracks(new_releases, access_token):
    tracks = []
    for album in tqdm(new_releases, desc="Processando álbuns"):
        album_info = {
            'album_name': album['name'],
            'album_id': album['id'],
            'artist_name': album['artists'][0]['name'],
            'release_date': album['release_date'],
            'total_tracks': album['total_tracks'],
            'album_uri': album['uri']
        }
        album_tracks = get_album_tracks(album['id'], access_token)
        for track in album_tracks['items']:
            track_info = {
                'track_name': track['name'],
                'track_id': track['id'],
                'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                'album_name': album_info['album_name'],
                'release_date': album_info['release_date'],
                'track_uri': track['uri']
            }
            tracks.append(track_info)
    return pd.DataFrame(tracks)
# Obter novos lançamentos para o Brasil e Estados Unidos
new_releases_br = get_new_releases(access_token, country='BR')
new_releases_us = get_new_releases(access_token, country='US')

# Filtrar as faixas dos novos lançamentos
df_brazilian_tracks = filter_new_tracks(new_releases_br, access_token)
df_american_tracks = filter_new_tracks(new_releases_us, access_token)

# Combinar os DataFrames das faixas brasileiras e americanas
df_combined_tracks = pd.concat([df_brazilian_tracks, df_american_tracks], ignore_index=True)

# Exibir o DataFrame combinado
print(df_combined_tracks.head())
# / Lista para armazenar as características das músicas
track_features_list = []

# Iterar sobre os track_ids em lotes
batch_size = 100  # Tamanho do lote pode ser ajustado conforme necessário
for i in range(0, len(df_combined_tracks["track_id"]), batch_size):
    batch_ids = df_combined_tracks["track_id"][i:i + batch_size].tolist()
    batch_features = get_track_features_batch(batch_ids, access_token)
    track_features_list.extend(batch_features)
    time.sleep(1)  # Pequeno atraso para evitar limites de taxa

    # Imprimir o progresso a cada lote de 100 registros
    print(f"Processados {i + len(batch_ids)} de {len(df_combined_tracks['track_id'])} músicas")


# Converter a lista de características em um DataFrame
df_track_features = pd.DataFrame(track_features_list)

# Combinar o DataFrame original com o DataFrame de características
df_combined_tracks = pd.merge(df_combined_tracks, df_track_features, left_on="track_id", right_on="id")
# Extraindo o nome do primeiro artista de cada música
df_combined_tracks['first_artist_name'] = df_combined_tracks['artist_name'].apply(lambda x: x.split(",")[0])
df_combined_tracks
# Obter IDs dos artistas com barra de progresso
artist_ids = []
for name in tqdm(df_combined_tracks['first_artist_name'], desc="Obtendo IDs dos artistas"):
    artist_id = get_artist_id(name, access_token, artist_id_cache)
    artist_ids.append(artist_id)

df_combined_tracks['artist_id'] = artist_ids
# Lista para armazenar os gêneros musicais
genres_dict = {}

# Iterar sobre os artist_ids em lotes
batch_size = 50  # Tamanho do lote pode ser ajustado conforme necessário
artist_ids = df_combined_tracks['artist_id'].dropna().unique().tolist()

# Função para adicionar backoff exponencial
def fetch_with_backoff(batch_ids, access_token, retries=5):
    delay = 1
    for _ in range(retries):
        try:
            return get_artists_info(batch_ids, access_token)
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                print(f"Limite de requisições atingido. Aguardando {delay} segundos...")
                time.sleep(delay)
                delay *= 2  # Aumenta o delay exponencialmente
            else:
                raise e
    raise Exception("Número máximo de tentativas excedido")

# Iterar sobre os artist_ids em lotes com barra de progresso
for i in tqdm(range(0, len(artist_ids), batch_size), desc="Buscando informações dos artistas"):
    batch_ids = artist_ids[i:i + batch_size]
    try:
        batch_artists_info = fetch_with_backoff(batch_ids, access_token)
        for artist_info in batch_artists_info:
            genres_dict[artist_info['id']] = ", ".join(artist_info.get('genres', []))
    except Exception as e:
        print(f"Erro ao buscar informações dos artistas: {e}")
    time.sleep(5)  # Atraso fixo de 5 segundos entre lotes para evitar limites de taxa

# Adicionar os gêneros ao DataFrame
df_combined_tracks['genres'] = df_combined_tracks['artist_id'].map(genres_dict)
# Excluir as colunas desnecessárias
columns_to_drop = ['track_uri', 'track_id', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'artist_id']
df_combined = df_combined.drop(columns=columns_to_drop)
# Dummyficar a coluna 'genres'
# Primeiro, precisamos garantir que a coluna 'genres' não contenha valores NaN
df_combined['genres'] = df_combined['genres'].fillna('')
# Modificando a coluna 'genres' para manter apenas o primeiro texto antes da vírgula
df_combined['genres'] = df_combined['genres'].apply(lambda x: x.split(',')[0].strip())
# Seleciona todas as colunas numéricas, excluindo 'track_duration_ms', 'track_popularity' e 'key'
numeric_columns = df_combined.select_dtypes(include=['number']).columns
irrelevant_columns = ['track_duration_ms', 'track_popularity', 'key', 'duration_ms', 'tempo']
numeric_columns = numeric_columns.difference(irrelevant_columns)

# Cria o DataFrame apenas com as colunas numéricas relevantes
X = df_combined[numeric_columns]

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Testar diferentes valores de k usando o método do cotovelo
sse = []
k_range = range(1, 50)  # Ajuste o intervalo de k conforme necessário
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot do método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bx-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Soma dos Quadrados das Distâncias (SSE)')
plt.title('Método do Cotovelo para Determinar o Número de Clusters')
plt.show()
# Escolher o número de clusters baseado no gráfico do cotovelo
k = 10  # Valor escolhido baseado no gráfico
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar a coluna de clusters ao DataFrame original
df_combined['cluster'] = clusters
# Análise dos clusters apenas com colunas numéricas relevantes
cluster_analysis = df_combined.groupby('cluster')[numeric_columns].mean()
cluster_analysis
# Identificar o cluster mais frequente nas músicas escutadas pelo usuário
most_common_cluster = df_combined['cluster'].mode()[0]
print(f"O cluster mais atrativo para o usuário é: {most_common_cluster}")
# Analisar as características do cluster mais atrativo
attractive_cluster_features = cluster_analysis.loc[most_common_cluster]
attractive_cluster_features
# Calcular métricas de avaliação
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg}")

inertia = kmeans.inertia_
print(f"Inertia: {inertia}")
df_combined_tracks = df_combined_tracks.drop_duplicates()
# Seleciona todas as colunas numéricas, excluindo 'track_duration_ms', 'track_popularity' e 'key'
numeric_columns_tracks = df_combined_tracks.select_dtypes(include=['number']).columns
numeric_columns_tracks = numeric_columns_tracks.difference(irrelevant_columns)
# Padronizando os dados das novas músicas
X_tracks = df_combined_tracks[numeric_columns_tracks]
X_tracks_scaled = scaler.transform(X_tracks)
# Predizer os clusters das novas músicas
clusters_tracks = kmeans.predict(X_tracks_scaled)
# Adicionar a coluna de clusters ao novo DataFrame
df_combined_tracks['cluster'] = clusters_tracks
# Filtrar músicas do cluster mais atrativo
recommended_songs = df_combined_tracks[df_combined_tracks['cluster'] == most_common_cluster]
# Selecionar algumas colunas relevantes para exibir
recommended_songs = recommended_songs[['track_name', 'artist_name', 'album_name']]
# Ordenar por popularidade e selecionar as top 20 músicas
top_10_songs = recommended_songs.sort_values(by='track_name', ascending=False).head(10)
top_10_songs
df_combined.columns
# Análise dos clusters apenas com colunas numéricas relevantes
cluster_analysis = df_combined.groupby('cluster')[numeric_columns].mean()
cluster_analysis