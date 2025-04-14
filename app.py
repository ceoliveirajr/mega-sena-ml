
import pandas as pd
import random
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CAMINHO_FIXO = r'C:/Users/ceoliveira/Downloads/mega_sena.xlsx'

st.set_page_config(page_title="Previsão Mega-Sena", layout="centered")
st.title("🎯 Previsão Inteligente da Mega-Sena")

@st.cache_data
def carregar_dados():
    df = pd.read_excel(CAMINHO_FIXO)
    col_dezenas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
    df = df[col_dezenas].astype(int)
    return df

df = carregar_dados()

def preparar_dados(df):
    X, y = [], []
    for i in range(len(df) - 1):
        entrada = df.iloc[i].values
        saida = df.iloc[i + 1].values
        for dezena in saida:
            X.append(entrada)
            y.append(dezena)
    return pd.DataFrame(X, columns=[f'D{i+1}' for i in range(6)]), pd.Series(y)

X, y = preparar_dados(df)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

entrada = df.iloc[-1].values.reshape(1, -1)
probas = modelo.predict_proba(entrada)
todas_dezenas = modelo.classes_
media_probs = probas[0]

top_n = st.slider("Quantidade de dezenas previstas", 6, 20, 12)
dezenas_previstas = sorted(zip(todas_dezenas, media_probs), key=lambda x: x[1], reverse=True)
dezenas_selecionadas = [int(d[0]) for d in dezenas_previstas[:top_n]]

qtd_jogos = st.number_input("Número de jogos a gerar", min_value=1, max_value=100, value=10)
jogos = set()
while len(jogos) < qtd_jogos:
    jogo = tuple(sorted(random.sample(dezenas_selecionadas, 6)))
    jogos.add(jogo)

st.subheader("🧠 Dezenas previstas com maior probabilidade:")
st.write(sorted(dezenas_selecionadas))

st.subheader("🎲 Sugestões de Jogos com Base nas Previsões:")
df_jogos = pd.DataFrame(sorted(jogos), columns=[f'Dezena {i+1}' for i in range(6)])
st.dataframe(df_jogos)

if st.button("⬇️ Baixar Jogos Previstos"):
    df_jogos.to_excel("jogos_previstos.xlsx", index=False)
    with open("jogos_previstos.xlsx", "rb") as f:
        st.download_button("Baixar Excel", f, file_name="jogos_previstos.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
