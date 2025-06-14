import json
import random
import os
import tkinter as tk
from tkinter import PhotoImage, Canvas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ======= Preparar dados =======
with open('intents.json') as f:
    intents = json.load(f)

frases, tags = [], []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        frases.append(pattern)
        tags.append(intent['tag'])

modelo = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=lambda txt: txt.lower().split())),
    ('classificador', MultinomialNB())
])
modelo.fit(frases, tags)

def prever_intencao(frase):
    return modelo.predict([frase])[0]

def obter_resposta(intencao):
    for intent in intents['intents']:
        if intent['tag'] == intencao:
            return random.choice(intent['responses'])

# ======= GUI com estilo WhatsApp =======
janela = tk.Tk()
janela.title("Chatbot Estilo WhatsApp")
janela.geometry("480x600")
janela.configure(bg="#e5ddd5")
janela.resizable(False, False)

# Cabeçalho
topo = tk.Frame(janela, bg="#075E54", height=60)
topo.pack(fill="x")

# Ícone do bot
if os.path.exists("bot_icon.png"):
    img = PhotoImage(file="bot_icon.png")
    icone = tk.Label(topo, image=img, bg="#075E54")
    icone.place(x=10, y=5)
else:
    icone = tk.Label(topo, text="🤖", font=("Arial", 24), bg="#075E54", fg="white")
    icone.place(x=10, y=10)

tk.Label(topo, text="ChatBot Inteligente", bg="#075E54", fg="white", font=("Arial", 16, "bold")).place(x=70, y=15)

# Canvas com mensagens
canvas = tk.Canvas(janela, bg="#e5ddd5", bd=0, highlightthickness=0)
canvas.place(x=0, y=60, width=480, height=480)

mensagens = tk.Frame(canvas, bg="#e5ddd5")
scrollbar = tk.Scrollbar(janela, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.place(x=460, y=60, height=480)
canvas.create_window((0, 0), window=mensagens, anchor="nw")

mensagens.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Campo de entrada
entrada = tk.Entry(janela, font=("Arial", 12))
entrada.place(x=10, y=550, width=380, height=30)

def mostrar_balao(texto, lado, cor_fundo, cor_texto):
    bal = tk.Frame(mensagens, bg=cor_fundo, padx=10, pady=5)
    tk.Label(bal, text=texto, bg=cor_fundo, fg=cor_texto, font=("Arial", 12), wraplength=300, justify='left').pack()
    if lado == "direita":
        bal.pack(anchor='e', pady=5, padx=10)
    else:
        bal.pack(anchor='w', pady=5, padx=10)

def enviar():
    texto = entrada.get()
    if not texto.strip():
        return
    entrada.delete(0, tk.END)

    mostrar_balao("Tu: " + texto, "direita", "#dcf8c6", "#000")

    intencao = prever_intencao(texto)
    resposta = obter_resposta(intencao)
    mostrar_balao("Bot: " + resposta, "esquerda", "#fff", "#000")

    canvas.update_idletasks()
    canvas.yview_moveto(1.0)

botao = tk.Button(janela, text="Enviar", command=enviar, bg="#25D366", fg="white", font=("Arial", 11, "bold"))
botao.place(x=400, y=550, width=70, height=30)

janela.mainloop()
