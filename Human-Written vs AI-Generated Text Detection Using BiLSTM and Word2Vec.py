#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,LSTM,Bidirectional,Embedding
from keras.callbacks import EarlyStopping,ModelCheckpoint


# In[2]:


df = pd.read_csv(r"C:\Users\boddu\Downloads\AI_Human1.csv")
df


# In[3]:


df['generated'].dtype


# In[4]:


df.isna().sum()


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.duplicated().sum()


# In[8]:


df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print("Duplicate texts:", df.duplicated(subset=["text"]).sum())


# In[9]:


df["text"] = df["text"].astype(str)


# In[10]:


df['generated'] = df['generated'].astype('int32')


# In[11]:


print(df['generated'].dtype)
print(df['generated'].value_counts())


# In[12]:


df.info()


# In[13]:


df['text'][2]


# In[14]:


stop_words = set(stopwords.words('english'))
def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# In[15]:


df['tokens'] = df['text'].apply(text_preprocess)


# In[16]:


df


# In[17]:


from gensim.models import Word2Vec
w2v_model = Word2Vec(window=10,min_count=2,vector_size=100,workers=4,epochs=10)

w2v_model.build_vocab(df["tokens"])


# In[18]:


print(f"Vocabulary size: {len(w2v_model.wv.index_to_key)}")
print(f"Corpus documents: {w2v_model.corpus_count}")
print(f"Epochs: {w2v_model.epochs}")


# In[19]:


w2v_model.train(df["tokens"],total_examples=w2v_model.corpus_count,epochs=w2v_model.epochs)
print(f"Total unique Word2Vec tokens: {len(w2v_model.wv.index_to_key)}")


# In[20]:


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df["tokens"])


# In[21]:


word_index = tokenizer.word_index
print(f"Tokenizer vocabulary size: {len(word_index)}")
print(f"Word2Vec vocabulary size: {len(w2v_model.wv.index_to_key)}")


# In[22]:


w2v_model.wv.similarity('projects','project')


# In[23]:


w2v_model.wv.similarity('student','child')


# In[24]:


w2v_model.wv.most_similar('democracy')


# In[25]:


w2v_model.wv.similar_by_word('phones',topn=5)


# In[26]:


w2v_model.wv.similar_by_key('driving',topn=5)


# In[27]:


embedding_dim = 100

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

covered = 0
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
        covered += 1


# In[28]:


print(f'\nEmbedding coverage: {covered}/{len(word_index)} words = {covered/len(word_index)*100:.1f}%')
print(f'Words with zero vectors (rare words not in W2V): {len(word_index)-covered}')
print(f'Embedding matrix shape: {embedding_matrix.shape}')


# In[29]:


token_lengths = df["tokens"].apply(len)

print(token_lengths.describe())
print("95th percentile:", token_lengths.quantile(0.95))
print("99th percentile:", token_lengths.quantile(0.99))


# In[30]:


MAX_LEN = int(token_lengths.quantile(0.95))

# Prevent unusually small or unnecessarily large sequence lengths
MAX_LEN = max(100, min(MAX_LEN, 500))

print("Selected MAX_LEN:", MAX_LEN)


# In[31]:


sequences = tokenizer.texts_to_sequences(df["tokens"])

X = pad_sequences(sequences,maxlen=MAX_LEN,padding="pre",truncating="post")
y = df['generated'].values


# In[32]:


from sklearn.model_selection import train_test_split

# 80% training, 20% temporary
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)


# In[33]:


print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# In[34]:


print("\nTraining labels:", np.bincount(y_train))
print("Testing labels:", np.bincount(y_test))


# In[35]:


model = Sequential()

model.add(Embedding(input_dim=len(word_index) + 1,output_dim=embedding_dim,weights=[embedding_matrix],
        input_length=MAX_LEN,trainable=True))

model.add(Bidirectional(LSTM(128,dropout=0.3)))

model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.build(input_shape=(None, MAX_LEN))


# In[36]:


model.summary()


# In[37]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[38]:


callbacks = [EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)]
history = model.fit(X_train,y_train,batch_size=64,epochs=15,validation_data=(X_test, y_test),callbacks=callbacks)


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix

y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y_test,y_pred,target_names=["Human Written", "AI Generated"]))


# In[42]:


print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# In[43]:


import json
import os
import pickle

SAVE_DIR = r'D:\AI Generated Text Detection'

os.makedirs(SAVE_DIR, exist_ok=True)

model.save(os.path.join(SAVE_DIR, "ai_generated_text_detector.keras"))

with open(os.path.join(SAVE_DIR, "text_tokenizer.pkl"),"wb") as file:
    pickle.dump(tokenizer, file)

with open(os.path.join(SAVE_DIR, "stop_words.pkl"),"wb") as file:
    pickle.dump(stop_words, file)

config = {"MAX_LEN": MAX_LEN,"embedding_dim": embedding_dim,"threshold": 0.5,"class_labels": {"0": "Human Written","1": "AI Generated"}}

with open(os.path.join(SAVE_DIR, "configuration.json"),"w") as file:
    json.dump(config, file, indent=4)

print("Model and supporting files saved successfully.")


# In[114]:


def predict_text(text,model,tokenizer,stop_words,max_len,threshold=0.5):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in stop_words]

    if not tokens:
        print("No useful tokens remained after preprocessing. \nPlease provide a longer passage.")
        return None

    sequence = tokenizer.texts_to_sequences([tokens])
    padded_sequence = pad_sequences(sequence,maxlen=max_len,padding="pre",truncating="post")

    ai_prob = float(model.predict(padded_sequence, verbose=0)[0][0])

    human_prob = 1 - ai_prob

    if ai_prob >= threshold:
        prediction = "AI Generated"
        confidence = ai_prob
    else:
        prediction = "Human Written"
        confidence = human_prob

    print(f"AI probability: {ai_prob:.4f}")
    print(f"Human probability: {human_prob:.4f}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")

    return {"ai_probability": ai_prob,"human_probability": human_prob,"prediction": prediction,"confidence": confidence}


# In[115]:


test_texts = [
    "Sure, here's my attempt at writing an essay as an average 8th grade student:\n\nFollowing someone else's agreement can have both beneficial and detrimental consequences when UT comes to self growth and development. On one hand, UT can be helpful to learn from others and gain new perspectives. For example, ugh a friend has a different way of thinking about a problem, hearing their perspective can help you see UT UN a new left and come up with a solution that you might not have thought of on your own. Thus, can be especially helpful when working on a group project or when trying to solve a difficult problem.\n\nOn the other hand, following someone else's agreement too closely can limit your own growth and development. When you only consider someone else's perspective, you might muss out on the opportunity to discover your own thoughts and ideas. Thus, can be especially detrimental un subjects Luke science and math, where UT's important to understand the underlying concepts and formulas un order to solve problems. If you're only relying on someone else's explanation, you might not fully understand the material and could struggle to apply UT to real world situations.\n\nOne example of thus us when I was working on a science project with my group. My partner had a perfect idea for the experiment, but I dud't fully understand the concept behind UT. I was too busy trying to follow their instructions that I dud't take the time to thunk about the science behind UT. As a result, I struggled to explain the project to the class and dud't do as well as I could have on the presentation.\n\nIn conclusion, following someone else's agreement can be helpful when UT comes to learning new perspectives, but UT's important to strike a balance between relying on others and trusting your own instincts. It's okay to seek help and guidance from others, but UT's also important to take the time to thunk things through on your own and come up with your own ideas. By funding thus balance, you can ensure that you're growing and developing UN the best way possible.",
    'Public schools are not always the best to attend. That\'s why some schools have other options for kids/ parents that does wait to attend them or does wait there children attending them. There usually\xa0a lot of bad situations that happen i.e. public schools. There a lot of disrespect that come from the students that does care about school. Students skip classes to fight or be disruptive.\n\nThis is why i think Homeschooling or home bound\xa0is a wonderful thing to have. Some kids never area get up eyed come to school so they EED up skipping school. So having this "Distance Certain" stuff is a wonderful thing to have. Even whee kids does area come to school they CAE still large eyed get there education from the comfort of there owe home.\n\nWhee doing home bound, you get assigned to a teacher eyed they pick up your work for you eyed come to your home or where ever else y\'all area meet eyed do the work together i.e. case\xa0the student has any questions. The students eyed teacher would meet at least twice a week so the teacher CAE teach the student an new lesson OE what they are teaching i.e. school. The one bad thing about home bound is that you cast work while OE home bound. Although not being able to work set that much of a problem for students OE home bound.\n\nStudents that get homeschooled by there parents or another adult have a little more freedom the kids OE home bound because they have teachers that check OE them eyed make sure they are doing there work eyed turning it i.e. OE time.\n\nI believe home schooling eyed or home bound is a wonderful thing to have for kids. I feel that we large better whee were alone eyed get to study a subject or wait as long as we need so we know it. IE school units get taught so fast that students that large slower the others does really have a chance to get to know it. IE conclusion I love the fact that we have all these different petunias for everyone.  '
] # this is datasets text 

for index, text in enumerate(test_texts, start=1):
    print(f"\n--- Text {index} ---")
    predict_text(text=text,model=model,tokenizer=tokenizer,stop_words=stop_words,max_len=MAX_LEN,threshold=0.5)


# In[ ]:




