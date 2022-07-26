import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
#classes = pickle.load(open('classes.pkl','rb'))
import joblib

def speech_to_text():

    #Importation des libriaires
    import speech_recognition as sr

    r = sr.Recognizer()

    try:
      #Utilisation du microphone par défaut
      with sr.Microphone() as source:
          print('Réglage du buit ambiant, patientez')
        #Délai pour permettre à l'enregistreur de faire ses ajustements
          r.adjust_for_ambient_noise(source, duration=0.3)
          print('Vous pouvez parler')
        #Prise de son
          audio2 = r.listen(source)
        #Utilisation de Google pour reconnaire la piste audio
          texte_STT = r.recognize_google(audio2, language='fr-FR')
    except sr.RequestError as e:
      print("Je n'ai pas pu répondre à votre question; {0}".format(e))
    
    return texte_STT


def clean_up_sentence(sentence):
  import pandas as pd
  import spacy
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  
  df = pd.DataFrame([sentence], columns = ['Patterns'])

  SW = stopwords.words('french')
  nlp = spacy.load('fr_core_news_md')

  #Elimination de la ponctuation (regex) et transformation en minuscule
  df['Patterns'] = df['Patterns'].replace("[^\w\s]", " ", regex = True).str.replace("\d+", '', regex=True).str.lower()
  #Suppression des stopwords
  df['Patterns'] = df['Patterns'].apply(lambda x:' '.join([word for word in x.split() if word not in SW]))
  #Suppression des accents
  df['Patterns'] = df['Patterns'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
  #Tokenization
      
  sentence_words = nltk.word_tokenize(df['Patterns'][0])
  #Lemmatisation
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  
  return sentence_words



# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    
    ress = np.argmax(res)
    decodeur = joblib.load('New_Le.joblib')
    label_decode = decodeur.inverse_transform([ress])
    
    return label_decode


def text_to_speech(label_decode):

    #Importation des librairies
    import gtts
    from playsound import playsound
    import numpy as np 
    import pandas as pd

    #On ouvre le dataframe contenant les tags et les réponses correspondantes
    Responses = pd.read_csv('Responses.csv')
    #On fait correspondre notre tag décodé au tag correspondant dans le dataframe
    R = np.where(Responses['Tags'] == label_decode[0])
    #On associe la réponse correspondante
    R = Responses['Responses'][R[0][0]]
    
    
    ##On passe la réponse R dans google text to speech
    #tts = gtts.gTTS5(R)
    #On enregistre la réponse sous fichier mp3, puis on retourne la réponse sous forme vocale
    #tts.save("label_decode.mp3")
    #speech = playsound("label_decode.mp3")

    return R



def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = text_to_speech(ints)
    return res


msg = speech_to_text()
res = chatbot_response(msg)
print(res)