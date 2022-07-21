import numpy as np
import pandas as pd
import spacy

import numpy as np
import matplotlib.pyplot as plt



nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('chunk_stress_prob.csv', sep = ';')
df[0] = df['0'].apply(lambda x: float(str(x).replace(',','.')))
df[1] = df['1'].apply(lambda x: float(str(x).replace(',','.')))
df = df[['chunk',0,1]]
df = df.dropna()
df['chunk'] = df['chunk'].apply(lambda x: x.lower())


def detect_stress(text):

    doc = nlp(text)
    sents = list(doc.sents)

    scores = []

    for sent in sents:
        print(sent)
        text = [t.lemma_.lower() for t in sent if 
            (t.is_stop == False) &
            (t.is_digit == False) &
            (t.is_punct == False) &
            (t.is_alpha == True) &
            (t.is_currency == False) &
            (t.is_bracket == False)]

        text = pd.DataFrame(text, columns = ['chunk'])
        out = pd.merge(text,df, how = 'left').fillna(0)


        score = {}

        stress_level = out[out[1] != 0][1].mean()

        if out[0].sum() == 0:
            score['stress_probability'] = 0
        else:
            score['stress_probability'] = np.mean([(1 - len(out[out[0] == 0])/len(out)),stress_level])
        
        if score['stress_probability'] > 0.5:
            score['stress_level'] = stress_level
        else:
            score['stress_level'] = 0

        scores.append(score)

    print(score)
    return scores

def color_map(x):
    if x < 0.35:
        return 'green'
    if (x > 0.35) & (x < 0.65):
        return 'orange'
    if x > 0.65:
        return 'red'

# Import dependencies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model():
    
    nlp = spacy.load('en_core_web_sm')
#    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    return nlp


@st.cache()
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


DEFAULT_TEXT = "I am feeling stressed today."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; 
                margin-bottom: 2.5rem">{}</div>"""


# Set header title
st.title('Myndoor Demo')

from PIL import Image
img= Image.open("myndoor_logo.png")
st.image(img, width=400)

# Description
#st.sidebar.title("")


# Text_Area
text = st.text_area("Please, paste your text below!", DEFAULT_TEXT)

stress = detect_stress(text)
stress = pd.DataFrame(stress)#.transpose().reset_index()

stress.columns = ['Stress Probability','Stress Level']
stress = stress.transpose()
stress.columns = ['value']
stress = stress.reset_index()

stress['colors'] = stress.apply(lambda x: color_map(x['value']), axis = 1)

fig,ax = plt.subplots(figsize = (6,4))

#sns.set_theme(style="white")

plt.bar(x = stress["index"], height=stress["value"], color = stress['colors'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


for i, v in enumerate(stress['value']):
    ax.text(i - 0.07, v + 0.03, str(round(v,2)), fontname='monospace', fontsize = 15)
    
plt.xticks([0,1], stress["index"], fontname='monospace', fontsize = 15)

st.pyplot(fig)