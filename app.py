import streamlit as st
from transformers import pipeline

st.title("Named Entity Recognition")

@st.cache_resource
def load_model():
    return pipeline(
        "ner", 
        aggregation_strategy="simple" #combine the tokens that are actually one entity
    )

ner = load_model()

text = st.text_area("Input the text", "Elon Musk discussed NASA's Artemis program with scientists in Houston")

if st.button("Analyze"):
    results = ner(text)

    for entity in results:
        st.write(
            f"{entity['word']} → "
            f"{entity['entity_group']} "
            f"({entity['score']:.2%})"
        )
