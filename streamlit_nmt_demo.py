import streamlit as st
import requests

st.title('English to Vietnamese Machine Translation Demo')
example_sentence = 'I live near this house , and I thought about how I could make it a nicer space for my\
                    neighborhood, and I also thought about something that changed my life forever.'
input_sentence = st.text_area('Enter an English sentence', max_chars=225, value=example_sentence)
api_url = "http://localhost:8000/translate/"  

if st.button('Translate'):
    response = requests.post(api_url, json={"input_sentence": input_sentence})
    if response.status_code == 200:
        data = response.json()
        pred_sentence = data.get('translated_sentence', 'No translation found.')
        st.write('Vietnamese translation:')
        st.write(pred_sentence)
    else:
        st.error("Failed to get translation from the API. Status Code: {}".format(response.status_code))
    
  
