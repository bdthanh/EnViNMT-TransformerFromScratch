import streamlit as st
from inference import translate, set_up_necessary_objects

model, src_tokenizer, trg_tokenizer, device, config = set_up_necessary_objects()

st.title('English to Vietnamese Machine Translation Demo')
example_sentence = 'I live near this house , and I thought about how I could make it a nicer space for my\
                    neighborhood, and I also thought about something that changed my life forever.'
input_sentence = st.text_area('Enter an English sentence', max_chars=225, value=example_sentence)
if st.button('Translate'):
    pred_sentence = translate(input_sentence, config, model, src_tokenizer, trg_tokenizer, device)
    st.write('Vietnamese translation:')
    st.write(pred_sentence)
    
  
