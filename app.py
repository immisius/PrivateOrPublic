import streamlit as st
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import numpy as np

model_ckpt = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

st.markdown("# 国立？公立？私立？")
st.text_input("大学名", key='input',value='東京大学')

name=st.session_state['input']
model=torch.load('10.pt')
with torch.no_grad():
    print(tokenizer.tokenize(name))
    out=(model(torch.tensor(tokenizer(name)['input_ids']).unsqueeze(0)).logits)
    out[0][1]=0.01*out[0][1]
    out=nn.Softmax(dim=-1)(out)
ans_dict=['国立大学','私立大学','公立大学']
st.markdown("# "+ans_dict[np.argmax(out)])
