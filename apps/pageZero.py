# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:54:34 2021

@author: JC17642
"""

import streamlit as st
from PIL import Image

def app(sesh):
    
    
    st.title('Welcome to AVIA - AI Video Interview Analyzer')
    
    image = Image.open('./img/avia.png')
    
    col1, col2, col3 = st.beta_columns([1,1,1])

    col2.image(image,  width=250)
    

    
    st.markdown('''Greetings! AVIA is a video interview tool that helps you prepare
             for an online interview. By doing this, you can enhance your online interview skills.
             AVIA consists of ***two stages***, the video interview stage and the writing interview stage. 
             The processes of these two stages are further explained below.
    ''')
    
    st.subheader('Stage 1 - Video Interview')
    
    st.write('''
             
    The video interview tests you on your facial expressions, eye contact, and posture while answering
    the questions. This is essential as you might not be conscious of your body language during an online
    interview. Try to be yourself during this process as it will make the performance review more genuine.
    
    
    ''')

    st.subheader('Stage 2 - Writing Interview')
    
    st.write('''
             
    The writing interview is designed to test your English language abilities and 
    also your writing skills. You will be asked various questions in terms of
    scenarios and business decision.
    
    ''')

    st.subheader('Performance Review')
    
    st.write('''
             
    After completing the video and writing interview, you will be able to see your performance on both these stages.
    You may review the analytics odf your performance for each question and the interview as a whole. There will
    be a final single computed score that will decide your overall performance.
    
    
    ''')
    
    st.markdown('### Scroll to Top and Click Next Page. Happy Practising! :sunglasses:')