# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:58:20 2021

@author: JC17642
"""

import streamlit as st
import pandas as pd

questions = [
        
        "You are asked to finish a large number of tasks within a day. How would you manage that?",
        "Everyone makes mistakes. How would you rectify the situation if you fail to meet the expectation?",
        "When there is an expansion of your team and many new teammates join in. How would you adjust for this?",
        "If you are 100% sure that your boss’s idea is wrong, what would you do?",
        "Assume you are the boss and one of your employees have been severely underperforming, what will you do?"
        ]

def app(sesh):
    

    st.title('Stage 2 - Writing Assessment')
    st.write('''The writing interview is designed to test your English language abilities and also your writing skills. You will be asked various questions in terms of scenarios and business decision.
             Please kindly follow the instructions on the instructions panel on the sidebar.''')

    st.sidebar.markdown("## Step 1")
    
    st.sidebar.markdown('''Read the questions carefully and answer them with the best of your abiltiies. Answer all questions. Please make sure you have at ***least 3 sentences of answer for each question.*** Do expand the text box shall you require larger space to write''')

    st.sidebar.markdown("## Step 2")
    
    st.sidebar.markdown('''After you finised writing your answers for all questions, and ***press submit.***''')
    
    st.sidebar.markdown("## Step 3")
    
    st.sidebar.markdown('''Scroll back to the top and click on next page to move on to the performance review section.''')
    
    form = st.form(key='my_form')
    
    q1val = "I will group them according to their level of priority and prioritize them. If the task is urgent and critical, I will finish it first. For the complicated task, I will break down the task into several pieces and finish it step by step. If I know I cannot finish it within a day, I will communicate with my supervisor to see if the deadline can be extended or if some colleagues can give help."
    
    q2val = "Once I know I made a mistake, I will admit my mistake to my supervisor and see if there are any bad influence on others. Then I rectify my mistakes as soon as possible. To prevent making the same type of mistake again, I will self reflect on what caused the mistake and find the source of the mistake and learn from it and grow."    
    
    q3val = "I will be friendly to them and give a helpful hand whenever they need it. Then, I will see if my supervisor has any training plan for the new teammates in which I can contribute my experience and knowledge. Clear division of labor and communication is needed since the team is getting larger."
    
    q4val = "I will step up, voice out, and justify to my boss that this is not a good idea. In a calm manner I shall explain the wrong things in his idea at the same time with facts and justifications. I will suggest that he gather more ideas and opinions from other teammates. It is important to be direct but in a manner such that he does not get offended and understands my point."
    
    q5val = "I will sit down with him or her and have an honest conversation. A will initiate a discussion on what is causing the underperformance and why is it happening. I will not judge a book by its cover as they may be some very sensitive things are he or she is facing personally. After understanding the situation, I will assess and provide advice to that employee on how to get back on track."
    
    ans1 = form.text_area(f"Question 1: {questions[0]}", value=q1val, height=150 ,key='q1')
            
    ans2 = form.text_area(f"Question 2: {questions[1]}", value=q2val, height=150, key='q2')
        
    ans3 = form.text_area(f"Question 3: {questions[2]}", value=q3val, height=150, key='q3')
        
    ans4 = form.text_area(f"Question 4: {questions[3]}", value=q4val, height=150, key='q4')
        
    ans5 = form.text_area(f"Question 5: {questions[4]}", value=q5val,height=150, key='q5')
    
    submit_button = form.form_submit_button(label='Submit')
    
    if submit_button:
        st.success("Writing interview successfully submitted, you may move on to the performance review. Please scroll up and click next page.")
        answer_df = pd.DataFrame(
            
            {
            
            "question_no" : [1,2,3,4,5],
            "question": questions,
            "answer": [ans1, ans2, ans3, ans4, ans5]
            }
            
            
            )
    
        answer_df.to_csv('./output/writing_data.csv', index=False)
        
    else:
        st.warning("Answers not submitted yet, please complete and press submit.")
    

    