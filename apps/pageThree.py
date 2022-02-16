# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:01:38 2021

@author: JC17642
"""
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from streamlit_metrics import metric, metric_row
import spacy
from spacy import displacy
from spacytextblob.spacytextblob import SpacyTextBlob


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER2 = """<div style="overflow-x: auto; border: 2px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 4.5rem">{}</div>"""

key_1 = ['priority', 'supervisor', 'prioritizing', 'communicate', 'helping', 'grouping', 'extending', 'deadline', 'report', 'liaise', 'continue', 'persist']
key_2 = ['admit', 'rectify', 'prevent', 'source', 'reflect', 'acknowledge', 'assisting', 'honest', 'discussion', 'learn']
key_3 = ['helpful', 'friendly', 'teaching', 'help', 'contribute', 'sharing', 'knowledge', 'communication', 'training', 'learn']
key_4 = ['justify', 'calm', 'opinions', 'understand', 'facts', 'explain', 'suggest', 'honest', 'discussing', 'communicating']
key_5 = ['honest', 'initiate', 'understanding', 'advicing', 'discussion', 'cause', 'assessing', 'listening', 'chance', 'communicating']

kw_lemmas = [key_1, key_2, key_3, key_4, key_5]



@st.cache(allow_output_mutation=True)
def load_model(name):
    model = spacy.load(name)
    model.add_pipe('spacytextblob')
    return model



def get_keywords(kw_lemma, sample, nlp):
    
    # expected answer
    ans = nlp(' '.join(kw_lemma))
    kw_lemma_tup = [(token.lemma_, token.pos_) for token in ans if token.pos_ == "NOUN" or token.pos_ == "VERB"] 
    d1 = {}
    
    kw_ans = [kw[0] for kw in kw_lemma_tup]
    
    d1['text'] = ' '.join(kw_ans)
    d1['ents'] = [{'start': d1['text'].find(kw[0]), 'end': d1['text'].find(kw[0])+len(kw[0]), 'label':kw[1]} for kw in kw_lemma_tup]
    d1['title'] = ""

    expected = [d1]

    #colors = {"VERB": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}

    colors = {"VERB":'#e36262', "NOUN":'#a8f280'}

    options = {"ents": ["VERB", "NOUN"], "colors": colors}
    
    html_exp = displacy.render(expected, style='ent', manual=True, options=options)
    
    
    # answer
    doc = nlp(sample)
    polarity = doc._.polarity     
    d2 = {}
    
    d2['text'] = sample
    d2['ents'] = [{'start':sample.find(token.text), 'end':sample.find(token.text) + len(token.text), 'label':"MATCH"} for token in doc if token.lemma_ in kw_ans]
    d2['title'] = ""
    
    answer = [d2]
    
    colors_ans = {"MATCH":"linear-gradient(90deg, #aa9cfc, #fc9ce7)"}

    options_ans = {"ents": ["MATCH"], "colors": colors_ans}
    
    html_ans = displacy.render(answer, style='ent', manual=True, options=options_ans)

    hit = len(d2['ents'])
    
    html_exp = html_exp.replace("\n", " ")
    html_ans = html_ans.replace("\n", " ")
    
    return HTML_WRAPPER.format(html_exp), html_ans, hit/len(kw_lemma), polarity

def app(sesh):
    
    st.sidebar.markdown('''
             
    Congratulations on completing all stages of AVIA! You may now review your performance in this section.
    This section will have a breakdown for Stage 1 - Video Interview and also Stage 2 - Writing Test Interview.
    Take your time to review these results, gather some insights, and identify area where you may improve.
    ''')
    
    st.sidebar.markdown("## Scores Explanation")
    
    st.sidebar.markdown('''
                        
    Below are the explanation of the scores:
                
    ''')
        
    st.sidebar.markdown("""* **Overall Score:** Average score of video interview and writing test.      """)

    st.sidebar.markdown("""* **Video Interview:** Score of video interview.      """)
    
    st.sidebar.markdown("""* **Writing Test:** Score of writing test.      """)
    
    st.title('Performance Review')
    
    # read in data
    video_df = pd.read_csv('./output/emotion_data.csv')
    video_df['datetime'] = pd.to_datetime(video_df['datetime'], format="%Y-%m-%d %H:%M:%S:%f")
    
    # fer chart
    agg_df = video_df.groupby(['question', 'name']).size().to_frame('count').reset_index()
    agg_df['percentage'] = round(agg_df['count'] / agg_df.groupby('question')['count'].transform('sum'), 2) 
    
    # eye contact chart
    eye_agg_df = video_df.groupby(['question', 'eye_contact']).size().to_frame('count').reset_index()
    eye_agg_df['percentage'] = round(eye_agg_df['count'] / eye_agg_df.groupby('question')['count'].transform('sum'), 2)
    
    # video score
    
    # fer score
    factor = 10/7
    
    emotion_map = {
        'happy':factor*7, 
        'neutral':factor*6, 
        'surprise':factor*5, 
        'sad':factor*4, 
        'angry':factor*3, 
        'disgust':factor*2, 
        'fear':factor*1
     
    }
    
    
    emotion_calc = agg_df[['question', 'name', 'percentage']].copy()
    emotion_calc['name'] = emotion_calc['name'].replace(emotion_map)
    emotion_calc['emotion_score'] = emotion_calc['name']*emotion_calc['percentage']
    emotion_calc = emotion_calc.groupby(['question'])['emotion_score'].sum().to_frame('emotion_score')
    emotion_calc = emotion_calc.reset_index()
    
    # eye score
    eye_factor = 10/3

    eye_map = {
        'center':eye_factor*3, 
        'blink':eye_factor*2, 
        'left':eye_factor*1, 
        'right':eye_factor*1,  
    }
    
    eye_calc = eye_agg_df[['question', 'eye_contact', 'percentage']].copy()
    eye_calc['eye_contact'] = eye_calc['eye_contact'].replace(eye_map)
    eye_calc['eye_contact_score'] = eye_calc['eye_contact']*eye_calc['percentage']
    eye_calc = eye_calc.groupby(['question'])['eye_contact_score'].sum().to_frame('eye_contact_score')
    eye_calc = eye_calc.reset_index()
    
    video_df = emotion_calc.merge(eye_calc, on='question', how='inner').set_index('question')    
    video_df["score"] = video_df.mean(axis=1)
    video_df.reset_index(inplace=True)
    video_score = round(video_df["score"].mean(), 1)
    
    # writing test analysis
    write_df = pd.read_csv('./output/writing_data.csv')
    answers = write_df["answer"].tolist()
    nlp = load_model("en_core_web_sm")
    #nlp.add_pipe('spacytextblob')
    
    writing_results = []
    
    for i, (kw_lemma, ans) in enumerate(zip(kw_lemmas, answers)):
        
        html_exp, html_ans, hit_rate, polarity = get_keywords(kw_lemma, ans, nlp)
        
        writing_results.append([i+1, html_exp, html_ans, hit_rate, polarity])
        
    writing_results_df = pd.DataFrame(writing_results, columns=['question', 'html_exp', 'html_ans', 'hit_rate', 'polarity'])
    writing_results_df = writing_results_df[['question', 'hit_rate', 'polarity']]
    writing_results_df["score"] = 5 + writing_results_df["hit_rate"]*4 + writing_results_df["polarity"]
    #writing_results_df["base_score"] = 5    
    write_score = round(writing_results_df["score"].mean(), 1)
    #writing_results_df.to_csv('writing_results.csv', index=False)
    
    metric("Overall Score", f"{round(np.mean([video_score, write_score]), 1)} / 10")
    
    metric_row(
        {
            "Video Interview": video_score,
            "Writing Test": write_score,
            
        }
    )
    
    fer_chart = alt.Chart(agg_df, title="Facial Expression % by Question").mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
    ).encode(
        x='question:O',
        y=alt.Y('percentage:Q', axis=alt.Axis(format='%')),
        color='name:N',
        tooltip=['name', alt.Tooltip('percentage:Q', format='.1%')],
        ).configure_axis(
    
        labelAngle=0,
        labelFontSize=11,
        titleFontSize=11
    ).configure_title(
        fontSize=14
    )
    
    st.markdown("# **Stage 1 - Video Interview**")
                
    st.markdown("## **Score Breakdown**")
                
    video_bars = alt.Chart(video_df, title="Score by Question").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='question:O',
            y=alt.Y('score:Q', axis=alt.Axis(format='.1f'), scale=alt.Scale(domain=[0, 10])),
            tooltip=['question', alt.Tooltip('score:Q', format='.1f')],
            )
    video_text = video_bars.mark_text(
        baseline='middle',
        dx=0, # Nudges text to right so it doesn't appear on top of the bar
        dy=-8,
        size=15
    ).encode(
        text=alt.Text('score:Q',  format='.1f')
    )
    
    video_score_chart = (video_bars + video_text).configure_title(
            fontSize=14
        ).configure_axis(
            labelAngle=0,
            labelFontSize=14,
            titleFontSize=14)
    
    st.altair_chart(video_score_chart, use_container_width=True)
                
    st.markdown("## **Percentage Breakdown**")

    eye_chart = alt.Chart(eye_agg_df, title="Eye Contact % by Question").mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x='question:O',
        y=alt.Y('percentage:Q', axis=alt.Axis(format='%')),
        color='eye_contact:N',
        tooltip=['eye_contact', alt.Tooltip('percentage:Q', format='.1%')],
        ).configure_axis(
    
        labelAngle=0,
        labelFontSize=11,
        titleFontSize=11
    ).configure_title(
        fontSize=14
    )

    pct_con = st.beta_columns(2)
    
    pct_con[0].altair_chart(fer_chart, use_container_width=True)
    pct_con[1].altair_chart(eye_chart, use_container_width=True)
    
# =============================================================================
#     st.markdown("## **Specific Score Breakdown**")
#     
#     fer_score = alt.Chart(emotion_calc, title="Emotion Score by Question").mark_bar(
#         cornerRadiusTopLeft=3,
#         cornerRadiusTopRight=3
#     ).encode(
#         x='question:O',
#         y=alt.Y('emotion_score:Q', axis=alt.Axis(format='.1f')),
#         tooltip=['question', alt.Tooltip('emotion_score:Q', format='.1f')],
#         ).configure_axis(
#         labelAngle=0,
#         labelFontSize=11,
#         titleFontSize=11
#         
#     ).configure_title(
#         fontSize=14
#     )
#     
#     eye_score = alt.Chart(eye_calc, title="Eye Contact Score by Question").mark_bar(
#     cornerRadiusTopLeft=3,
#     cornerRadiusTopRight=3
#     ).encode(
#         x='question:O',
#         y=alt.Y('eye_contact_score:Q', axis=alt.Axis(format='.1f'),  scale=alt.Scale(domain=[0, 10])),
#         tooltip=['question', alt.Tooltip('eye_contact_score:Q', format='.1f')],
#         ).configure_axis(
#         labelAngle=0,
#         labelFontSize=11,
#         titleFontSize=11
#     ).configure_title(
#         fontSize=14
#     )
#     
#     
#     score_con = st.beta_columns(2)
#     
#     score_con[0].altair_chart(fer_score, use_container_width=True)
#     score_con[1].altair_chart(eye_score, use_container_width=True)    
# =============================================================================

    st.markdown("# **Stage 2 - Writing Test Assessment**")
                
    st.markdown("## **Score Breakdown**")
                
    write_score_bars = alt.Chart(writing_results_df, title="Score by Question").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='question:O',
            y=alt.Y('score:Q', axis=alt.Axis(format='.1f'), scale=alt.Scale(domain=[0, 10])),
            tooltip=['question', alt.Tooltip('score:Q', format='.1f')],
            )
    write_text = write_score_bars.mark_text(
        baseline='middle',
        dx=0, # Nudges text to right so it doesn't appear on top of the bar
        dy=-8,
        size=14
    ).encode(
        text=alt.Text('score:Q',  format='.1f')
    )
    
    write_score_chart = (write_score_bars + write_text).configure_title(
            fontSize=14
        ).configure_axis(
            labelAngle=0,
            labelFontSize=14,
            titleFontSize=14)
    
    st.altair_chart(write_score_chart, use_container_width=True)
    st.markdown("## **Keywords Hit Rate and Sentiment Breakdown**")
    writing_con = st.beta_columns(2)
    
    hit_rate_bars = alt.Chart(writing_results_df, title="Keywords Hit Rate by Question").mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x='question:O',
            y=alt.Y('hit_rate:Q', axis=alt.Axis(format='.1f'), scale=alt.Scale(domain=[0, 1])),
            tooltip=['question', alt.Tooltip('hit_rate:Q', format='.1f')],
            )
    
    hit_rate_text = hit_rate_bars.mark_text(
        baseline='middle',
        dx=0, # Nudges text to right so it doesn't appear on top of the bar
        dy=-8,
        size=14
    ).encode(
        text=alt.Text('hit_rate:Q',  format='.1f')
    )
    
    hit_rate_chart = (hit_rate_bars + hit_rate_text).configure_title(
            fontSize=11
        ).configure_axis(
            labelAngle=0,
            labelFontSize=11,
            titleFontSize=11)
    
    
    
    sentiment_bars = alt.Chart(writing_results_df, title="Sentiment by Question").mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x='question:O',
        y=alt.Y('polarity:Q', axis=alt.Axis(format='.1f')),
        color=alt.condition(
        alt.datum.polarity >= 0,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('red')   # And if it's not true it sets the bar steelblue.
    ),
        tooltip=['question', alt.Tooltip('polarity:Q', format='.1f')],
        )
        
    sentiment_text = sentiment_bars.mark_text(
        align='center',
        baseline='middle',
        dx=0, # Nudges text to right so it doesn't appear on top of the bar
        dy=-8,
        size=14        
    ).encode(
        text=alt.Text('polarity:Q',  format='.1f')
        
    )
    
    sentiment_chart = (sentiment_bars + sentiment_text).configure_title(
            fontSize=11
        ).configure_axis(
            labelAngle=0,
            labelFontSize=11,
            titleFontSize=11)
        
        
        
    writing_con[0].altair_chart(hit_rate_chart, use_container_width=True)
    writing_con[1].altair_chart(sentiment_chart, use_container_width=True)
    

    
        
    for i, html_exp, html_ans, hit_rate, polarity in writing_results:
        
        st.markdown(f"## **Question {i}**")    
        st.markdown("### **Expected Keywords (Lemmatized)**")
        #st.write(kw_lemma)
        st.write(html_exp, unsafe_allow_html=True)
        st.markdown("### **Candidate's Answer**")
        st.write(html_ans, unsafe_allow_html=True)
        st.markdown(f"### **Hit Rate: {round(hit_rate*100, 1)}%  $~~~~~~~~~~~$ Sentiment Polarity: {round(polarity, 3)}**")
        st.markdown("\n")
    
    
    
                
    
                
                

    