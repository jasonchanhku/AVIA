import SessionState
import streamlit as st
import io
from apps import pageZero, pageOne, pageTwo, pageThree, pageFour

st.set_page_config(
    page_icon=":robot_face:",
    page_title="AVIA - AI Video Interview Analyzer",
    layout="centered",
    initial_sidebar_state="expanded"
)

sesh = SessionState.get(curr_page = 0)
PAGES = [pageZero.app, pageOne.app, pageTwo.app, pageThree.app]

def main():
    ####SIDEBAR STUFF

    st.sidebar.title("Instructions Panel")

    st.sidebar.markdown('''This sidepar panel has instructions for the application.
                        Just follow them accordingly to complete the interview process.
                        
    
    
    
                        ''')
    
    #####MAIN PAGE APP:
    #st.write('PAGE NUMBER:', sesh.curr_page)
    
    #st.write(sesh.curr_page)

    
    #####MAIN PAGE NAV BAR:
    st.markdown("<h3 style='text-align: center;'>Navigation</h3>", unsafe_allow_html=True)
    #st.markdown('Click Next to go to the next page')
    
    a,b,c,d, e = st.beta_columns([1,1,1,1,1])
    #print(sesh.curr_page)
    if a.button('<- Back Page'):
        sesh.curr_page = max(0, sesh.curr_page-1)
    if e.button('Next Page ->'):
    #    print('pre_inc', sesh.curr_page)
        sesh.curr_page = min(len(PAGES)-1, sesh.curr_page+1)
    #    print('next: ', sesh.curr_page)
    
    st.markdown('--------------------------------')
    
    page_turning_function = PAGES[sesh.curr_page]
    page_turning_function(sesh)
    


if __name__=='__main__':
    main()