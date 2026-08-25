import streamlit as st

# Define the Home page
def Home():
    st.Page('home.py', title='Home')
    st.header('Home Page')

# Define the pages dictionary
pages = {
    "Home":{
        st.Page(Home)
    },
    "Models": {
        st.Page('app/logistic.py', title='Logistic'),
        st.Page('app/svm.py', title='SVM')
    }
}

# Create navigation
pg = st.navigation(pages, position='top')
pg.run()
