import streamlit as st 
import pandas as pd 
import plotly.express as px 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline Satisfaction Prediction Project", page_icon="🛫")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Training Models & Their Evaluations", "Make Your Own Predictions!"])

# Load dataset
#cleaned train dataset
df = pd.read_csv('data/clean_train.csv')
# Original train dataset used for EDA for label purposes
odf = pd.read_csv('data/train.csv')
# Modifying original train dataset to drop 2 columns and reassigning it to the name 'odf'
odf = odf.drop(columns = ['Unnamed: 0', 'id'])


# Home Page
if page == "Home":
    st.title("Predicting Airline Passenger Satisfaction Using Classification Machine Learning")
    st.subheader("By Ian Glish")
    st.write("""
        This Streamlit app provides an interactive platform to explore the airline satisfaction dataset, sourced from Kaggle. This app provides data overview information, exploratory data analysis, training 3 varities of classification machine learning models with their accuracy evaluations and a Random Forest classification machine learning model that helps predict the satisfaction of a airline passenger based on 22 sources of user input via sliding scales. 
    """)
    st.image('https://apex.aero/wp-content/uploads/2020/09/Screen-Shot-2019-12-18-at-11.29.30-e1576668292951.png')
    st.write("""
    Please be aware that some aspects of this app might be slow due to dataset size!
    """)


# Data Overview
elif page == "Data Overview":
    st.title("Data Overview")

    st.subheader("About the Data")
    st.write("""
      The airline satisfaction dataset includes data from over 100,000 passenger surveys, where passengers rated their experiences across 14 preflight and inflight aspects. It also contains categorical and numeric details about the passengers and their specific flight information, concluding with whether they were 'satisfied' or 'neutral/dissatisfied' with the overall experience.
    """)
    st.write("Kaggle dataset: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data.")
    st.subheader("Data Dictionary")
    st.markdown("""
    | Feature   | Explination|
    | ----------- | ------------ |  
    | *Gender* | Gender of airpline passenger - male/female |        
    | *Customer Type* | The customer type - loyal/disloyal | 
    | *Age* | Age of the customer |      
    |*Type of Travel*| Purpose of the flight of passenger - business/personal|          
    |*Class*| Travel class of passenger - eco/eco plus/business|           
    |*Flight Distance*| The flight distance of trip in miles|          
    |*Inflight WiFi Service*| Satisfaction rating of inflight Wi-Fi service - 0:Not Applicable, Rating: 1-5|           
    |*Departure/Arrival Time Convenience*| Convenience of departure/arrival time - 0:Not Applicable, Rating: 1-5|          
    |*Ease of Online Booking*| Satisfaction rating of booking online - 0:Not Applicable, Rating: 1-5|            
    |*Gate Location*| Satisfaction rating of gate location - 0:Not Applicable, Rating: 1-5|            
    |*Food and Drink*| Satisfaction rating of inflight food and drink - 0:Not Applicable, Rating: 1-5|         
    |*Online Boarding*| Satisfaction rating of  online boarding process - 0:Not Applicable, Rating: 1-5|          
    |*Seat Comfort*| Satisfaction rating of comfort of seat - 0:Not Applicable, Rating: 1-5|          
    |*Inflight Entertainment*| Satisfaction rating of inflight entertainment - 0:Not Applicable, Rating: 1-5|     
    |*On-board Service*| Satisfaction rating of on-board service - 0:Not Applicable, Rating: 1-5|          
    |*Leg Room Service*| Satisfaction rating of  legroom - 0:Not Applicable, Rating: 1-5|          
    |*Baggage Handling*| Satisfaction rating of  baggage handling - 0:Not Applicable, Rating: 1-5|         
    |*Check-in Service*| Satisfaction rating of check-in service - 0:Not Applicable, Rating: 1-5|      
    |*Inflight Service*| Satisfaction rating of inflight service - 0:Not Applicable, Rating: 1-5|         
    |*Cleanliness*|  Satisfaction rating of cleanliness - 0:Not Applicable, Rating: 1-5|         
    |*Departure Delay in Minutes*| Minutes of departure delay|         
    |*Arrival Delay in Minutes*| Minutes of arrival delay|        
    |*Satisfaction*| Customer satisfaction - satisfied/neutral or dissatisfied|
    """)

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis\nUsing Plotly Visualizations")


    container = st.container(border=True)
    container.subheader("Select the type of visualization you'd like to explore:")
    eda_type = container.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = odf.select_dtypes(include='object').columns.tolist()
    num_cols = odf.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numeric variable for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating"):
                st.plotly_chart(px.histogram(odf, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(odf, x=h_selected_col, title=chart_title))
                

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numeric variable for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating"):
                st.plotly_chart(px.box(odf, x=b_selected_col, color='satisfaction', title=chart_title))
            else:
                st.plotly_chart(px.box(odf, x=b_selected_col, title=chart_title))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating"):
                st.plotly_chart(px.scatter(odf, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))
            else:
                st.plotly_chart(px.scatter(odf, x=selected_col_x, y=selected_col_y, title=chart_title))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical feature:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            if st.checkbox("Show by Overall Satisfaction Rating"):
                st.plotly_chart(px.histogram(odf, x=selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(odf, x=selected_col, title=chart_title))
        


# Model Training and Evaluation Page
elif page == "Training Models & Their Evaluations":
    st.title("Classification Model Types & Trainings with Performance Evaluations")
    st.subheader("Choose a classification model type to train on the dataset, to see its accuracy scores & corresponding confusion matrix")
    st.write("For any model type, it needs to have a better accuracy score than 56.67%!")

    # Sidebar for model selections
    st.sidebar.subheader("Choose a Classification Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a Model Type", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select The Number of Neighbors (K)", min_value=1, max_value=21, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Random Forest":
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    container = st.container(border=True)
    container.write(f" **Model Selected: {model_option}**")
    container.write(f" **Training Accuracy: {model.score(X_train_scaled, y_train)*100:.2f}%**")
    container.write(f" **Test Accuracy: {model.score(X_test_scaled, y_test)*100:.2f}%**")

    # Display confusion matrix
    st.subheader(f"Confusion Matrix for {model_option} Model")
    fig, ax = plt.subplots()
    if model_option == "K-Nearest Neighbors":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'satisfied' customers as 'neutral or dissatisfied' (false negatives) than misclassifying 'neutral or dissatisfied' customers as 'satisfied' (false positives).")
        st.write(f"This suggests the {model_option} model has difficulty identifying satisfied customers.")
        st.write(f"**Out of all the model types, {model_option} had the second highest overall accuracy scores, regaurdless of what the selected K value was.**")
    elif model_option == "Logistic Regression":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Greens')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'satisfied' customers as 'neutral or dissatisfied' (false negatives) than misclassifying 'neutral or dissatisfied' customers as 'satisfied' (false positives).")
        st.write(f"This suggests the {model_option} model has difficulty identifying satisfied customers.")
        st.write(f"**Out of all the model types, {model_option} had the lowest overall accuracy scores.**")
    elif model_option == "Random Forest":
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Oranges')
        st.pyplot(fig)
        st.write(f"Using the {model_option} model, the confusion matrix shows a higher rate of misclassifying 'satisfied' customers as 'neutral or dissatisfied' (false negatives) than misclassifying 'neutral or dissatisfied' customers as 'satisfied' (false positives).")
        st.write(f"This suggests the {model_option} model has difficulty identifying satisfied customers.")
        st.write(f"**Out of all the model types, {model_option} had the highest overall accuracy scores, making it the best model to use for the 'Make Your Own Predictions!' page.**")
    # Make Predictions Page
elif page == "Make Your Own Predictions!":
    st.title("Make Your Own Airline Passenger Satisfaction Prediction")
    container = st.container(border=True)
    container.subheader("Use 22 features to input in a Random Forest classification model")
    container.subheader("**Adjust the feature scale values below to make your own predictions on whether an airpline passenger will be satisfied or not**")
    

    # User inputs for prediction
    gender = st.slider("Gender ---> Female: 0, Male: 1", min_value=0, max_value=1, value=1)
    customer_type = st.slider("Customer Type ---> Disloyal Customer: 0, Loyal Customer: 1", min_value=0, max_value=1, value=1)
    age = st.slider("Age --> Pick An Age From 1 To 90", min_value=1, max_value=90, value=50)
    travel_type = st.slider("Type of Travel ---> Personal Travel: 0 , Business Travel: 1", min_value=0, max_value=1, value=1)
    travel_class = st.slider("Travel Class Types ---> Eco: 1, Eco Plus: 2, Business: 3", min_value=1, max_value=3, value=2)
    flight_distance = st.slider("Flight Distance ---> Pick A Distance Between 1 and 5000 Miles", min_value=1, max_value=5000, value= 2500)
    inflight_wifi = st.slider("Inflight WiFi Service Rating ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    time_convenient = st.slider("Departure/Arrival Time Convenience ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    online_booking = st.slider("Ease of Online Booking ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    gate_location = st.slider("Gate Location ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=0)
    food_drink = st.slider("Food and Drink ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=2)
    online_boarding = st.slider("Online Boarding ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat Comfort ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=1)
    entertainment = st.slider("Inflight Entertainment---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-Board Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=2)
    leg_room = st.slider("Leg Room Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage Handling ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=5)
    checkin_service = st.slider("Check-in Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=0)
    inflight_service = st.slider("Inflight Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    departure_delay = st.slider("Departure Delay ---> Pick A Rating Between 0 and 1600 Minutes", min_value=0, max_value=1600, value=60)
    arrival_delay = st.slider("Arrival Delay ---> Pick A Rating Between 0 and 1200 Minutes", min_value=0, max_value=1200, value=60)

    # User input dataframe
    user_input = pd.DataFrame({
        'gender': [gender],
        'customer type': [customer_type],
        'age': [age],
        'type of travel': [travel_type],
        'class': [travel_class],
        'flight distance': [flight_distance],
        'inflight wifi service': [inflight_wifi],
        'departure/arrival time convenient': [time_convenient],
        'ease of online booking': [online_booking],
        'gate location': [gate_location],
        'food and drink': [food_drink],
        'online boarding': [online_boarding],
        'seat comfort': [seat_comfort],
        'inflight entertainment': [entertainment],
        'on-board service': [on_board_service],
        'leg room service': [leg_room],
        'baggage handling': [baggage_handling],
        'checkin service': [checkin_service],
        'inflight service': [inflight_service],
        'cleanliness': [cleanliness],
        'departure delay in minutes': [departure_delay],
        'arrival delay in minutes': [arrival_delay]
    })

    st.write("### Your Input Values:")
    st.dataframe(user_input)

    # Using Random Forest model for predictions since this was the most accurate in terms of understanding the training and test data:
    model = RandomForestClassifier()
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]


    # Display the result
    st.write(" ### Based on your input features, the model predicts that particular airline passenger will be:")
    st.write(f"# {prediction}")