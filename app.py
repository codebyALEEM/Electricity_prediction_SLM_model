import streamlit as st
import numpy as np
import pickle


st.markdown(
    """
    <style>
    .stApp {
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url("https://paylesspower.com/wp-content/uploads/2023/11/i1_How-to-Pay-Your-Texas-Electric-Bill.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Load the dataset
model = pickle.load(open(r'C:\Users\VICTUS\Desktop\mastering git\Practise git\Electricity_prediction_SLM_model\SLR_Electricity_Prediction.pkl','rb'))

# Set the title of the streamlit app
st.title('Electricity Bill Prediction')

# Add a brief description 
st.write('This app predicts the Electricity Bill based on number of units used using a simple linear regression model.')


# Add input widget for user to enter years of experience
No_of_units_used = st.number_input('Enter number of units used :',min_value=0.0,value=100.0,step=5.0)


# When the button is clicked , make predictions
if st.button("Predict Electricity bill"):
    #Make a prediction using the trained model
    No_of_units_input = np.array([[No_of_units_used]]) # Convert the input to a 2D array for prediction
    prediction = model.predict(No_of_units_input)
    
    # Display the result
    st.success(f"The predicted Electricity Bill for {No_of_units_used} units is : ${prediction[0]:,.2f}") 
    

#Display information about the model
st.write("The model was trained using a dataset of Electricity Bill and number of unit used.")
