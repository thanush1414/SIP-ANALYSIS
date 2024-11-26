import streamlit as st
import json
import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import ipaddress
import re
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def preprocess_csv(file_path):
    combined_df = pd.read_csv(file_path)
    
    combined_df = combined_df.dropna()
    
    combined_df['Source'] = combined_df['Source'].apply(convert_address_to_numeric)
    combined_df['Destination'] = combined_df['Destination'].apply(convert_address_to_numeric)
    combined_df = combined_df.drop(['No.', 'Info'], axis=1)
    
    protocol_dummies = pd.get_dummies(combined_df['Protocol'], prefix='Protocol')
    combined_df = pd.concat([combined_df, protocol_dummies.astype(int)], axis=1)
    combined_df = pd.concat([combined_df, protocol_dummies], axis=1)
    combined_df1 = combined_df.drop('Protocol', axis=1)
    combined_df1['Source'] = combined_df1['Source'].astype(float)
    print(combined_df1.info)
    return combined_df1  
def convert_address_to_numeric(address):
    if ':' in address:
        # MAC address
        mac_address = re.sub(r':', '', address)
        return int(mac_address, 16)
    else:
        # IPv4 address
        try:
            return int(ipaddress.IPv4Address(address))
        except ValueError:
            # Handle non-IPv4 addresses
            return address

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None
def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")
        
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")

def main(json_file_path="data.json"):
    st.sidebar.title("SIP Signal Detection")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "SIP Signal Detection"),
        key="Arrhythmia Detection",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "SIP Signal Detection":
        if session_state.get("logged_in"):
            st.title('SIP Signal Detection')
            st.write('Please upload packet info in csv format')
            all_protocol_columns = [
                'Protocol_ARP',
                'Protocol_CLASSIC-STUN', 'Protocol_DCERPC', 'Protocol_DHCP',
                'Protocol_DISCARD', 'Protocol_DNS', 'Protocol_DTLS', 'Protocol_HTTP',
                'Protocol_ICMP', 'Protocol_ICMPv6', 'Protocol_MDNS', 'Protocol_NBNS',
                'Protocol_NBSS', 'Protocol_QUAKE3', 'Protocol_RDP', 'Protocol_RTCP',
                'Protocol_RTP', 'Protocol_RTP EVENT', 'Protocol_SIP',
                'Protocol_SIP/SDP', 'Protocol_SIP/XML', 'Protocol_SMB', 'Protocol_SSH',
                'Protocol_SSHv2', 'Protocol_STUN', 'Protocol_TCP', 'Protocol_UDP',
                'Protocol_WTLS+WTP+WSP'
            ]

            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if st.button("Submit"):
                if uploaded_file is not None:
                    # Read the uploaded file
                    combined_df = pd.read_csv(uploaded_file)

                    # Preprocess the data
                    combined_df = combined_df[(combined_df['Source'].apply(lambda x: '.' in x or ':' in x)) & (combined_df['Destination'].apply(lambda x: '.' in x or ':' in x))]
                    combined_df['Source'] = combined_df['Source'].apply(convert_address_to_numeric)
                    combined_df['Destination'] = combined_df['Destination'].apply(convert_address_to_numeric)
                    combined_df = combined_df.drop(['No.', 'Info'], axis=1)
                    combined_df['Protocol'] = combined_df['Protocol'].astype('category')
                    protocol_dummies = pd.get_dummies(combined_df['Protocol'], prefix='Protocol')
                    missing_protocol_columns = set(all_protocol_columns) - set(protocol_dummies.columns)
                    for col in missing_protocol_columns:
                        protocol_dummies[col] = 0

                    #Replace new protocols with 'TCP'
                    for col in protocol_dummies.columns:
                        if col not in all_protocol_columns:
                            protocol_dummies['Protocol_TCP'] = 1
                            break

                    # Concatenate the protocol dummies with the original DataFrame
                    combined_df = pd.concat([combined_df, protocol_dummies.astype(int)], axis=1)
                    combined_df = combined_df.drop('Protocol', axis=1)
                    combined_df['Source'] = combined_df['Source'].astype(float)
                    combined_df['Destination'] = combined_df['Destination'].astype(float)
                    # combined_df
                    # Split the data into features (X_test) and target (y_test)
                    X_test = combined_df
                    # st.write(print(X_test.info()))
                    # y_test = combined_df['Destination']

                    # Convert the data to PyTorch tensors
                    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                    test_dataset = TensorDataset(X_test_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                    # Load the pre-trained model
                    loaded_model = CNN(32, 2)
                    loaded_model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))

                    loaded_model.eval()

                    # Make predictions
                    predictions = []
                    with torch.no_grad():
                        for inputs in test_loader:
                            inputs = inputs[0]
                            outputs = loaded_model(inputs)
                            _, predicted_classes = torch.max(outputs, 1)
                            predictions.extend(predicted_classes.numpy().tolist())

                    # Count the number of predictions for each class
                    num_class_0 = predictions.count(0)
                    num_class_1 = predictions.count(1)

                    # Display the counts in the Streamlit UI
                    st.markdown("""
                    <style>
                    .prediction-container {
                        background-color: #f0f0f0;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                    }
                    .prediction-result {
                        font-size: 18px;
                        font-weight: bold;
                    }
                    .normal {
                        color: green;
                    }
                    .attack {
                        color: red;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    st.write("Results:")
                    st.write('<div class="prediction-container" >', unsafe_allow_html=True)
                    st.write(f'<p class="prediction-result">Number of predictions for class 0 (Attack): {num_class_0}</p>', unsafe_allow_html=True)
                    st.write(f'<p class="prediction-result">Number of predictions for class 1 (Normal): {num_class_1}</p>', unsafe_allow_html=True)

                    if num_class_0 > num_class_1:
                        st.write('<p class="prediction-result normal">Overall Result: Attack Detected</p>', unsafe_allow_html=True)
                    else:
                        st.write('<p class="prediction-result attack">Overall Result: Normal</p>', unsafe_allow_html=True)

                    st.write('</div>', unsafe_allow_html=True)

                else:
                    st.warning("Please upload a CSV file to use the app.")

        else:
            st.warning("Please login/signup to use app!!")

               


if __name__ == "__main__":
    initialize_database()
    main()
