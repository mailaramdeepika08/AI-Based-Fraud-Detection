import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Based Fraud Detection & Transaction Monitoring System",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔐 AI Based Fraud Detection & Transaction Monitoring System")

st.write(
    "An AI and Machine Learning based system for detecting "
    "fraudulent transactions and monitoring transaction risk."
)

st.divider()


# ============================================================
# CREATE SAMPLE TRANSACTION DATA
# ============================================================

np.random.seed(42)

number_of_transactions = 2000

data = pd.DataFrame({

    "Amount": np.random.randint(
        100,
        100000,
        number_of_transactions
    ),

    "Time": np.random.randint(
        0,
        24,
        number_of_transactions
    ),

    "Transaction_Type": np.random.choice(
        ["Online", "POS", "ATM"],
        number_of_transactions
    ),

    "Location": np.random.choice(
        [
            "Hyderabad",
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Chennai",
            "Unknown"
        ],
        number_of_transactions
    ),

    "Device": np.random.choice(
        [
            "Mobile",
            "Laptop",
            "Card"
        ],
        number_of_transactions
    )
})


# ============================================================
# CREATE FRAUD LABEL
# ============================================================

data["Is_Fraud"] = (

    (data["Amount"] > 70000)

    |

    (
        (data["Time"] < 5)
        &
        (data["Amount"] > 30000)
    )

    |

    (data["Location"] == "Unknown")

).astype(int)


# ============================================================
# ENCODE TEXT DATA
# ============================================================

encoders = {}

categorical_columns = [
    "Transaction_Type",
    "Location",
    "Device"
]

for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(
        data[column]
    )

    encoders[column] = encoder


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

X = data.drop(
    "Is_Fraud",
    axis=1
)

y = data["Is_Fraud"]


model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


model.fit(
    X,
    y
)


# ============================================================
# DASHBOARD CALCULATIONS
# ============================================================

total_transactions = len(data)

fraud_transactions = int(
    data["Is_Fraud"].sum()
)

genuine_transactions = (
    total_transactions
    -
    fraud_transactions
)

fraud_rate = (
    fraud_transactions
    /
    total_transactions
) * 100


# ============================================================
# DASHBOARD
# ============================================================

st.subheader("📊 Transaction Monitoring Dashboard")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        total_transactions
    )


with col2:

    st.metric(
        "Genuine Transactions",
        genuine_transactions
    )


with col3:

    st.metric(
        "Fraud Transactions",
        fraud_transactions
    )


with col4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


st.divider()


# ============================================================
# NEW TRANSACTION
# ============================================================

st.subheader("💳 Analyze New Transaction")


col1, col2 = st.columns(2)


with col1:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        max_value=1000000.0,
        value=1000.0,
        step=100.0
    )


    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "Online",
            "POS",
            "ATM"
        ]
    )


    location = st.selectbox(
        "Transaction Location",
        [
            "Hyderabad",
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Chennai",
            "Unknown"
        ]
    )


with col2:

    transaction_time = st.slider(
        "Transaction Time",
        min_value=0,
        max_value=23,
        value=12
    )


    device = st.selectbox(
        "Device Used",
        [
            "Mobile",
            "Laptop",
            "Card"
        ]
    )


st.write("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    # Encode input values

    transaction_type_encoded = (
        encoders["Transaction_Type"]
        .transform(
            [transaction_type]
        )[0]
    )


    location_encoded = (
        encoders["Location"]
        .transform(
            [location]
        )[0]
    )


    device_encoded = (
        encoders["Device"]
        .transform(
            [device]
        )[0]
    )


    # Create transaction dataframe

    new_transaction = pd.DataFrame({

        "Amount": [amount],

        "Time": [transaction_time],

        "Transaction_Type": [
            transaction_type_encoded
        ],

        "Location": [
            location_encoded
        ],

        "Device": [
            device_encoded
        ]

    })


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    prediction = model.predict(
        new_transaction
    )[0]


    probability = model.predict_proba(
        new_transaction
    )[0][1]


    risk_score = probability * 100


    st.divider()

    st.subheader("🔎 Transaction Analysis Result")


    col1, col2 = st.columns(2)


    # ========================================================
    # RESULT
    # ========================================================

    with col1:

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

        else:

            st.success(
                "✅ TRANSACTION APPEARS GENUINE"
            )


    # ========================================================
    # RISK SCORE
    # ========================================================

    with col2:

        st.metric(
            "Fraud Risk Score",
            f"{risk_score:.2f}%"
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk_score >= 70:

        st.error(
            "🔴 HIGH RISK"
        )

        st.write(
            "Immediate investigation is recommended."
        )


    elif risk_score >= 40:

        st.warning(
            "🟠 MEDIUM RISK"
        )

        st.write(
            "Transaction should be monitored."
        )


    else:

        st.success(
            "🟢 LOW RISK"
        )

        st.write(
            "Transaction appears to be normal."
        )


# ============================================================
# FRAUD ANALYSIS GRAPH
# ============================================================

st.divider()

st.subheader("📈 Fraud Analysis")


chart_data = pd.DataFrame({

    "Transaction Status": [
        "Genuine",
        "Fraud"
    ],

    "Number of Transactions": [
        genuine_transactions,
        fraud_transactions
    ]

})


st.bar_chart(
    chart_data.set_index(
        "Transaction Status"
    )
)


# ============================================================
# TRANSACTION TABLE
# ============================================================

st.subheader("📋 Recent Transactions")


display_data = data.copy()


display_data["Transaction Status"] = (
    display_data["Is_Fraud"]
    .map({
        0: "Genuine",
        1: "Fraud"
    })
)


display_data = display_data[
    [
        "Amount",
        "Time",
        "Transaction Status"
    ]
].head(15)


st.dataframe(
    display_data,
    use_container_width=True
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ System Information")


st.write(
    """
    **Project:** AI Based Fraud Detection & Transaction Monitoring System

    **Technology:** Python, Machine Learning and Streamlit

    **Algorithm:** Random Forest Classifier

    **Purpose:** Detect potentially fraudulent transactions,
    calculate fraud risk and monitor transaction activity.
    """
)


st.caption(
    "AI Based Fraud Detection & Transaction Monitoring System"
)