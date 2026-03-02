import streamlit as st
import pandas as pd
import joblib
from google.cloud import bigquery
import os
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from google.auth import default

st.set_page_config(layout="wide")

# -------------------------
# CSS DESIGN
# -------------------------

st.markdown("""
<style>

/* REMOVE STREAMLIT HEADER */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* REMOVE TOP WHITE BAR */
.stApp > header {
background-color: transparent;
}

/* REMOVE TOP GAP */
.block-container{
padding-top:1rem;
padding-bottom:2rem;
}

/* BACKGROUND IMAGE */
.stApp {
background-image:url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
background-size:cover;
background-repeat:no-repeat;
background-attachment:fixed;
}



/* TITLE STYLE */
.title{

text-align:center;
font-size:48px;
font-weight:bold;

padding:20px;

width:60%;
margin:auto;
margin-bottom:30px;

border-radius:20px;

/* Glass Effect */
background:rgba(0,0,0,0.35);
backdrop-filter:blur(10px);
box-shadow:0px 5px 25px rgba(0,0,0,0.3);

/* Animated Gradient Text */
background-image: linear-gradient(
270deg,
#00ffcc,
#00c6ff,
#66ff66,
#f9a825,
#ff4ecd
);

background-size:600% 600%;
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;

animation:gradientMove 6s ease infinite;

}

/* Animation */
@keyframes gradientMove{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}


/* BUTTON STYLE */
.stButton > button{

background:linear-gradient(90deg,#00bcd4,#1976d2);
color:white;
border-radius:12px;
height:45px;
width:100%;
font-size:18px;
font-weight:bold;

}

/* DARK TEXT */
h1,h2,h3,h4,label{
color:#222 ;
}
/* BEST PREDICTION CARD (ANIMATED) */

.bestcard{

background:linear-gradient(
270deg,
#66bb6a,
#43a047,
#2e7d32,
#81c784,
#a5d6a7
);

background-size:400% 400%;
animation:bestMove 6s ease infinite;

padding:25px;
border-radius:18px;
margin-top:20px;

text-align:center;
color:white;

box-shadow:
0px 10px 35px rgba(0,0,0,0.35),
0px 0px 20px rgba(76,175,80,0.6);

border-left:10px solid #1b5e20;

}


.bestcard h2{
font-size:38px;
margin:5px;
font-weight:bold;
}

.bestcard h3{
margin:5px;
font-weight:600;
}


/* Animation */

@keyframes bestMove{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}


</style>
""",unsafe_allow_html=True)
# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load("model.pkl")
le_state = joblib.load("le_state.pkl")
le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")


# -------------------------
# BIGQUERY
# -------------------------

@st.cache_data(ttl=60)
def load_data():

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    client = bigquery.Client(
        credentials=credentials,
        project="smart-agriculture-488414"
    )

    query = """
    SELECT state, crop, season, year, area, production
    FROM `smart-agriculture-488414.smart_yield.crop_data`
    """

    return client.query(query).to_dataframe()

df = load_data()

# -------------------------
# TITLE
# -------------------------

st.markdown(
"<div class='title'>🌾 Crop Yield Prediction System</div>",
unsafe_allow_html=True
)


# -------------------------
# LAYOUT
# -------------------------

left,right = st.columns([1,2])


# -------------------------
# LEFT PANEL
# -------------------------

with left:

    st.markdown("<div class='smallcard'>",unsafe_allow_html=True)

    st.subheader("AI Prediction Inputs")

    area_input = st.number_input(
    "Enter Area",
    100,
    10000,
    3000
    )

    production_input = st.number_input(
    "Enter Production",
    100,
    20000,
    9000
    )

    predict = st.button("Predict Best State")

    st.markdown("</div>",unsafe_allow_html=True)



# -------------------------
# RIGHT PANEL
# -------------------------

with right:

    st.markdown("<div class='bigcard'>",unsafe_allow_html=True)

    st.subheader("Select Crop, Season and Year")

    crop = st.selectbox("Select Crop", le_crop.classes_)

    season = st.selectbox("Select Season", le_season.classes_)

    year = st.selectbox(
    "Select Year",
    sorted(df['year'].unique())
    )

    st.markdown("</div>",unsafe_allow_html=True)



# -------------------------
# PREDICTION
# -------------------------

if predict:

    states = df['state'].unique()

    results=[]

    for state in states:

        sample = pd.DataFrame([{

        'state':le_state.transform([state])[0],
        'crop':le_crop.transform([crop])[0],
        'season':le_season.transform([season])[0],
        'year':year,
        'area':area_input,
        'production':production_input

        }])

        pred=model.predict(sample)[0]

        results.append([state,pred])


    results_df=pd.DataFrame(
    results,
    columns=['State','Predicted Yield']
    )

    results_df=results_df.sort_values(
    by='Predicted Yield',
    ascending=False
    )


# -------------------------
# TOP 5 TABLE
# -------------------------

    st.markdown("## Top 5 States")

    top5=results_df.head(5).reset_index(drop=True)

    top5.index=top5.index+1


    def highlight(row):

        colors={
        1:"background-color:#FFD54F",
        2:"background-color:#81C784",
        3:"background-color:#4DB6AC",
        4:"background-color:#64B5F6",
        5:"background-color:#90CAF9"
        }

        return [colors.get(row.name,"")]*len(row)


    st.dataframe(
    top5.style.apply(highlight,axis=1),
    width='stretch'
    )


# -------------------------
# BEST STATE CARD
# -------------------------

    best=top5.iloc[0]

    st.markdown(f"""
    <div class='bestcard'>
    <h3>🏆 Best State</h3>
    <h2>{best['State']}</h2>
    <h3>Yield : {best['Predicted Yield']:.2f}</h3>
    </div>
    """,unsafe_allow_html=True)



# -------------------------
# SMALL CHART
# -------------------------

    st.markdown("### Yield Chart")
    fig, ax = plt.subplots(figsize=(4,2))
    colors = ['red','green','blue','orange','purple']
    ax.bar(top5["State"], top5["Predicted Yield"], color=colors[:len(top5)])
    plt.xticks(rotation=45)
    # REMOVE BACKGROUND
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # REMOVE BORDER
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig, use_container_width=False)

# -------------------------
# SMALL MAP
# -------------------------

    st.markdown("### State Map")

    map_data=pd.DataFrame({

    'lat':[28.6,13.0,10.8,31.1,15.5],
    'lon':[77.2,80.2,76.3,75.3,73.8]

    })

    st.map(map_data,height=400)



# -------------------------
# DOWNLOAD
# -------------------------

    csv=top5.to_csv(index=False).encode('utf-8')

    st.download_button(

    label="Download Report",
    data=csv,
    file_name="report.csv"

    )



# -------------------------
# EXIT
# -------------------------

if st.button("Exit Application"):

    os._exit(0)
