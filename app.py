import streamlit as st
import pickle
import re

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


#Custom “Re-run” Button instead Rerun
if st.button("🔄 Re-run Analysis"):
    st.rerun()



##CSS for Cards
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 18px;
    font-weight: 600;
}
.fake {
    background-color: #fdecea;
    color: #c0392b;
    border-left: 6px solid #e74c3c;
}
.real {
    background-color: #eafaf1;
    color: #1e8449;
    border-left: 6px solid #2ecc71;
}
.confidence {
    font-size: 14px;
    color: gray;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)




# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Header
st.markdown(
    "<h1 style='text-align:center;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered system to classify news as Fake or Real</p>",
    unsafe_allow_html=True
)

# Input
news_input = st.text_area(
    "📝 Paste News Article",
    height=220,
    placeholder="Paste full news article here..."
)

##LITTLE CSS
st.markdown("""
<style>
div.stButton > button {
    background-color: #2ecc71;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #27ae60;
    color: white;
}
</style>
""", unsafe_allow_html=True)



# Predict
if st.button("🔍 Analyze News", use_container_width=True):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        st.markdown("---")

        # ✅ ICON + CARD BASED RESULT (INSIDE BUTTON)
        if prediction == 0:
            st.markdown(
                f"""
                <div class="card fake">
                    ❌ Fake News Detected
                    <div class="confidence">
                        Confidence: {probability[0]*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="card real">
                    ✅ Real News Detected
                    <div class="confidence">
                        Confidence: {probability[1]*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )



# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:right;color:gray;'>Built by Karan | Only AI & ML Project</p>",
    unsafe_allow_html=True
)