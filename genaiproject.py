import streamlit as st
import base64
import os
from gtts import gTTS
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import fitz  # PyMuPDF

# ✅ Updated LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ✅ New OpenAI v1 client
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="GenAI Projects", layout="wide")
st.title("🤖 Generative AI Projects")

# Tabs for two projects
tab1, tab2 = st.tabs(["🧑 Avatar Assistant", "🖼️📄 Image Captioning + Doc Q&A"])

# -------------------------------------------------
# TAB 1: Avatar Assistant
# -------------------------------------------------
with tab1:
    st.header("🧑 Generative AI Avatar Assistant")

    user_input = st.text_input("Say something to the avatar:")

    if user_input:
        # GPT Response with new OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}]
        )
        answer = response.choices[0].message.content
        st.markdown(f"**Avatar says:** {answer}")

        # Text-to-Speech
        tts = gTTS(answer)
        tts.save("avatar_response.mp3")

        audio_file = open("avatar_response.mp3", "rb").read()
        b64 = base64.b64encode(audio_file).decode()
        st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")

        # Avatar placeholder
        st.image("https://i.imgur.com/0X2GJ2j.gif", caption="AI Avatar Talking...")

# -------------------------------------------------
# TAB 2: Image Captioning + Document Q&A
# -------------------------------------------------
with tab2:
    st.header("🖼️ Image Captioning + 📄 Document Q&A Bot")

    # ----- Image Captioning -----
    st.subheader("Upload an Image for Captioning")
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if img_file:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        image = Image.open(img_file).convert("RGB")

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        st.image(image, caption=f"Caption: {caption}", use_column_width=True)

    # ----- Document Q&A -----
    st.subheader("Upload a PDF for Q&A")
    pdf_file = st.file_uploader("Choose a PDF...", type="pdf")

    if pdf_file:
        # Extract text from PDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        # Create embeddings + retriever
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db = FAISS.from_texts([text], embeddings)

        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever())

        query = st.text_input("Ask a question about the document:")
        if query:
            result = qa.run(query)
            st.markdown(f"**Answer:** {result}")
