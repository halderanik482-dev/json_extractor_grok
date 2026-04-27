import streamlit as st
import pdfplumber
from groq import Groq
import json

# --- UI Setup ---
st.title("Medical Data JSON Extractor 📄➡️⚙️")
st.write("Powered by Groq & Llama 3. Upload a lab report to extract data into a clean JSON format.")

# --- API Setup ---
try:
    # Initialize the Groq client securely using Streamlit Secrets
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except KeyError:
    st.error("⚠️ API Key not found. Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()
except Exception as e:
    st.error(f"API Connection Error: {e}")
    st.stop()

# --- App Logic ---
uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Groq is reading the report at lightning speed..."):
        try:
            # 1. Read the PDF (Safely limited for free tier)
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                # Only read the first 4 pages to avoid token limits
                for page in pdf.pages[:4]: 
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            
            # Failsafe: Force cut the text at 18,000 characters (roughly 4,500 tokens)
            # This guarantees you will NEVER hit the 6,000 limit.
            text = text[:18000]
            
            # 2. Command the AI to return STRICT JSON
            prompt = f"""
            You are a data extraction API. Look at the medical report text below. 
            Extract ONLY the following three values: Haemoglobin, Vitamin B12, and Vitamin D.
            
            You MUST return a valid JSON object. 
            Use exactly these keys: "Haemoglobin", "Vitamin_B12", "Vitamin_D".
            Include the units in the value (e.g., "11.9 g/dL").
            If a value is missing, output "Not Found".
            Do NOT include markdown formatting, backticks, or any conversational text. Just the raw JSON.
            
            Text:
            {text}
            """
            
            # 3. Get the AI Response using Llama 3
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant", # This is Meta's fast, free open-source model
                temperature=0, # Keeps the AI focused on pure data extraction, no creativity
            )
            
            # 4. Clean and Parse the JSON
            raw_response = chat_completion.choices[0].message.content
            clean_text = raw_response.strip().removeprefix('```json').removesuffix('```').strip()
            data = json.loads(clean_text)
            
            # 5. Display the Results
            st.subheader("Extracted JSON Output:")
            st.json(data)
            
            # 6. Create the Download Button
            json_string = json.dumps(data, indent=4)
            st.download_button(
                label="📥 Download JSON File",
                file_name="extracted_medical_data.json",
                mime="application/json",
                data=json_string,
            )
            
        except json.JSONDecodeError:
            st.error("The AI failed to return properly formatted JSON. Please try again.")
            st.code(raw_response) 
        except Exception as e:
            st.error(f"An error occurred: {e}")
