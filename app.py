import json
import base64
import requests
import streamlit as st

st.set_page_config(page_title="Binary inference", layout="centered")
st.title("Chest x-ray diagnosis")

# Gateway URL for invoking the endpoint
API_URL = "https://7czg695149.execute-api.us-east-1.amazonaws.com/ProductionStage/Predict-pneumonia"

uploaded = st.file_uploader("Upload an x-ray image", type=["jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption=uploaded.name, use_container_width=True)
    img_bytes=uploaded.getvalue()
    # content_type = uploaded.type or "application/octet-stream"
    content_type = "application/x-image"
    st.caption ( f"Sending as content-type {content_type} | Size: {len(img_bytes)} bytes")
    if st.button("Predict"):
        try:
            with st.spinner("Calling your API ... "):
                r=requests.post(
                    API_URL,
                    data=img_bytes,
                    headers={"Content-Type":content_type},
                    timeout=60
                )
            st.caption(f"HTTP {r.status_code}")
            if r.ok:
                # Read text once (works for both JSON and plain text)
                text = r.text.strip()

                # API Gateway sometimes wraps plain string responses in quotes
                if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                    text = text[1:-1].strip()

                # If it looks like JSON, render as JSON; otherwise render as text
                looks_like_json = (
                    (text.startswith("{") and text.endswith("}")) or
                    (text.startswith("[") and text.endswith("]"))
                )

                if looks_like_json:
                    try:
                        st.json(json.loads(text))
                    except Exception:
                        st.write(text)
                else:
                    st.success("Prediction")
                    st.write(text)

            else:
                st.error("API returned an error.")
                st.code(r.text)
                st.stop()
        except requests.exceptions.RequestException as e:
            st.error("Request failed")
            st.write(str(e))
else:
    st.info("Upload an image to start")



