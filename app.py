import streamlit as st
import os
from huggingface_hub import InferenceClient

# Set page configuration
st.set_page_config(
    page_title="Hugging Face Text Generation",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Hugging Face Text Generation")
st.markdown("Generate text using Hugging Face models via Featherless AI")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Get HF_TOKEN from Streamlit secrets or environment variable
try:
    hf_token = st.secrets["HF_TOKEN"]
except (KeyError, FileNotFoundError):
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    st.error("❌ HF_TOKEN not found. Please configure it in Streamlit secrets or environment variables.")
    st.info("For deployment, add your token to the Streamlit Community Cloud secrets.")
    st.stop()

# Model selection
model = st.sidebar.selectbox(
    "Select Model",
    ["marcelbinz/Llama-3.1-Centaur-70B"],
    index=0
)

# Max tokens input
max_tokens = st.sidebar.number_input(
    "Max New Tokens",
    min_value=1,
    max_value=2000,
    value=70,
    step=1,
    help="Maximum number of new tokens to generate"
)

# Temperature setting
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.70,
    step=0.01,
    help="Controls randomness in generation"
)

# Main interface
st.header("Enter Your Query")

# Text input area
user_query = st.text_area(
    "Your prompt:",
    placeholder="Enter your text generation prompt here...",
    height=150,
    help="Enter the text you want the model to complete or respond to"
)

# Generate button
if st.button("🚀 Generate Text", type="primary"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a query before generating.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Add this before the InferenceClient initialization
                st.write(f"🔍 **Debug Info:**")
                st.write(f"- Token starts with: {hf_token[:10]}...")
                st.write(f"- Token length: {len(hf_token)}")
                st.write(f"- Model: {model}")
                st.write(f"- Provider: featherless-ai")

                # Test model access first
                try:
                    # Try without provider first
                    test_client = InferenceClient(model=model, token=hf_token)
                    st.success("✅ Direct HF access successful")
                except Exception as e:
                    st.warning(f"⚠️ Direct HF access failed: {e}")

                # Initialize client
                client = InferenceClient(
                    provider="featherless-ai",
                    api_key=hf_token,
                )
                
                # Generate text - Updated to handle response properly
                response = client.text_generation(
                    user_query,
                    model=model,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Handle different response formats
                if isinstance(response, str):
                    result = response
                elif isinstance(response, dict):
                    # Try different possible keys
                    if 'generated_text' in response:
                        result = response['generated_text']
                    elif 'text' in response:
                        result = response['text']
                    elif 'content' in response:
                        result = response['content']
                    else:
                        # If none of the expected keys exist, show the raw response
                        result = str(response)
                        st.warning("⚠️ Unexpected response format. Showing raw response.")
                else:
                    result = str(response)
                
                # Display results
                st.success("✅ Generation completed!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Generated Text")
                    st.markdown("---")
                    st.write(result)
                
                with col2:
                    st.subheader("Settings Used")
                    st.markdown("---")
                    st.write(f"**Model:** {model}")
                    st.write(f"**Max Tokens:** {max_tokens}")
                    st.write(f"**Temperature:** {temperature}")
                
            except Exception as e:
                st.error(f"❌ Error: {type(e).__name__}: {e}")
                
                # Debug information
                st.expander("🔍 Debug Information").write(f"""
                **Error Type:** {type(e).__name__}
                **Error Message:** {str(e)}
                **Model:** {model}
                **Provider:** featherless-ai
                """)

# Footer
st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. Enter your text generation prompt in the text area above
2. Adjust settings in the sidebar as needed
3. Click generate to get your response

### 🔧 Troubleshooting:
- If you encounter errors, check the debug information in the expandable section
- Ensure your HF_TOKEN has access to the selected model
""")
