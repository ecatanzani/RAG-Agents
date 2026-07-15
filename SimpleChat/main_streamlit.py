import streamlit as st
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def main():
    st.set_page_config(page_title="Bedrock Chat Assistant", page_icon="🤖")
    st.title("🤖 AWS Bedrock Chat Assistant")
    st.caption("Powered by Amazon Nova Lite")

    @st.cache_resource
    def load_model():
        model_id = 'eu.amazon.nova-lite-v1:0'
        return ChatBedrock(
            model_id=model_id,
            temperature=0.7
        )
    
    model = load_model()

    SYSTEM_PROMPT = (
        "You are a helpful, brilliant, and slightly witty AI assistant. "
        "Your goal is to help the user solve technical problems, brainstorm ideas, "
        "and write clean code. Keep your responses structured, concise, and engaging."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    if user_query := st.chat_input("Ask me anything..."):
        
        with st.chat_message("user"):
            st.write(user_query)
        
        st.session_state.messages.append(HumanMessage(content=user_query))

        # Generate assistant response with a visual loading spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = model.invoke(st.session_state.messages)
                    response_text = response.content
                    st.write(response_text)
                    st.session_state.messages.append(AIMessage(content=response_text))
                except Exception as e:
                    st.error(f"Error communicating with AWS Bedrock: {e}")
                    st.info("Make sure your AWS credentials are configured properly in your environment.")

if __name__ == '__main__':
    main()