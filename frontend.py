import streamlit as st
from PIL import Image
import os
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnableLambda, RunnableBranch, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional
import json
from langchain.agents import tool, Tool, Agent, AgentExecutor, initialize_agent, AgentType
from langchain_core.messages import HumanMessage, ToolMessage
from video_json import *
from langchain_integration import *


CWD = os.getcwd()
logo_name = "chatbot_image.jpg"
logo_path = os.path.join(CWD,logo_name)
logo_image = Image.open(logo_path)
folder_name = "frames"
folder_path = os.path.join(CWD,folder_name)
os.makedirs(folder_path,exist_ok=True)

if __name__ == "__main__":
    fps = 1  # 1 frame per second
    result = process_frames_and_caption(folder_path, fps=fps)

    temporal_result_dict = build_temporal_object_dict(result)
    pretty_output = json.dumps(temporal_result_dict)



# Session state to cache summary
if 'video_summary' not in st.session_state:
    st.session_state.video_summary = ''
if "chat" not in st.session_state:
    st.session_state.chat_history = []


# --- Header with Title and Image ---
logo_path = "chatbot_image.jpg"  # Put your logo path here
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("## Personal Assistant")
with col2:
    st.image(Image.open(logo_path), width=50)

st.divider()

# --- File Uploader ---
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
new=False


# If video is uploaded, process it and cache result
if uploaded_video:
    video_path = os.path.join("temp_video", uploaded_video.name)
    os.makedirs("temp_video", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    new = True
    st.success("Video uploaded successfully")
    video_summary = generate_video_summary(pretty_output)
    st.session_state.video_summary = video_summary
    st.success("Video summary generated.")
    

# --- User Query Input ---
query = st.text_input("Ask a question...")

if st.button("Generate Answer"):
    st.success("Answer Generating...")

    if(query) :
        st.session_state.chat_history.append(('user', ))
        

    # --- Handle User Query ---
    if query:

        if (st.session_state.video_summary):
                st.success("JSON already exists. Loading it now...")
        else:
            with st.spinner("Generating JSON for your output... Please wait"):
                    pass
            st.success("JSON generated successfully!")

        if st.session_state.video_summary:
            # If video summary exists, answer based on it
            response = answer_from_summary(query, st.session_state.chat_history + list(st.session_state.video_summary))
            st.session_state.chat_history.append(('user', query))
            st.session_state.chat_history.append(('ai_video_based', response))
        else:
            # No video, answer using general LLM
            response = general_qa(query)
            st.session_state.chat_history.append(('user', query))
            st.session_state.chat_history.append(('ai_general_model', response))

        st.markdown("### Response:")
        st.write(response)






# # RunnableBranch
# branch1 = RunnableBranch(
#     (new, )
# )

# branch2 = RunnableBranch(
#     (lambda input: input["tool_name"] == "video_event_summary", RunnableLambda(prompt1.invoke({"video_json_here" : pretty_output}))),
#     (lambda input: input["tool_name"] == "querry_ans", RunnableLambda())
# )


# chain = prompt4 | model | 

# result = chain.invoke({"USER_QUESTION" : "summarize this video", "NEW": True})