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



# Path to your JSON file
file_path = 'prompts.json'

# Load JSON file and convert to dictionary
with open(file_path, 'r') as file:
    prompts = json.load(file)

load_dotenv()

model = ChatGroq(
    temperature=0.08,
    model_name="llama3-8b-8192"
)


#String OutPut Parser

strop = StrOutputParser()

# Prompts

# summary generating prompt
prompt1 = PromptTemplate(
    template=prompts['prompt1'],
    input_variables=["video_json_here"],
)


# prompt to be passed to the 1st model in and 2nd step of llm call if question asked about the video releated

class VideoAnswer(BaseModel):
    answer: str = Field(description='Answer the user question releated to summary provided above.')
    source: str = 'video'

output_parser_2 = PydanticOutputParser(pydantic_object=VideoAnswer)

prompt2 = PromptTemplate(
    template=prompts['prompt2'],
    input_variables=['VIDEO_SUMMARY', 'USER_QUESTION'],
    partial_variables={"format_instructions": output_parser_2.get_format_instructions()}
)



# prompt to be passed to the 1st model in and 2nd step of llm call if question asked general querry not about the video releated

class GeneralAnswer(BaseModel):
    answer: str = Field(description='Answer the user question based on your memory.')
    source: str = 'general'

output_parser_3 = PydanticOutputParser(pydantic_object=GeneralAnswer)

prompt3 = PromptTemplate(
    template=prompts['prompt3'],
    input_variables=['USER_QUESTION'],
    partial_variables={"format_instructions": output_parser_3.get_format_instructions()}        
)




# Prompt which wil execute first

class ToolSelectorOutput(BaseModel):
    tool_name: str = Field(description='Name the tool')

output_parser_4 = PydanticOutputParser(pydantic_object=ToolSelectorOutput)

prompt4 = PromptTemplate(
    template=prompts['prompt4'],
    input_variables=["USER_QUESTION", "NEW"],
    partial_variables={"format_instructions": output_parser_4.get_format_instructions()}
)


# Tools Used in AI AGENTS

# Querry Answer
# Input schema
class QueryAnsInput(BaseModel):
    user_question: str = Field(description="The question asked by the user.")
    context: Optional[str] = Field(default="", description="Optional video summary context.")

# Output schema
class QueryAnsOutput(BaseModel):
    answer: str = Field(description="Answer to the user's question.")
    source: Literal["video", "general"]

# Mock implementations for demo
def answer_from_summary(question: str, context: str) -> str:
    # prompt = prompt2.invoke({'VIDEO_SUMMARY' : context, 'USER_QUESTION' : question})
    # ans_s = model.invoke(prompt2)

    chain = prompt2|model
    ans_s = chain.invoke({'VIDEO_SUMMARY' : context, 'USER_QUESTION' : question})
    ans_s = ans_s.content


    return f"Answered from video context: {ans_s}"

def general_qa(question: str) -> str:
    ans_s = model.invoke(question)
    return f"General answer to: {ans_s}"

# Tool
@tool(args_schema=QueryAnsInput)
def querry_ans(input: QueryAnsInput) -> QueryAnsOutput:
    """
    Answers user's question either using general knowledge or based on the video summary (if context is provided).
    """
    if input.context.strip():
        response = answer_from_summary(input.user_question, input.context)
        source = "video"
    else:
        response = general_qa(input.user_question)
        source = "general"

    return QueryAnsOutput(answer=response, source=source)





# Summary Event 
# Input schema
class VideoEventSummaryInput(BaseModel):
    video_json: Dict[str, Any] = Field(description="Generate summary of the captions given by user")

# Output schema
class VideoEventSummaryOutput(BaseModel):
    summary: str = Field(description="A summary and events detected in the video.")

def generate_video_summary(video_json: json) -> str:
    # prompt = prompt1.invoke({"video_json_here" : video_json})
    # vidsm = model.invoke(prompt)

    chain = prompt1|model
    vidsm = chain.invoke({"video_json_here" : video_json})
    vidsm = vidsm.content

    return f"Summary of video : {vidsm} "

# Tool
@tool(args_schema=VideoEventSummaryInput)
def video_event_summary(input: VideoEventSummaryInput) -> VideoEventSummaryOutput:
    """
    Generates a summary of the video events.
    Use this only when a new video has been uploaded and the user is asking for a summary or what events occurred.
    """
    summary = generate_video_summary(input.video_json)
    return VideoEventSummaryOutput(summary=summary)

# tools list
tools = [
    video_event_summary,
    querry_ans,
]

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,  # or another type as needed
    verbose=True
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main Function
def run_agent(user_input: str):
    result = executor.invoke({"input": user_input})
    print("\n Final Output:\n", result["output"])






