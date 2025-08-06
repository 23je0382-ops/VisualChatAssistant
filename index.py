import




# RunnableBranch
branch = RunnableBranch(
    (lambda input: input["tool_name"] == "video_event_summary", prompt1.invoke()),
    (lambda input: input["tool_name"] == "querry_ans", branch_b)
)


chain = prompt4 | model | 

result = chain.invoke({"USER_QUESTION" : "summarize this video", "NEW": True})