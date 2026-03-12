import warnings
warnings.filterwarnings("ignore")

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-6")
parser = StrOutputParser()

#----State---------
class SolutionState(TypedDict):
    problem:        str
    research:       str
    architecture:   str
    human_feedback: str
    final_brief:    str
    next_agent:     str

#---Supervisor----
def supervisor_node(state: SolutionState) -> dict:
    print("\n[SUPERVISOR] Checking State")

    if not state.get("research"):
        print("[SUPERVISOR] -> RESEARCHER")
        return{"next_agent": "researcher"}
    
    elif not state.get("architecture"):
        print("[SUPERVISOR] -> ARCHITECT")
        return{"next_agent": "architect"}
    
    elif not state.get("final_brief"):
        print("[SUPERVISOR] -> Writer")
        return{"next_agent": "writer"}
    
    else:
        print("[SUPERVISOR] -> Done")
        return {"next_agent": "done"}
    
#----Researcher----
def researcher_node(state: SolutionState) -> dict:
    print("\n[RESEARCHER] Anlalysing problem....")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Research Analyst at Maveric.
         Analyse the client problem and extract:
         1.Root Cause of the business problem
         2.Key analytical requirements
         3.Data Landscape"""),
         ("human", "Analyse this problem:\n{problem}")
    ])

    chain = prompt | model | parser
    research = chain.invoke({"problem": state["problem"]})
    return {"research": research}

#-----Architect-----
def architect_node(state: SolutionState) -> dict:
    print("\n[ARCHITECTURE] Designing solution...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior solution AI Architect in Maveric
         Based on the research provided design:
         1. Architecture patterns and components
         2. Technology stack
         3. Cloud deployment approach"""),
         ("human", """Problem: {problem}
          Research: {research}
          Design the architecture.""")
    ])

    chain = prompt | model | parser
    architecture = chain.invoke({
        "problem":  state["problem"],
        "research": state["research"]
    })
    return {"architecture": architecture}

#----WRITER-----
def writer_node(state: SolutionState) -> dict:
    print("\n[WRITER] writng client brief...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior consultant at Maveric.
         Synthesis everything into a client-facing brief.
         If human feedback is provided  - incorporate into the brif.
         1. Executive Summary
         2. Recommended Solution
         3. Key Benefits
         4. Delivery Timeline
         5. Next Steps"""),
         ("human", """Problem: {problem}
          Research: {research}
          Architecture: {architecture}
          Human feedback to incorporate: {human_feedback}
          write the brief.""")
    ])

    chain = prompt | model | parser
    final_brief = chain.invoke({
        "problem": state["problem"],
        "research": state["research"],
        "architecture": state["architecture"],
        "human_feedback": state.get("human_feedback provided", "No Feedback Provided")
    })

    return{"final_brief": final_brief}

#---Routing----
def route_next(state: SolutionState) -> dict:
    return state["next_agent"]

#---Build Grpah----
graph = StateGraph(SolutionState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("architect", architect_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "researcher": "researcher",
        "architect": "architect",
        "writer": "writer",
        "done": END
    }
)

graph.add_edge("researcher", "supervisor")
graph.add_edge("architect", "supervisor")
graph.add_edge("writer", "supervisor")

#--- Memory Saver----
#checkpoint state after every node
#enable pause + resume
memory = MemorySaver()
agent = graph.compile(
    checkpointer=memory,
    interrupt_before=["writer"]
)

#--- Run with Human in the Loop----

problem = """
A large BPO company needs analysts to query 500K documents
on SharePoint using natural language. Users should only see
documents they are authorised to view. Answers needed in
under 5 seconds with source citations.
"""

# thread_id = unique session identifier
# same concept as session_id from Day 2 memory
config = {"configurable": {"thread_id": "session_yuvaraj_1"}}

print("=" * 60)
print("PHASE 1 — Running until human checkpoint")
print("=" * 60)

#first run - agent will pause before writer
agent.invoke({
    "problem": problem,
    "research": "",
    "architecture": "",
    "human_feedback": "",
    "final_brief": "",
    "next_agent":""
}, config = config)

#---human review point----
print("\n" + "=" * 60)
print("⏸️  AGENT PAUSED — Human review required")
print("=" * 60)

# Inspect current state — what did the agent produce?
current_state = agent.get_state(config)
print("\n📋 ARCHITECTURE PRODUCED:")
print(current_state.values["architecture"])

#human provides feedback
print("\n" + "=" * 60)
feedback = input("Your feedback (or press Enter to approve): ")
print("=" * 60)

# Update state with human feedback
agent.update_state(
    config,
    {"human_feedback": feedback if feedback else "Approved — proceed as designed"}
)
# ── RESUME ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("▶️  RESUMING — Writer incorporating feedback")
print("=" * 60)

# Resume from checkpoint — continues where it paused
result = agent.invoke(None, config=config)

# Save output
with open("hitl_brief.md", "w") as f:
    f.write(result["final_brief"])

print("\n" + "=" * 60)
print("✅ FINAL BRIEF")
print("=" * 60)
print(result["final_brief"])
print("\n✅ Saved to hitl_brief.md")


              

