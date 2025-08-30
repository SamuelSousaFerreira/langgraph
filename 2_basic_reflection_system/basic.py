from typing import List, Sequence
# typing helpers (List, Sequence) are available if you need to annotate types

from dotenv import load_dotenv
# load_dotenv reads environment variables from a .env file into the process env

from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage: generic message type used by LangChain
# HumanMessage: concrete message type representing user/human input

from langgraph.graph import END, MessageGraph
# END: sentinel to indicate graph termination
# MessageGraph: class used to build/compile a message-processing graph

from chains import generation_chain, reflection_chain
# Import two chains defined in chains.py:
# - generation_chain: chain that generates content based on messages
# - reflection_chain: chain that critiques/reflects on messages


# Load environment variables from .env (important before initializing LLMs)
load_dotenv()


# --- Graph node name constants -------------------------------------------------
REFLECT = "reflect"   # node name used for the reflection step
GENERATE = "generate" # node name used for the generation step


# Create an empty MessageGraph instance to which we'll add nodes and edges
graph = MessageGraph()


# --- Node implementations -----------------------------------------------------
def generate_node(state):
    """Node function for generation.

    Args:
        state: a list/sequence of messages representing the current conversation/state.

    Returns:
        The output of the generation_chain invoked with the current messages.
    """
    # Call the generation chain with the current messages state and return its result
    return generation_chain.invoke({
        "messages": state
    })


def reflect_node(messages):
    """Node function for reflection.

    Args:
        messages: the messages to reflect on (usually the conversation so far).

    Returns:
        A list containing a HumanMessage built from the reflection chain's output.
        Returning a list of messages allows the graph to append them to the state.
    """
    # Invoke the reflection chain using the provided messages
    response = reflection_chain.invoke({
        "messages": messages
    })

    # Wrap the chain response as a HumanMessage so it can be consumed by next nodes
    return [HumanMessage(content=response.content)]


# --- Register nodes and entry point -------------------------------------------
# Add the generation and reflection nodes to the graph
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# Set the starting node for the graph execution
graph.set_entry_point(GENERATE)


# --- Control flow function ---------------------------------------------------
def should_continue(state):
    """Decides whether to continue the loop or finish.

    The function receives the current state (list of messages) and returns either
    END (to terminate) or the name of the next node to execute (REFLECT).
    """
    # If the conversation/state grows beyond 6 messages, end the graph
    if (len(state) > 6):
        return END

    # Otherwise, continue by going to the reflection node
    return REFLECT


# Add a conditional transition from GENERATE using should_continue
graph.add_conditional_edges(GENERATE, should_continue)

# After reflection, always go back to generation
graph.add_edge(REFLECT, GENERATE)


# Compile the graph into an executable app (resolves nodes/edges into a runnable flow)
app = graph.compile()


# --- Optional: visualize the graph -------------------------------------------
# Print a Mermaid diagram (useful for documentation/visual checks)
print(app.get_graph().draw_mermaid())

# Also print a simple ASCII representation of the graph to the console
app.get_graph().print_ascii()


# --- Run the graph -----------------------------------------------------------
# Invoke the compiled graph with an initial HumanMessage (starting user prompt)
# Note: some graph runtimes accept a single message or a list; keep the same shape
response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

# Print the final response produced by the graph
print(response)