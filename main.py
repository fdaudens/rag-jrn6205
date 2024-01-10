
# If trying this app locally, comment out these 3 lines
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

#Adapted from Charly Wargnier's code: https://github.com/CharlyWargnier/LangchainRAG-Trubrics-Langsmith. 

import os
import streamlit as st
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from embeddings import initialize_chain
#from vanilla_chain import get_llm_chain

# Set LangSmith environment variables
openai_api_key = st.secrets.OPENAI_API_KEY
langchain_tracing = st.secrets.LANGCHAIN_TRACING_V2
langchain_endpoint = st.secrets.LANGCHAIN_ENDPOINT
langchain_api_key = st.secrets.LANGCHAIN_API_KEY
langchain_project = st.secrets.LANGCHAIN_PROJECT

client = Client()

st.set_page_config(
    page_title="L'assistant-robot du cours JRN6205 💬📚",
    page_icon="💬📚",
)

#initialisation du chat

if "last_run" not in st.session_state:
    st.session_state["last_run"] = "some_initial_value"

st.write("")


st.subheader("💬📚 L'assistant-robot du cours JRN6205")

st.info(
"""
Cet outil expérimental permet de poser des questions sur les notes de cours et les présentations. Les notes de cours seront accessibles peu après chaque cours. 

*Si vous changez de sujet, n'oubliez pas d'effacer l'historique des messages pour que le contexte ne soit pas pris en compte dans les interactions avec le modèle.*
"""
)

# Initialize State
if "trace_link" not in st.session_state:
    st.session_state.trace_link = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None

_DEFAULT_SYSTEM_PROMPT = ""
system_prompt = _DEFAULT_SYSTEM_PROMPT = """
Tu es un assistant pédagogique spécialisé dans le soutien des étudiants du cours JRN6205 de Journalisme Numérique. Ta mission principale est d'aider ces étudiants à comprendre et à approfondir les sujets enseignés dans ce cours. Pour cela, tu dois :

1. Utiliser les outils et la base de connaissances spécifiques au cours de Journalisme Numérique pour rechercher et vérifier les informations avant de donner une réponse. Cela garantit que tes réponses sont toujours précises, actualisées et pertinentes au sujet.

2. Ne jamais fournir de réponse sans avoir préalablement consulté les outils pertinents. Cela t'aide à maintenir un haut niveau d'exactitude et de pertinence dans tes réponses.

3. T'abstenir de référencer ou de suggérer des sources de données spécifiques dans tes réponses. Ta source de données est implicite dans tes capacités et doit être intégrée de manière transparente dans les réponses fournies, afin d'éviter toute confusion.

4. Toujours interagir en français avec les utilisateurs, respectant ainsi la langue principale du cours.

Ton rôle est de fournir des réponses informatives, précises et utiles, en t'assurant toujours de la clarté et de la pertinence de l'information transmise.
"""

system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")

memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

chain = initialize_chain(system_prompt, _memory=memory)

if st.button("Effacer l'historique des messages"):
    print("Effacement...")
    memory.clear()
    st.session_state.trace_link = None
    st.session_state.run_id = None

def _get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type

for msg in st.session_state.langchain_messages:
    streamlit_type = _get_openai_type(msg)
    avatar = "🧑‍🏫" if streamlit_type == "assistant" else None
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

def _reset_feedback():
    st.session_state.feedback_update = None
    st.session_state.feedback = None


MAX_CHAR_LIMIT = 500

if prompt := st.chat_input(placeholder="Ex: Quels sont les critères d'un bon titre SEO?"):

    if len(prompt) > MAX_CHAR_LIMIT:
        st.warning(f"⚠️ Ta question est trop longue, max {MAX_CHAR_LIMIT} caractères.")
        prompt = None  # Reset the prompt so it doesn't get processed further
    else:
        st.chat_message("user").write(prompt)
        _reset_feedback()
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            message_placeholder = st.empty()
            full_response = ""

            # input_structure = {"input": prompt}
            input_structure = {
                "question": prompt,
                "chat_history": [
                    (msg.type, msg.content)
                    for msg in st.session_state.langchain_messages
                ],
            }

            for chunk in chain.stream(input_structure, config=runnable_config):
                full_response += chunk["answer"]  # Updated to use the 'answer' key
                message_placeholder.markdown(full_response + "▌")
            memory.save_context({"input": prompt}, {"output": full_response})

        message_placeholder.markdown(full_response)

        # The run collector will store all the runs in order. We'll just take the root and then
        # reset the list for next interaction.
    run = run_collector.traced_runs[0]
    run_collector.traced_runs = []
    st.session_state.run_id = run.id
    wait_for_all_tracers()
    # Requires langsmith >= 0.0.19
    url = client.share_run(run.id)
    # Or if you just want to use this internally
    # without sharing
    # url = client.read_run(run.id).url
    st.session_state.trace_link = url


has_chat_messages = len(st.session_state.get("langchain_messages", [])) > 0

if has_chat_messages:
    feedback_option = "thumbs"
else:
    pass

if st.session_state.get("run_id"):
    feedback = streamlit_feedback(
        feedback_type=feedback_option, 
        key=f"feedback_{st.session_state.run_id}",
    )
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string and optional comment
            feedback_record = client.create_feedback(
                st.session_state.run_id,
                feedback_type_str,  # Updated feedback type
                score=score,
                #comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Feedback invalide")
