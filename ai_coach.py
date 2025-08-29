import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from firebase_admin import firestore
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# --- NEW: Load environment variables from .env file ---
load_dotenv()

# Initialize the LLM (Gemini)
# This will now automatically find the GOOGLE_API_KEY from your .env file
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", verbose=True, temperature=0.7)


# --- Tool Definition (Updated) ---
class UserDataTool(BaseTool):
    name: str = "User Wellness Data Tool"
    description: str = "Fetches a user's complete history, including all yoga sessions and all personal journal entries, from the Firestore database. Input must be the user's ID (uid)."

    def _run(self, user_id: str) -> str:
        try:
            db = firestore.client()

            # 1. Fetch last 10 sessions
            sessions_ref = db.collection("users").document(user_id).collection("sessions").order_by("date",
                                                                                                    direction=firestore.Query.DESCENDING).limit(
                10).stream()
            session_summary = [
                f"- Date: {s.to_dict()['date'].strftime('%Y-%m-%d')}, Total Time: {s.to_dict()['total_time']}s"
                for s in sessions_ref
            ]

            # 2. Fetch last 10 journal entries
            journal_ref = db.collection("users").document(user_id).collection("journal").order_by("date",
                                                                                                  direction=firestore.Query.DESCENDING).limit(
                10).stream()
            journal_summary = [
                f"- Journal on {j.to_dict()['date'].strftime('%Y-%m-%d')}: '{j.to_dict()['entry_text']}' (Sentiment: {j.to_dict()['sentiment']})"
                for j in journal_ref
            ]

            if not session_summary and not journal_summary:
                return "No data found for this user."

            return (
                f"Recent Session History:\n{' '.join(session_summary) if session_summary else 'No recent sessions.'}\n\n"
                f"Recent Journal Entries:\n{' '.join(journal_summary) if journal_summary else 'No recent journal entries.'}")
        except Exception as e:
            return f"Error fetching data from database: {e}"


# --- Agent Definitions (Updated) ---
data_analyst_agent = Agent(
    role='Holistic Wellness Data Analyst',
    goal='Fetch and clearly summarize a user\'s complete wellness profile, including both their physical practice (yoga sessions) and their mental/emotional state (journal entries).',
    backstory='You are an expert data analyst who understands that wellness is more than just physical. You synthesize quantitative session data with qualitative journal entries to create a full picture of the user.',
    verbose=True,
    llm=llm,
    tools=[UserDataTool()]
)

yoga_coach_agent = Agent(
    role='Personal AI Wellness Coach',
    goal='Create personalized, empathetic, and actionable suggestions, plans, and answers to user queries based on their complete wellness profile.',
    backstory='You are "Zen," a world-class yoga and wellness coach. You look at the whole person—their practice, their feelings, and their history—to provide insightful and supportive guidance.',
    verbose=True,
    llm=llm,
)


# --- Task Definitions (Updated) ---
def create_ai_tasks(user_id, user_query="Generate a progress summary and weekly plan."):
    data_analysis_task = Task(
        description=f'Analyze the complete wellness history for the user with ID: {user_id}.',
        expected_output='A concise summary report of the user\'s recent sessions and journal entries.',
        agent=data_analyst_agent
    )

    coach_response_task = Task(
        description=f'''Based on the data analyst's report, address the user's specific query: "{user_query}". 
        If they ask for a plan, provide one. If they ask a question, answer it. 
        Your response must be a valid JSON object with two keys: "response_text" (a conversational, markdown-formatted answer) and "follow_up_questions" (an array of 3 engaging follow-up questions).''',
        expected_output='''A JSON object containing:
        1.  `response_text` (string): A detailed, conversational answer to the user's query, formatted in markdown.
        2.  `follow_up_questions` (array of strings): An array of 3 new, insightful follow-up questions.''',
        agent=yoga_coach_agent,
        context=[data_analysis_task]
    )
    return [data_analysis_task, coach_response_task]


# --- Crew Definition ---
def get_ai_coach_crew(user_id, query):
    tasks = create_ai_tasks(user_id, query)
    return Crew(
        agents=[data_analyst_agent, yoga_coach_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )