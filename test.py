from flask import Flask, render_template, request, redirect, url_for, session, flash  , jsonify
from werkzeug.utils import secure_filename
from langchain_core.documents import Document
from markdown import markdown
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
import os
import plotly.express as px
import json
import plotly
from crewai import Agent, Task, Crew
import mysql.connector
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
import pandas as pd
from io import StringIO
import plotly.express as px
import json
import re
import pymongo
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
# from flask import Flask, session
# from flask_session import Session
from langchain_community.document_loaders import YoutubeLoader

client = pymongo.MongoClient("localhost:27017")
database = client['ai']
collection = database['tests']

data = list(collection.find())  # Convert cursor to list of dicts

# Convert to DataFrame (excluding the MongoDB `_id` field)
q = pd.DataFrame(data).drop(columns=['_id'], errors='ignore')

ids = q['creator_id'].tolist()

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

parser = StrOutputParser()

# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)

db = None

import re

def clean_analysis(analysis_text):
    # Remove asterisks (*) used as bullet points
    analysis_text = re.sub(r'^\* ', '', analysis_text, flags=re.MULTILINE)
    
    # Remove hash symbols (#) used for headings
    analysis_text = re.sub(r'^#+\s*', '', analysis_text, flags=re.MULTILINE)

    return analysis_text

agent = Agent(
    role='Learning Assessment Analyzer',
    goal='Analyze student responses and provide comprehensive feedback on knowledge gaps',
    backstory="""An AI educational assistant specialized in:
    - Pattern recognition in student mistakes
    - Identifying knowledge gaps
    - Providing analysis on student strengths and weaknesses """,
    llm="groq/llama3-70b-8192",
    max_iter=5,
    verbose=True
)

def analyze_student(topic , wq , rq):
    """
    Analyze student work and generate targeted feedback
    
    Parameters:
        student_answers: List of student responses to questions
        subject_area: The academic subject being assessed

    Process:
        1. Analyze each answer for common mistakes
        2. Identify conceptual misunderstandings
        3. Generate 100-word strength/weakness summary
        4. Create targeted practice questions
        5. Suggest personalized learning resources

    Returns:
        Dict containing:
        - Performance analysis
        - Knowledge gap summary
        - Recommended practice questions
        - Learning resource suggestions
    """
    task = Task(
        description=f"""
        Analyze student performance in {topic} , with wrong questions {wq} , right questions being {rq}
        
        Analysis Requirements:
        - Review all submitted answers
        - Identify recurring error patterns
        - Create summary of strengths (50 words)
        - Create summary of weaknesses (50 words)
        
        Output Format:
        - Clear separation of strengths/weaknesses
        - Only summaries of strength and weakness
        - No extra information
        """ , agent = agent , expected_output= "An well seperated strength and weakness summary of the student"
    )

    crew = Crew(
        agents=[agent],
        tasks=[task]
    )

    # Execute the task
    result = crew.kickoff()
 # Pretty-printed string
    return result.raw



agent7 = Agent(
    role='User Performance Diagnostics Specialist',
    goal="""Analyze incorrect questions and identify specific knowledge gaps, learning weaknesses, and improvement areas for users across different topics.""",
    backstory="""You are an advanced diagnostic agent specialized in educational performance analysis. Your primary objective is to:
    - Systematically evaluate user's incorrect responses
    - Identify precise knowledge gaps""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True,
    allow_delegation=False
)



def weak_areas(paragraph, topic):
    performance_task = Task(
        description=f"""Analyze incorrect questions {paragraph} for topic: {topic}
        
        Analysis Requirements:
        - Identify specific knowledge gaps
        - Highlight weak areas in the concept
        
        Diagnostic Dimensions:
        - Conceptual Weakness Mapping
        
        Output Format:
        - Structured markdown report
        - Quantitative performance metrics
        
        Key Focus Areas:
        - Pattern of misconceptions
        - Difficulty level correlation
        - Topic-specific challenge areas""",
        
        agent=agent7,
        expected_output="""Comprehensive performance diagnostic report with identifying all the highlighting all the weak areas"""
    )

    crew = Crew(
        agents=[agent7],
        tasks=[performance_task]
    )

    # Execute the task
    result = crew.kickoff()
 # Pretty-printed string
    return result.raw

agent8 = Agent(
    role='Personalized Learning Strategy Developer',
    goal='Generate targeted, adaptive learning improvement strategies based on individual performance analysis',
    backstory="""An AI-powered learning optimization specialist focused on:
    - Suggest topics to work on 
    - Suggest difficulty levels to work on 
    - Suggest type of questions to work on
    - Designing customized improvement pathways
    - Recommending strategic learning interventions
    - Transforming weaknesses into learning opportunities""",
    llm="groq/llama3-70b-8192",
    max_iter=3,
    verbose=True
)

def generate_improvement_strategy(paragraph, topic):
    improvement_task = Task(
        description=f"""Generate actionable learning improvement strategy for the mistakes in questions with difficulty {paragraph} from {topic}
        
        Strategy Development Requirements:
        - Propose targeted improvement actions
        - Create structured learning roadmap
        
        Recommendation Dimensions:
        1. Topic-specific skill enhancement
        2. Question type practice strategies
        3. Difficulty level progression
        4. Conceptual understanding reinforcement
        """,
        
        agent=agent8,
        expected_output="""Comprehensive improvement strategy with:
        - Precise learning recommendations
        - Targeted practice suggestions
        - Skill development roadmap"""
    )

    crew = Crew(
        agents=[agent8],
        tasks=[improvement_task]
    )

    # Execute the task
    result = crew.kickoff()  # Pretty-printed string
    return result.raw

agent9 = Agent(
   role='Performance Analysis and Skill Profiler',
   goal='Conduct comprehensive user performance evaluation by identifying strengths, weaknesses, and learning potential',
   
   backstory="""Advanced performance diagnostic specialist focused on:
   - Detailed answer pattern analysis
   - Skill competency mapping
   - Personalized performance insights
   - Constructive feedback generation""",

   
   llm="groq/llama3-70b-8192",
   max_iter=3,
   verbose=True
)

def analyze_strength(correct_answers,  topic):
   """
   Generate comprehensive performance analysis
   """
   performance_task = Task(
       description=f"""Analyze user strengths across answers with correct questions with difficulty as {correct_answers}  for {topic} 
       
       Analysis Dimensions:
       - Strengths of the user
       - Answer pattern recognition
       - Skill competency mapping
       - Strength identification
       - Learning potential evaluation""",
       
       agent=agent9,
       expected_output=f"Detailed performance diagnostic report including the strengths of the user in the topic {topic}"
   )




   crew = Crew(
        agents=[agent9],
        tasks=[performance_task]
    )

        # Execute the task
   result = crew.kickoff()
   return result.raw

 # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'instance/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'passwd': 'naruto',
    'database': 'result'
}

DB1_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'passwd': 'naruto',
    'database': 'tests'
}



# Initialize LLM
os.environ['GROQ_API_KEY'] = "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"
llm = ChatGroq(groq_api_key="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", model="llama3-70b-8192")
embedder = OllamaEmbeddings(model="nomic-embed-text")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_db_connection1():
    return mysql.connector.connect(**DB1_CONFIG)

@app.route('/')
def username():
    return render_template('index.html')

@app.route('/code')
def code():
    return render_template('code.html')

@app.route('/set_code', methods=['POST'])
def set_code():
    username = request.form['username']
    if not username:
        flash('Username is required')
        return redirect(url_for('index'))
    
    if username not in ids:
        flash('No such room')
        return redirect(url_for('index'))

    index_value = q.index[q['creator_id'] == f'{username}'].tolist()
    value = index_value[0]
    topic = q['topic_name'].iloc[value]
    session['topic'] = topic
    questions = pd.DataFrame(q['csv_data'].iloc[value])

    session['questions'] = questions.to_dict('records')
    session['current_question'] = 0
    session['answers'] = [None] * len(questions)
    
    session['code'] = username

    return redirect(url_for('name_2'))

@app.route('/name_2')
def name_2():
    return render_template('name_2.html')

@app.route('/set_username_2', methods=['POST'])
def set_username_2():
    name = request.form['username']
    if not name:
        flash('Username is required')
        return redirect(url_for('index'))
    
    table_name = f'{name}_result'
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            test_name VARCHAR(255),
            percentage INT,
            correct INT,
            wrong INT,
            date_of_test DATE,
            tokens INT
        )
    """)
    conn.commit()
    
    
    session['name'] = name
    return redirect(url_for('quiz_2'))

@app.route('/name')
def name():
    return render_template('name.html')

@app.route('/set_username', methods=['POST'])
def set_username():
    username = request.form['username']
    if not username:
        flash('Username is required')
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user table exists, if not create it
    table_name = f'{username}_result'
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            test_name VARCHAR(255),
            percentage INT,
            correct INT,
            wrong INT,
            date_of_test DATE,
            tokens INT
        )
    """)
    conn.commit()
    
    session['username'] = username
    return redirect(url_for('choice'))

@app.route('/link')
def link():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('link.html')

@app.route('/process_upload_link', methods=['POST'])
def process_upload_link():
    global db
    
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        flash('No YouTube link provided')
        return redirect(url_for('link'))
        
        
    video_id = youtube_url.split("youtu.be/")[1].split("?")[0]

# Get transcript directly
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    transcript_text = " ".join([t["text"] for t in transcript_list])

# Create a Document object
    metadata = {"source": youtube_url}
    document = Document(page_content=transcript_text, metadata=metadata)

    documents = [document]
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final = splitter.split_documents(documents)
    
    # Create embeddings and FAISS database
    db = FAISS.from_documents(final, embedder)
    retriever = db.as_retriever()
        

    
    retriever_prompt = """You are a specialized question generator for CSV-formatted assessment for the {context}.

        STRICT OUTPUT FORMAT:
        - Just output the asked questions, nothing else.
        - No headers, introductions, or explanatory text allowed
        - Produce ONLY  in CSV format.

        Question Generation Rules:
        1. Generate 5 unique questions in context of the topic
        2. Each question must:
        - Be distinct in wording
        - Cover different aspects of the topic
        - Avoid repetition
                                
        OUTPUT FORMAT: Generate questions in CSV format with columns: "question","optionA","optionB","optionC","optionD","correct","difficulty"

        STRICT CSV COLUMN REQUIREMENTS THAT NEEDS TO BE FOLLOWED:
        - question
        - optionA
        - optionB
        - optionC
        - optionD
        - correct 
        - difficulty

        Question Characteristics:
        - Test comprehensive understanding

        Difficulty Distribution:
        - 2 Easy, 2 Moderate and 1 Challenging questions

        Note:              
        NO EXTRA TEXT OR HEADLINES OR ENDING LINES
        DO NOT INCLUDE ANY HEADINGS OR EXTRA TEXTS
        Remember that you are a MCQ generating machine and you do not generate anything except that.

        Output Expectation:
        Precise, structured CSV questions with all the listed column names without any supplementary information
                                
        Example Output:
        "question","optionA","optionB","optionC","optionD","correct","difficulty"
        "What is the proper procedure when approaching a blind curve on a wet road?","Maintain current speed but move to the center of lane","Increase speed slightly to maintain momentum","Reduce speed and move to the right side of lane","Apply brakes firmly while straightening the bike","Reduce speed and move to the right side of lane","moderate" """
    
    prompt = ChatPromptTemplate.from_messages([("system", retriever_prompt), ("human", "{input}")])
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    response = chain.invoke({"input": "Generate 5 questions from the context"})
    csv_content = response['answer']
    cleaned_content = re.sub(r"^.*generated questions in CSV format.*\n?", "", csv_content, flags=re.MULTILINE)
    csv_data = StringIO(cleaned_content)
    df = pd.read_csv(csv_data)
    csv_data = df.to_dict(orient="records")

    prompt_2 = ChatPromptTemplate.from_template(
        """Use ONLY the following context to answer the question. 
        If the answer is not in the context, say "I cannot find the answer in the provided document."
        
        Context:
        {context}
        
        Question: {input}
        """
    )   
        
    retriever_2 = db.as_retriever()
    document_chain_2 = create_stuff_documents_chain(llm, prompt_2)
    chain_2 = create_retrieval_chain(retriever_2, document_chain_2)
    
    inpu = "what is this video about in 5 words or less"
    response = chain_2.invoke({"input": inpu})
    topic = response['answer']
    session['topic'] = topic
    
    # Store questions in session
    session['questions'] = df.to_dict('records')
    session['current_question'] = 0
    session['answers'] = [None] * len(df)
    
    

    return redirect(url_for('quiz'))

@app.route('/choice')
def choice():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('choice.html')

@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/process_upload', methods=['POST'])
def process_upload():

    global db

    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('upload'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF and generate questions
        loader = PyPDFLoader(filepath)
        text = loader.load()
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final = splitter.split_documents(text)
        
        # Create embeddings and FAISS database
        db = FAISS.from_documents(final, embedder)
        retriever = db.as_retriever()
        
        # Define prompt and create chain
        retriever_prompt = """You are a specialized question generator for CSV-formatted assessment for the {context}.

        STRICT OUTPUT FORMAT:
        - Just output the asked questions, nothing else.
        - No headers, introductions, or explanatory text allowed
        - Produce ONLY  in CSV format.

        Question Generation Rules:
        1. Generate 5 unique questions in context of the topic
        2. Each question must:
        - Be distinct in wording
        - Cover different aspects of the topic
        - Avoid repetition
                                
        OUTPUT FORMAT: Generate questions in CSV format with columns: "question","optionA","optionB","optionC","optionD","correct","difficulty"

        STRICT CSV COLUMN REQUIREMENTS THAT NEEDS TO BE FOLLOWED:
        - question
        - optionA
        - optionB
        - optionC
        - optionD
        - correct 
        - difficulty

        Question Characteristics:
        - Test comprehensive understanding

        Difficulty Distribution:
        - 2 Easy, 2 Moderate and 1 Challenging questions

        Note:              
        NO EXTRA TEXT OR HEADLINES OR ENDING LINES
        DO NOT INCLUDE ANY HEADINGS OR EXTRA TEXTS
        Remember that you are a MCQ generating machine and you do not generate anything except that.

        Output Expectation:
        Precise, structured CSV questions with all the listed column names without any supplementary information
                                
        Example Output:
        "question","optionA","optionB","optionC","optionD","correct","difficulty"
        "What is the proper procedure when approaching a blind curve on a wet road?","Maintain current speed but move to the center of lane","Increase speed slightly to maintain momentum","Reduce speed and move to the right side of lane","Apply brakes firmly while straightening the bike","Reduce speed and move to the right side of lane","moderate" """
        prompt = ChatPromptTemplate.from_messages([("system", retriever_prompt), ("human", "{input}")])
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate questions
        response = chain.invoke({"input": "Generate 5 questions from the file provided"})
        csv_content = response['answer']
        cleaned_content = re.sub(r"^.*generated questions in CSV format.*\n?", "", csv_content, flags=re.MULTILINE)
        csv_data = StringIO(cleaned_content)
        df = pd.read_csv(csv_data)

        
        prompt_2 = ChatPromptTemplate.from_template(
            """Use ONLY the following context to answer the question. 
            If the answer is not in the context, say "I cannot find the answer in the provided document."

            Context:
            {context}

            Question: {input}

            """
        )   
            
        retriever_2 = db.as_retriever()

        document_chain_2 = create_stuff_documents_chain(llm ,prompt_2 )

        chain_2 = create_retrieval_chain(retriever_2 , document_chain_2)

        inpu = "what is this document about in 5 words or less"

        response = chain_2.invoke({"input" : inpu})

        topic = response['answer']

        session['topic'] = topic
        
        # Store questions in session
        session['questions'] = df.to_dict('records')
        session['current_question'] = 0
        session['answers'] = [None] * len(df)
        
        os.remove(filepath)  # Clean up uploaded file
        return redirect(url_for('quiz'))
    
    flash('Invalid file type')
    return redirect(url_for('upload'))

@app.route('/quiz')
def quiz():
    if 'questions' not in session:
        return redirect(url_for('upload'))
    
    current_q = session['current_question']
    questions = session['questions']
    if current_q >= len(questions):
        return redirect(url_for('results'))
    
    return render_template('quiz.html', 
                         question=questions[current_q],
                         question_number=current_q + 1,
                         total_questions=len(questions))

@app.route('/quiz_2')
def quiz_2():
    if 'questions' not in session:
        return redirect(url_for('upload'))
    
    current_q = session['current_question']
    questions = session['questions']
    if current_q >= len(questions):
        return redirect(url_for('results_2'))
    
    return render_template('quiz_2.html', 
                         question=questions[current_q],
                         question_number=current_q + 1,
                         total_questions=len(questions))

@app.route('/answer', methods=['POST'])
def answer():
    answer = request.form.get('answer')
    current_q = session['current_question']
    session['answers'][current_q] = answer
    session['current_question'] = current_q + 1
    return redirect(url_for('quiz'))

@app.route('/answer_2', methods=['POST'])
def answer_2():
    answer = request.form.get('answer')
    current_q = session['current_question']
    session['answers'][current_q] = answer
    session['current_question'] = current_q + 1
    return redirect(url_for('quiz_2'))


# Add these imports at the top of your file


@app.route('/analyze_weak_areas', methods=['GET'])
def analyze_weak_areas():
    if 'questions' not in session or 'answers' not in session:
        return redirect(url_for('upload'))
    
    questions = session['questions']
    answers = session['answers']
    wrong_questions = []
    
    # Build list of wrong questions with their details
    for q, a in zip(questions, answers):
        if q['correct'] != a:
            wrong_questions.append({
                'question': q['question'],
                'difficulty': q['difficulty']
            })
    
    # Store wrong questions in session for template access
    session['wrong_questions'] = wrong_questions
    
    if not wrong_questions:
        flash('No incorrect questions found to analyze.')
        return redirect(url_for('results'))
    
    # Create paragraph for analysis
    paragraph = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
                          for q in wrong_questions])
    
    try:
        topic = session['topic']
        result = weak_areas(paragraph, topic)
        result = markdown(result)
        session['analysis_result'] = result
        session['analysis_type'] = 'Weak Areas Analysis'
        return render_template('analyze_weak_areas.html',
                             analysis_result=result,
                             analysis_type='Weak Areas Analysis',
                             wrong_questions=wrong_questions)
    except Exception as e:
        flash(f'Error during analysis: {str(e)}')
        return redirect(url_for('results'))

@app.route('/generate_strategy', methods=['GET'])
def generate_strategy():
    if 'questions' not in session or 'answers' not in session:
        return redirect(url_for('upload'))
    
    questions = session['questions']
    answers = session['answers']
    wrong_questions = []
    
    # Build list of wrong questions with their details
    for q, a in zip(questions, answers):
        if q['correct'] != a:
            wrong_questions.append({
                'question': q['question'],
                'difficulty': q['difficulty']
            })
    
    # Store wrong questions in session for template access
    session['wrong_questions'] = wrong_questions
    

    if not wrong_questions:
        flash('No incorrect questions found to analyze.')
        return redirect(url_for('results'))
    
    # Create paragraph for analysis
    paragraph = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
                          for q in wrong_questions])
    
    try:
        topic = session['topic']
        result = generate_improvement_strategy(paragraph, topic)
        result = markdown(result)
        session['strategy'] = result
        session['strategy_type'] = 'Improvement Strategy'
        return render_template('strategy.html',
                             strategy=result,
                             strategy_type='Improvement Strategy',
                             wrong_questions=wrong_questions)
    except Exception as e:
        flash(f'Error during analysis: {str(e)}')
        return redirect(url_for('results'))
    

@app.route('/generate_strengths', methods=['GET'])
def generate_strengths():
    if 'questions' not in session or 'answers' not in session:
        return redirect(url_for('upload'))
    
    questions = session['questions']
    answers = session['answers']
    wrong_questions = []
    
    # Build list of wrong questions with their details
    for q, a in zip(questions, answers):
        if q['correct'] == a:
            wrong_questions.append({
                'question': q['question'],
                'difficulty': q['difficulty']
            })
    
    # Store wrong questions in session for template access
    session['wrong_questions'] = wrong_questions
    
    if not wrong_questions:
        flash('No ncorrect questions found to analyze.')
        return redirect(url_for('results'))
    
    # Create paragraph for analysis
    paragraph = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
                          for q in wrong_questions])
    
    try:
        topic = session['topic']
        result = analyze_strength(paragraph, topic)
        result = markdown(result)
        session['strategy'] = result
        session['strategy_type'] = 'Analyze Strengths'
        return render_template('strength.html',
                             strategy=result,
                             strategy_type='Analyze Strengths',
                             wrong_questions=wrong_questions)
    except Exception as e:
        flash(f'Error during analysis: {str(e)}')
        return redirect(url_for('results'))

def chatbot(input, result):
    prompt_2 = ChatPromptTemplate.from_template(
    """You are an educational assessment specialist who analyzes student feedback and resolves their doubts. Your goal is to help students improve through specific, actionable guidance.

    Student feedback = {result}

    Your tasks:
    - Identify key strengths and weaknesses
    - Spot patterns in mistakes
    - Suggest improvement strategies
    - Create targeted practice questions

    Response Guidelines:
    - Be encouraging and supportive
    - Provide clear, actionable feedback
    - Focus on the most important concepts
    - Maintain clarity and brevity
    - Respond only if the input is relevant to student learning, feedback, or doubts
    - Ignore greetings, small talk, or nonsensical inputs unless they are part of a genuine learning inquiry

    Context:
    {context}

    Student Query: {input}

    If the query is vague, off-topic, or irrelevant, politely ask the student to provide more details related to their learning."""
)
  

    retriever_3 = db.as_retriever()
    document_chain_3 = create_stuff_documents_chain(llm, prompt_2)
    chain_3 = create_retrieval_chain(retriever_3, document_chain_3)
    response = chain_3.invoke({"input": input, "result": result})
    
    # Extract the answer from the response dictionary
    if isinstance(response, dict) and "answer" in response:
        answer_text = response["answer"]
        # Only parse if it's a string
        if isinstance(answer_text, str):
            try:
                parsed_response = parser.invoke(answer_text)
                return parsed_response
            except Exception as e:
                # If parsing fails, return the raw answer
                return {"response": answer_text}
        else:
            # If answer is not a string, convert appropriately
            return {"response": str(answer_text)}
    
    # If the response doesn't have the expected structure
    return {"response": str(response)}

app.static_folder = 'static'

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/response")
def get_bot_response():
    userText = request.args.get('msg')
    questions = session['questions']
    answers = session['answers']
    wrong_questions = []
    
    # Build list of wrong questions with their details
    for q, a in zip(questions, answers):
        if q['correct'] != a:
            wrong_questions.append({
                'question': q['question'],
                'difficulty': q['difficulty']
            })
    
    # Store wrong questions in session for template access
    session['wrong_questions'] = wrong_questions

    
    # Create paragraph for analysis
    wrong = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
                          for q in wrong_questions])

    right_questions = []
    
    # Build list of wrong questions with their details
    for q, a in zip(questions, answers):
        if q['correct'] == a:
            right_questions.append({
                'question': q['question'],
                'difficulty': q['difficulty']
            })
    
    # Store wrong questions in session for template access
    session['right_questions'] = right_questions
    
    # Create paragraph for analysis
    right = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
                          for q in right_questions])    
    
    topic = session['topic']
    result = analyze_student(topic, wrong , right)
    response = chatbot(userText, result)  
    response = markdown(response)# Removes *, #, @, and !
    return jsonify(response)




# @app.route("/chat")
# def chat():
#     print("db exists globally:", 'db' in globals())
#     return render_template("chat.html")
# @app.route("/response")
# def get_bot_response():
#     userText = request.args.get('msg')
#     # if 'questions' not in session or 'answers' not in session:
#     #     return redirect(url_for('upload'))
    
#     # questions = session['questions']
#     # answers = session['answers']
#     # wrong_questions = []
    
#     # # Build list of wrong questions with their details
#     # for q, a in zip(questions, answers):
#     #     if q['correct'] != a:
#     #         wrong_questions.append({
#     #             'question': q['question'],
#     #             'difficulty': q['difficulty']
#     #         })
    
#     # # Store wrong questions in session for template access
#     # session['wrong_questions'] = wrong_questions

    
#     # # Create paragraph for analysis
#     # wrong = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
#     #                       for q in wrong_questions])

#     # right_questions = []
    
#     # # Build list of wrong questions with their details
#     # for q, a in zip(questions, answers):
#     #     if q['correct'] == a:
#     #         right_questions.append({
#     #             'question': q['question'],
#     #             'difficulty': q['difficulty']
#     #         })
    
#     # # Store wrong questions in session for template access
#     # session['right_questions'] = right_questions
    
#     # # Create paragraph for analysis
#     # right = ". ".join([f"{q['question']} (Difficulty: {q['difficulty']})" 
#     #                       for q in right_questions])    
    

#     # result = analyze_student(session.get('topic', 'the test'), wrong , right)
#     # return chatbot(userText , result)
#     return userText




@app.route('/graph', methods=['GET'])  # Ensure GET is allowed
def graph():
        
    conn = get_db_connection()
    cursor = conn.cursor()

        
    try:
        if 'username' in session:
            table_name = f"{session['username']}_result"
        if 'name' in session:
            table_name = f"{session['name']}_result"
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        data = cursor.fetchall()
        column_names = [i[0] for i in cursor.description]
        df = pd.DataFrame(data, columns=column_names)

        if 'percentage' not in df.columns or 'date_of_test' not in df.columns:
            flash("Required data columns not found!")
            return redirect(url_for('results'))
                
        date = df['date_of_test'].astype(str).tolist()
        percentage = df['percentage'].tolist()

        return render_template("graph.html", labels=date, values=percentage)
        
    except Exception as e:
        flash(f'Error during analysis: {str(e)}')
        return redirect(url_for('results'))
    finally:
        cursor.close()
        conn.close()

# Update the results route to include analysis results
@app.route('/results_2')
def results_2():
    if 'answers' not in session:
        return redirect(url_for('upload'))
    
    questions = session['questions']
    answers = session['answers']
    correct_count = 0
    wrong_answers = []
    l = ['correct' , 'wrong']
    l= [str(a) for a in l]
    c1 = 0
    c2 = 0
    c3 = 0
    w1 = 0
    w2 = 0
    w3 = 0
    
    for q, a in zip(questions, answers):
        if q['correct'] == a:
            correct_count += 1

        else:
            wrong_answers.append({
                'question': q['question'],
                'your_answer': a,
                'correct_answer': q['correct']
            })

    

    for q, a in zip(questions, answers):
        if q['correct'] == a:
            if q["difficulty"] == "easy":
                c1 += 1
            elif q["difficulty"] == "moderate":
                c2 += 1
            else:
                c3 += 1
        else:
            if q["difficulty"] == "easy":
                w1 += 1
            elif q["difficulty"] == "moderate":
                w2 += 1
            else:
                w3 += 1
    
    data = {
        "Easy": [c1, w1],
        "Medium": [c2, w2],  # Changed from "Category B" to match difficulty level
        "Hard": [c3, w3]     # Changed from "Category C" to match difficulty level
    }
    
   

    percentage = (correct_count / len(questions)) * 100


    # Save results to database
    conn = get_db_connection()
    conn1 = get_db_connection1()
    cursor1 = conn1.cursor()
    tn = session['code']  #table name 2
    query1 = f"INSERT INTO {tn} (Name, Marks) VALUES (%s, %s)"
    N = session['name']
    values = (N , correct_count)
    cursor1.execute(query1, values)
    conn1.commit()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    table_name = f"{session['name']}_result"
    topic = session['topic']
    query = f"INSERT INTO {table_name} (test_name, percentage, correct, wrong, date_of_test, tokens) VALUES (%s, %s, %s, %s, %s , %s)"
    values = (topic, percentage, correct_count, len(questions) - correct_count, today, correct_count)
    cursor.execute(query, values)
    conn.commit()
    return render_template('results_2.html',
                         percentage=percentage,
                         correct_count=correct_count,
                         total_questions=len(questions),
                         wrong_answers=wrong_answers , # Ensure labels is always defined
                         data=data
                         ,labels = l  )



@app.route('/results')
def results():
    if 'answers' not in session:
        return redirect(url_for('upload'))
    
    questions = session['questions']
    answers = session['answers']
    correct_count = 0
    wrong_answers = []
    l = ['correct' , 'wrong']
    l= [str(a) for a in l]
    c1 = 0
    c2 = 0
    c3 = 0
    w1 = 0
    w2 = 0
    w3 = 0
    
    for q, a in zip(questions, answers):
        if q['correct'] == a:
            correct_count += 1

        else:
            wrong_answers.append({
                'question': q['question'],
                'your_answer': a,
                'correct_answer': q['correct']
            })

    

    for q, a in zip(questions, answers):
        if q['correct'] == a:
            if q["difficulty"] == "easy":
                c1 += 1
            elif q["difficulty"] == "moderate":
                c2 += 1
            else:
                c3 += 1
        else:
            if q["difficulty"] == "easy":
                w1 += 1
            elif q["difficulty"] == "moderate":
                w2 += 1
            else:
                w3 += 1
    
    data = {
        "Easy": [c1, w1],
        "Medium": [c2, w2],  # Changed from "Category B" to match difficulty level
        "Hard": [c3, w3]     # Changed from "Category C" to match difficulty level
    }
    
   

    percentage = (correct_count / len(questions)) * 100


    # Save results to database
    conn = get_db_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    table_name = f"{session['username']}_result"
    topic = session['topic']
    query = f"INSERT INTO {table_name} (test_name, percentage, correct, wrong, date_of_test, tokens) VALUES (%s, %s, %s, %s, %s , %s)"
    values = (topic, percentage, correct_count, len(questions) - correct_count, today, correct_count)
    cursor.execute(query, values)
    conn.commit()
    return render_template('results.html',
                         percentage=percentage,
                         correct_count=correct_count,
                         total_questions=len(questions),
                         wrong_answers=wrong_answers , # Ensure labels is always defined
                         data=data
                         ,labels = l  )

if __name__ == '__main__':
    app.run(debug=True,port= 5500, use_reloader=False)

# prompt_2 = ChatPromptTemplate.from_template(
# """You are an educational assessment specialist who analyzes student feedback and solve their doubts. Your goal is to help students improve through specific, actionable guidance.

# Student feedback = {result}

# Review student responses to:
# - Identify key strengths and weaknesses
# - Spot patterns in mistakes
# - Suggest improvement strategies
# - Create targeted practice questions


# Guidelines:
# - Be encouraging and supportive
# - Provide clear, actionable feedback
# - Focus on most important concepts
# - Maintain clarity and brevity."

# Context:
# {context}

# Question: {input}

# """
# )   


# userText = "Hi how are you "
# result = "Failed in maths "
# retriever_3 = db.as_retriever()
# document_chain_3 = create_stuff_documents_chain(llm ,prompt_2 )
# chain_3 = create_retrieval_chain(retriever_3 , document_chain_3)
# a = chain_3.invoke({"input": userText , "result" : result })
# print(a)

