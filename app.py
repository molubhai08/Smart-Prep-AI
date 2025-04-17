from flask import Flask, render_template, request, redirect, url_for, session, flash  , jsonify
from werkzeug.utils import secure_filename
from markdown import markdown
import os
import json
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from io import StringIO
import re
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import base64
import os
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.output_parsers import StrOutputParser
import pymongo


client = pymongo.MongoClient("localhost:27017")
db = client['chat']
collection = db['history']

parser = StrOutputParser()



os.environ['GROQ_API_KEY'] = "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"

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

os.environ['GROQ_API_KEY'] = "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"
llm = ChatGroq(groq_api_key="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", model="llama3-70b-8192")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-image', methods=['POST' ])
def analyze_image():
    global analysis 
    # Get the JSON data from the request
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    data = request.get_json()
    
    # The cropped image data is in the "image" field of the JSON
    cropped_image = data.get('image')
    
    if not cropped_image:
        return jsonify({'error': 'No image provided'}), 400
    
    # If the image data is in data URL format, extract just the base64 part
    if cropped_image.startswith('data:'):
        # Split by the comma - the base64 data comes after the comma
        base64_data = cropped_image.split(',')[1]
    else:
        base64_data = cropped_image
    
    # Now use the base64 data correctly in the Groq API call
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": """Analyze this image thoroughly and provide:
                        
                        1. A summary of any text content in the image
                        2. Description of visual elements and layout of the image
                        3. Detailed transcription of any tables with their structure and data
                        4. Any charts, graphs, or diagrams with their key information
                        
                        Separate each section clearly in your response."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    analysis = chat_completion.choices[0].message.content

    return redirect(url_for('chat'))
    

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/response")
def get_bot_response():
    userText = request.args.get('msg')
    p = f"""You are a helpful assistant.

You have been provided with detailed image analysis data that includes:
1. Text summaries extracted from images
2. Descriptions of visual elements present in images
3. Transcriptions of tables from images
4. Interpretations of charts, graphs, and diagrams

image analysis = {analysis}

Your primary role is to act as a doubt solver, helping users understand and utilize this information by:
- Drawing from the image analysis data to answer specific questions
- Explaining concepts related to the analyzed content
- Connecting information across different parts of the analysis
- Providing additional context when helpful
- Clarifying complex elements from the analyzed images

When responding to queries:
- Reference the relevant parts of the image analysis to support your answers
- Be precise about which elements of the analysis you're referring to
- Maintain a helpful, patient tone focused on resolving the user's doubts
- Ask clarifying questions when needed to better understand what specific information from the analysis would be most helpful

Ensure all interactions are respectful, informative, and focused on helping the user understand the content that has been analyzed from their images."""

    messages = [
        SystemMessage(content=p),
        HumanMessage(content=userText)
    ]
    result = llm.invoke(messages)
    result = parser.invoke(result)
    response = markdown(result)# Removes *, #, @, and !
    return jsonify(response)


    

    # Now you can process the image (e.g., pass it to your model or save it)




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)