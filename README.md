

# MCQ-generator-with-student-analysis-using-RAG-and-Agentic-AI

# RAG Test Application

## Overview

The RAG (Retrieval-Augmented Generation) Test Application is a Streamlit-based web application designed to:

- Upload and process PDF documents.
- Generate tailored assessments (questions) based on the document content using LLMs.
- Allow users to take tests, evaluate their performance, and analyze weak areas.
- Provide personalized improvement strategies and track performance over time.

The project integrates advanced machine learning components like LLMs, embeddings, and task-oriented agents to deliver dynamic and adaptive educational insights.

---

## Features

1. **PDF Upload & Processing**: Users can upload PDF files, which are processed and chunked for text extraction.
2. **Test Generation**:
   - Generates questions in a structured CSV format.
   - Includes varied difficulty levels (Easy, Moderate, Hard).
3. **Test Taking**:
   - Users can take generated tests.
   - Performance is evaluated, and results are displayed.
4. **Weak Area Analysis**:
   - Identifies knowledge gaps and misconceptions.
   - Provides a detailed markdown report.
5. **Improvement Strategy**:
   - Suggests personalized learning pathways.
   - Recommends topics, question types, and difficulty levels to focus on.
6. **Performance Tracking**:
   - Stores past test results in a MySQL database.
   - Visualizes trends with interactive Plotly charts.
7. **Identifying Weakness And Strengths Of The User**
   - Using the test results of teh user to identify their strengths and weaknesses

---

## Project Components

### Backend Components

1. **LLM Integration**:
   - Uses Groq's ChatGroq and Ollama Embeddings to generate and embed content.
   - Supports advanced agents with CrewAI for performance diagnostics and learning strategy generation.

2. **Database**:
   - MySQL database stores user test results for analysis.

3. **PDF Processing**:
   - Utilizes `PyPDFLoader` and `RecursiveCharacterTextSplitter` for text extraction and splitting.

### Frontend Components

1. **Streamlit Interface**:
   - Provides a user-friendly interface for uploading documents, taking tests, and viewing analysis.

2. **Interactive Visualizations**:
   - Implements Plotly for trend analysis and performance visualization.

---

## Usage

1. **Upload Document**:
   - Navigate to the "Upload Document" section and upload a PDF.
   - The document will be processed, and its content will be used to generate test questions.

2. **Generate and Take Test**:
   - Generate a test from the document.
   - Answer the questions in the interface.
   - Submit the test to evaluate performance.

3. **Analyze Weak Areas**:
   - Click "Analyze Weak Areas" to get insights into knowledge gaps.

4. **Generate Improvement Strategy**:
   - Click "Generate Improvement Strategy" to receive personalized recommendations.

5. **View Past Test Analysis**:
   - Review past test results and track performance trends.
  
6. **Identifying Weakness And Strengths Of The User**
   - Using the test results of teh user to identify their strengths and weaknesses

---

## Approach

1. **Retrieval-Augmented Generation (RAG)**:
   - Combines document retrieval with LLMs to generate questions and insights directly from the content.

2. **Task-Oriented Agents**:
   - Uses CrewAI agents for domain-specific analysis and strategy generation.

3. **Data-Driven Insights**:
   - Stores and analyzes user performance data to deliver targeted feedback and learning strategies.\

4. **Identifying Weakness And Strengths Of The User**
   - Using the test results of teh user to identify their strengths and weaknesses

---

## Future Enhancements

1. Add support for:
   - Real-time collaboration on test creation.
   - Multi-language document processing.
2. Integrate more advanced visualization dashboards.
3. Expand LLM capabilities with other models like GPT-4 or Claude.

---

## Contact

For inquiries or contributions, reach out to:
- **Name**: Sarthak
- **Email**: sarthak.molu08@.com# Smart-Prep-AI
