import os
import base64
import time
import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import html

# LangChain and OpenAI imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configure folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Session storage
sessions = {}

# PDF Processing Functions
def get_pdf_text(pdf_paths):
    """Extract text from PDF files"""
    text = ""
    for pdf_path in pdf_paths:
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create FAISS vectorstore from text chunks"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain"""
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )
    return conversation_chain

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': False,
            'upload_completed_time': None
        }
    
    return jsonify({
        'success': True,
        'pdf_files': [f['filename'] for f in sessions[session_id]['pdf_files']],
        'feedback_submitted': sessions[session_id]['feedback_submitted'],
        'chat_active': sessions[session_id]['chat_active']
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': False,
            'upload_completed_time': None
        }
    
    # Clear existing files
    for file_info in sessions[session_id]['pdf_files']:
        filepath = os.path.join(UPLOAD_FOLDER, file_info['filename'])
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file: {e}")
    
    sessions[session_id]['pdf_files'] = []
    sessions[session_id]['chat_active'] = False
    sessions[session_id]['vectorstore'] = None
    sessions[session_id]['conversation_chain'] = None
    
    uploaded_files = []
    files = request.files.getlist('files')
    saved_paths = []
    
    # Save all uploaded PDF files
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            saved_paths.append(filepath)
            
            sessions[session_id]['pdf_files'].append({
                'filename': filename,
                'original_name': file.filename
            })
            
            uploaded_files.append({
                'filename': filename,
                'original_name': file.filename
            })
    
    # Process PDFs and create conversation chain
    if saved_paths:
        try:
            raw_text = get_pdf_text(saved_paths)
            if raw_text.strip():
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation_chain = get_conversation_chain(vectorstore)
                
                sessions[session_id]['vectorstore'] = vectorstore
                sessions[session_id]['conversation_chain'] = conversation_chain
                sessions[session_id]['chat_active'] = True
                
                # Add welcome message
                sessions[session_id]['messages'].append({
                    'role': 'assistant',
                    'content': 'PDFs processed successfully! How can I help you with the documents?',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No text could be extracted from the PDFs'
                })
        except Exception as e:
            print(f"PDF processing error: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to process PDFs: {str(e)}'
            })
    
    upload_time = time.time()
    sessions[session_id]['last_interaction'] = upload_time
    sessions[session_id]['upload_completed_time'] = upload_time
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'chat_active': sessions[session_id]['chat_active'],
        'upload_completed_time': upload_time,
        'welcome_message': 'PDFs processed successfully! How can I help you with the documents?' if sessions[session_id]['chat_active'] else None
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    is_voice_input = data.get('is_voice_input', False)
    
    if session_id not in sessions:
        return jsonify({
            'error': 'Session not found',
            'response': 'Please upload PDF files first.'
        })
    
    # Check if chat is active
    if not sessions[session_id]['chat_active']:
        return jsonify({
            'error': 'Chat not active',
            'response': 'Please upload PDF files to start the conversation.'
        })
    
    # Update last interaction time
    sessions[session_id]['last_interaction'] = time.time()
    
    try:
        # Add user message to session
        sessions[session_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Normalize message
        normalized_message = message.lower().strip().replace("'", "").replace(",", "").replace(".", "")
        
        # Check for negative/dismissive responses
        negative_responses = ['no', 'nope', 'nah', 'not needed', 'no need', 'no thanks', 
                             'not really', 'im good', "i'm good", 'all good', 'thats all', 
                             "that's all", 'nothing else', 'nothing more']
        
        is_negative = any(neg in normalized_message for neg in negative_responses)
        
        # Track consecutive "no" responses
        if is_negative:
            sessions[session_id]['consecutive_no_count'] = sessions[session_id].get('consecutive_no_count', 0) + 1
        else:
            sessions[session_id]['consecutive_no_count'] = 0
        
        # End session after 2 consecutive "no" responses
        if sessions[session_id]['consecutive_no_count'] >= 2:
            bot_response = "Thank you for using PDF Assistant! Have a great day!"
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            sessions[session_id]['consecutive_no_count'] = 0
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': True,
                'trigger_feedback': True
            })
        
        # Handle greetings
        greetings = ['hello', 'hi', 'hii', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greet in normalized_message for greet in greetings) and len(normalized_message.split()) <= 3:
            bot_response = 'Hello! How can I assist you with the PDF documents today?'
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': False
            })
        
        # Handle goodbyes
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'exit', 'quit']
        if any(goodbye in normalized_message for goodbye in goodbyes):
            bot_response = 'Thank you for using PDF Assistant! We hope to see you again soon.'
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': True,
                'trigger_feedback': True
            })
        
        # Check for acknowledgments
        acknowledgments = [
            'ok', 'okay', 'okey', 'oke', 'k',
            'nice', 'good', 'great', 'excellent', 'awesome', 'perfect', 'cool', 'fine',
            'thanks', 'thank you', 'thankyou', 'thx', 'ty',
            'alright', 'got it', 'understood', 'i see', 'i understand',
            'yes', 'yeah', 'yep', 'yup', 'sure', 'of course'
        ]
        
        acknowledgments.extend(negative_responses)
        
        is_acknowledgment = (
            normalized_message in acknowledgments or
            (len(normalized_message.split()) <= 3 and any(ack in normalized_message for ack in acknowledgments))
        ) and not any(question_word in normalized_message for question_word in [
            'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 
            'would', 'should', 'is', 'are', 'does', 'do', 'explain', 'tell', 'show', 'describe'
        ])
        
        # Handle acknowledgments with brief responses
        if is_acknowledgment:
            if is_negative:
                bot_response = "Understood. Let me know if you need anything else."
            else:
                bot_response = "You're welcome! Feel free to ask if you need anything else or have questions about the documents."
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': False
            })
        
        # Get response from conversation chain
        conversation_chain = sessions[session_id]['conversation_chain']
        
        if conversation_chain:
            response = conversation_chain({'question': message})
            bot_response = response['answer']
            
            # Store as last analysis
            sessions[session_id]['last_analysis'] = bot_response
            sessions[session_id]['awaiting_followup'] = True
        else:
            bot_response = "I apologize, but the conversation chain is not initialized. Please try uploading the PDFs again."
        
        # Add bot message to session
        sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'is_voice_input': is_voice_input,
            'feedback_submitted': sessions[session_id]['feedback_submitted'],
            'session_ended': False
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': str(e),
            'response': 'An error occurred while processing your request.'
        })

@app.route('/export/json', methods=['POST'])
def export_json():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        return jsonify({
            'session_id': session_id,
            'messages': sessions[session_id]['messages'],
            'pdf_files': [f['filename'] for f in sessions[session_id]['pdf_files']]
        })
    else:
        return jsonify({'error': 'Session not found'})

@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions or not sessions[session_id]['messages']:
        return jsonify({'error': 'No chat history found'}), 404
    
    try:
        pdf_filename = f'chat_export_{session_id}_{int(time.time())}.pdf'
        pdf_path = os.path.join(STATIC_FOLDER, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#2c3e50',
            spaceAfter=30
        )
        
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2980b9',
            leftIndent=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        bot_style = ParagraphStyle(
            'BotMessage',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#34495e',
            leftIndent=20,
            spaceAfter=15
        )
        
        story = []
        story.append(Paragraph("PDF Assistant - Chat Export", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        for msg in sessions[session_id]['messages']:
            if msg['role'] == 'user':
                story.append(Paragraph(f"<b>You:</b> {html.escape(msg['content'])}", user_style))
            else:
                content = msg['content'].replace('**', '')
                story.append(Paragraph(f"<b>Assistant:</b> {html.escape(content)}", bot_style))
        
        doc.build(story)
        
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_data,
            'filename': pdf_filename
        })
    
    except Exception as e:
        print(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        # Delete uploaded files
        for file_info in sessions[session_id]['pdf_files']:
            filepath = os.path.join(UPLOAD_FOLDER, file_info['filename'])
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        # Clear session data
        sessions[session_id]['messages'] = []
        sessions[session_id]['pdf_files'] = []
        sessions[session_id]['vectorstore'] = None
        sessions[session_id]['conversation_chain'] = None
        sessions[session_id]['chat_active'] = False
        sessions[session_id]['last_interaction'] = time.time()
        sessions[session_id]['last_analysis'] = None
        sessions[session_id]['awaiting_followup'] = False
        sessions[session_id]['consecutive_no_count'] = 0
        
        return jsonify({'success': True})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    session_id = data.get('session_id')
    rating = data.get('rating')
    comment = data.get('comment', '')
    
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': False
        }
    
    feedback_entry = {
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    sessions[session_id]['feedback'].append(feedback_entry)
    sessions[session_id]['feedback_submitted'] = True
    sessions[session_id]['last_interaction'] = time.time()
    
    return jsonify({
        'success': True,
        'feedback_submitted': True
    })

@app.route('/check_idle', methods=['POST'])
def check_idle():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        current_time = time.time()
        last_interaction = sessions[session_id]['last_interaction']
        upload_completed_time = sessions[session_id].get('upload_completed_time')
        
        idle_time = current_time - last_interaction
        
        # 10 seconds after upload, 7 seconds otherwise
        if upload_completed_time and (current_time - upload_completed_time) < 15:
            idle_threshold = 10
        else:
            idle_threshold = 7
            if upload_completed_time:
                sessions[session_id]['upload_completed_time'] = None
        
        if idle_time >= idle_threshold:
            return jsonify({
                'is_idle': True,
                'idle_time': idle_time
            })
        else:
            return jsonify({
                'is_idle': False,
                'idle_time': idle_time
            })
    else:
        return jsonify({
            'is_idle': False,
            'idle_time': 0
        })

@app.route('/export/feedback', methods=['POST'])
def export_feedback():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions and sessions[session_id]['feedback']:
        csv_data = "Timestamp,Rating,Comment\n"
        for fb in sessions[session_id]['feedback']:
            csv_data += f"{fb['timestamp']},{fb['rating']},\"{fb['comment']}\"\n"
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'feedback_{session_id}.csv'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No feedback data available'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)