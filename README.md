# 📚 AI Study Assistant

An AI-powered study assistant that turns study notes and textbooks into useful revision material.

The application allows students to upload PDF or TXT study material and use AI to ask questions, summarise content, extract specific information, generate quizzes, and create flashcards.

## 🚀 Features

### 💬 Ask a Question
Ask questions about the uploaded study material. The application uses semantic search to find the most relevant sections of the document before generating an answer.

### 📝 Summarise
Generate a structured summary of the uploaded document containing important facts, concepts, definitions, explanations, and examples.

### 🔎 Extract Information
Request specific information from the document and have the AI find and organise the relevant content.

### 🧠 Generate Quizzes
Generate revision quizzes based only on the uploaded material.

Supported question types:
- Multiple Choice
- True/False
- Short Answer
- Mixed

Difficulty levels:
- Easy
- Medium
- Hard

### 🗂️ Generate Flashcards
Automatically create study flashcards from the uploaded material using a simple:

**Front → Back**

format.

## ⚙️ How It Works

The application combines document processing, semantic search, vector embeddings, and large language models.

```text
Study Material
      ↓
PDF/TXT Text Extraction
      ↓
Text Cleaning & Chunking
      ↓
Sentence Embeddings
      ↓
FAISS Vector Search
      ↓
Relevant Study Material
      ↓
AI Model
      ↓
Answer / Summary / Quiz / Flashcards
