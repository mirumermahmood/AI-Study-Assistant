import streamlit as st
from openai import OpenAI
import PyPDF2
import io
import os
import re
import time
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="📃", layout="centered")

st.title("AI Study Assistant")

st.markdown("Upload your notes and either summarise the entire document, ask questions, extract information, generate a quiz, or generate flashcards.")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

API_KEY = os.getenv("API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

file_upload = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"])

mode = st.selectbox(
    "What would you like to do?",
    [
        "Ask a question",
        "Summarise",
        "Extract information",
        "Generate a quiz",
        "Generate flashcards"
    ]
)

run = st.button("Run")


def extract_from_pdf(file_upload):
    reader = PyPDF2.PdfReader(file_upload)
    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_from_file(file_upload):
    if file_upload.type == "application/pdf":
        return extract_from_pdf(io.BytesIO(file_upload.read()))

    return file_upload.read().decode("utf-8")


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=7000, overlap=500):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            step = chunk_size - overlap

            for i in range(0, len(paragraph), step):
                chunks.append(paragraph[i:i + chunk_size])

        elif len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def embed(chunks, embeddings, value, max_chars=16000):
    index = create_faiss_index(embeddings)

    question_embedding = embedding_model.encode([value])
    question_embedding = np.array(question_embedding).astype("float32")

    k = min(5, len(chunks))

    distances, indexes = index.search(question_embedding, k)

    relevant_chunks = []
    current_length = 0

    for i in indexes[0]:

        chunk = chunks[i]

        if current_length + len(chunk) > max_chars:
            remaining = max_chars - current_length

            if remaining > 500:
                relevant_chunks.append(chunk[:remaining])

            break

        relevant_chunks.append(chunk)
        current_length += len(chunk)

    return "\n\n".join(relevant_chunks)


def select_chunks(chunks, embeddings, number_chunks=5, max_chars=16000):
    index = create_faiss_index(embeddings)

    document_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    document_embedding = document_embedding.astype("float32")

    k = min(number_chunks, len(chunks))

    distances, indexes = index.search(document_embedding, k)

    selected_chunks = []
    current_length = 0

    for i in indexes[0]:

        chunk = chunks[i]

        if current_length + len(chunk) > max_chars:
            remaining = max_chars - current_length

            if remaining > 500:
                selected_chunks.append(chunk[:remaining])

            break

        selected_chunks.append(chunk)
        current_length += len(chunk)

    return selected_chunks


@st.cache_data
def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


def safe_api_call(model, messages, max_completion_tokens, wait_time=2):
    time.sleep(wait_time)

    return client.chat.completions.create(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=messages
    )


def generate_answer(content, value, mode):
    response = safe_api_call(
        model="openai/gpt-oss-120b",
        max_completion_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert study assistant.

The user wants to {mode.lower()}.

Use ONLY the provided study material.

Do not invent information.

Give a clear, accurate and useful answer.

If the material does not contain enough information to answer the request, say so.

For an extraction request, extract the requested information directly and organise it clearly.

For a question, explain the answer clearly enough for a student to understand it.
"""
            },
            {
                "role": "user",
                "content": f"""
User request:
{value}

Relevant study material:
{content}
"""
            }
        ],
        wait_time=2
    )

    return response.choices[0].message.content


def summarise_chunk(chunk):
    response = safe_api_call(
        model="openai/gpt-oss-20b",
        max_completion_tokens=700,
        messages=[
            {
                "role": "system",
                "content": """
Summarise the provided study material.

Include:
- Important facts
- Important concepts
- Definitions
- Important explanations
- Important examples

Do not add information that is not present.

Remove unnecessary repetition.

Use clear headings and bullet points.

Return only the summary.
"""
            },
            {
                "role": "user",
                "content": chunk
            }
        ],
        wait_time=3
    )

    return response.choices[0].message.content


def generate_summary(chunks):
    summaries = []

    progress = st.progress(0)
    status = st.empty()

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        status.write(f"Summarising section {i + 1} of {total}...")

        summary = summarise_chunk(chunk)
        summaries.append(summary)

        progress.progress((i + 1) / total)

    progress.empty()
    status.empty()

    combined_summaries = "\n\n".join(summaries)

    while len(combined_summaries) > 14000:

        smaller_chunks = chunk_text(
            combined_summaries,
            chunk_size=6000,
            overlap=300
        )

        new_summaries = []

        progress = st.progress(0)
        status = st.empty()

        total = len(smaller_chunks)

        for i, chunk in enumerate(smaller_chunks):

            status.write(f"Condensing summary section {i + 1} of {total}...")

            summary = summarise_chunk(chunk)
            new_summaries.append(summary)

            progress.progress((i + 1) / total)

        progress.empty()
        status.empty()

        combined_summaries = "\n\n".join(new_summaries)

    response = safe_api_call(
        model="openai/gpt-oss-20b",
        max_completion_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": """
Combine the provided section summaries into ONE coherent study summary.

Requirements:
- Remove repetition.
- Keep important facts.
- Keep definitions.
- Keep important explanations.
- Keep important examples.
- Do not add new information.
- Organise the material logically.
- Use clear headings and bullet points.
- Make it useful for revision.

Return only the final summary.
"""
            },
            {
                "role": "user",
                "content": combined_summaries
            }
        ],
        wait_time=3
    )

    return response.choices[0].message.content


def generate_quiz_batch(content, number_questions, difficulty, quiz_type):
    response = safe_api_call(
        model="openai/gpt-oss-20b",
        max_completion_tokens=800,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an expert study quiz generator.

Create exactly {number_questions} questions.

Difficulty:
{difficulty}

Question type:
{quiz_type}

Use ONLY the provided study material.

Do not invent information.

Cover different concepts.

Avoid repetition.

For Multiple Choice use:

Question 1: [question]

A) [option]
B) [option]
C) [option]
D) [option]

Answer: [correct option]

For True/False use:

Question 1: [statement]

Answer: True/False

For Short Answer use:

Question 1: [question]

Answer: [answer]

For Mixed, use a mixture of the formats above.

Always number every question.

Return ONLY the questions.
"""
            },
            {
                "role": "user",
                "content": content
            }
        ],
        wait_time=4
    )

    return response.choices[0].message.content


def extract_questions(text):
    pattern = r"(?i)(?:^|\n)\s*(?:Question\s*)?(\d+)\s*[:.)-]\s*"

    matches = list(re.finditer(pattern, text))

    questions = []

    if not matches:
        return questions

    for i, match in enumerate(matches):

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        question = text[start:end].strip()

        if question:
            questions.append(question)

    return questions


def generate_quiz(chunks, embeddings, number_questions, difficulty, quiz_type):
    selected_chunks = select_chunks(
        chunks,
        embeddings,
        number_chunks=min(5, len(chunks)),
        max_chars=14000
    )

    content = "\n\n".join(selected_chunks)

    batch_size = 3
    all_questions = []

    while len(all_questions) < number_questions:

        remaining = number_questions - len(all_questions)
        questions_this_batch = min(batch_size, remaining)

        quiz = generate_quiz_batch(
            content,
            questions_this_batch,
            difficulty,
            quiz_type
        )

        questions = extract_questions(quiz)

        for question in questions:

            if len(all_questions) >= number_questions:
                break

            all_questions.append(question)

        if not questions:
            break

    final_questions = []

    for i, question in enumerate(all_questions, start=1):

        question = re.sub(
            r"(?i)^(Question\s*)?\d+\s*[:.)-]\s*",
            "",
            question
        )

        final_questions.append(
            f"Question {i}: {question}"
        )

    return "\n\n".join(final_questions)


def generate_flashcard_batch(content, number_cards):
    response = safe_api_call(
        model="openai/gpt-oss-20b",
        max_completion_tokens=800,
        messages=[
            {
                "role": "system",
                "content": f"""
Create exactly {number_cards} study flashcards.

Use ONLY the provided study material.

Format every flashcard EXACTLY like this:

Front: question
Back: answer

Requirements:
- Create exactly {number_cards} flashcards.
- Cover different concepts.
- Avoid repetition.
- Do not invent information.
- Answers should be concise but useful.
- Every Front must have a Back.
- Do not leave incomplete flashcards.

Return ONLY the flashcards.
"""
            },
            {
                "role": "user",
                "content": content
            }
        ],
        wait_time=4
    )

    return response.choices[0].message.content


def extract_flashcards(text):
    matches = list(re.finditer(r"(?i)(?:^|\n)\s*Front\s*:", text))

    cards = []

    if not matches:
        return cards

    for i, match in enumerate(matches):

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        card = text[start:end].strip()

        if re.search(r"(?i)\bBack\s*:", card):
            cards.append(card)

    return cards


def generate_flashcards(chunks, embeddings, number_cards):
    selected_chunks = select_chunks(
        chunks,
        embeddings,
        number_chunks=min(5, len(chunks)),
        max_chars=14000
    )

    content = "\n\n".join(selected_chunks)

    batch_size = 3
    all_cards = []

    while len(all_cards) < number_cards:

        remaining = number_cards - len(all_cards)
        cards_this_batch = min(batch_size, remaining)

        flashcards = generate_flashcard_batch(
            content,
            cards_this_batch
        )

        cards = extract_flashcards(flashcards)

        for card in cards:

            if len(all_cards) >= number_cards:
                break

            all_cards.append(card)

        if not cards:
            break

    return "\n\n".join(all_cards[:number_cards])


if mode == "Ask a question":

    value = st.text_input("What would you like to know?")

elif mode == "Summarise":

    value = "Summarise the document"

elif mode == "Extract information":

    value = st.text_input("What information would you like to extract?")

elif mode == "Generate a quiz":

    number_questions = st.number_input(
        "How many questions?",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"]
    )

    quiz_type = st.selectbox(
        "Question type",
        ["Multiple Choice", "True/False", "Short Answer", "Mixed"]
    )

elif mode == "Generate flashcards":

    number_cards = st.number_input(
        "How many flashcards?",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )


messages = st.session_state.messages

for i in range(len(messages) - 2, -1, -2):

    user_message = messages[i]
    assistant_message = messages[i + 1]

    with st.chat_message("user"):
        st.markdown(user_message["content"])

    with st.chat_message("assistant"):
        st.markdown(assistant_message["content"])


if run and file_upload:

    try:

        file_content = extract_from_file(file_upload)

        if not file_content.strip():
            st.error("File does not have any content...")
            st.stop()

        file_content = clean_text(file_content)

        chunks = chunk_text(file_content)

        embeddings = create_embeddings(chunks)

        if mode == "Ask a question" or mode == "Extract information":

            if not value.strip():
                st.warning("Please enter your request first.")
                st.stop()

            with st.spinner("Finding relevant information..."):

                content = embed(
                    chunks,
                    embeddings,
                    value,
                    max_chars=16000
                )

            with st.spinner("Generating answer..."):

                results = generate_answer(
                    content,
                    value,
                    mode
                )

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": value
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": results
                }
            )

            st.rerun()

        elif mode == "Summarise":

            with st.spinner("Generating summary..."):

                results = generate_summary(chunks)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Summarise the document"
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": results
                }
            )

            st.rerun()

        elif mode == "Generate a quiz":

            with st.spinner("Generating your quiz..."):

                results = generate_quiz(
                    chunks,
                    embeddings,
                    number_questions,
                    difficulty,
                    quiz_type
                )

            if not results:

                st.error(
                    "The quiz could not be generated. "
                    "Please try again after the Groq rate limit resets."
                )

                st.stop()

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"Generate a {difficulty} {quiz_type} quiz with {number_questions} questions."
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": results
                }
            )

            st.rerun()

        elif mode == "Generate flashcards":

            with st.spinner("Generating flashcards..."):

                flashcards = generate_flashcards(
                    chunks,
                    embeddings,
                    number_cards
                )

            if not flashcards:

                st.error(
                    "The flashcards could not be generated. "
                    "Please try again after the Groq rate limit resets."
                )

                st.stop()

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"Generate {number_cards} flashcards."
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": flashcards
                }
            )

            st.rerun()

    except Exception as e:

        st.error(f"Something went wrong: {e}")