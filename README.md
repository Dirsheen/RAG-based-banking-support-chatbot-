# RAG-based-banking-support-chatbot-
A RAG-based Banking Support Chatbot is an AI-powered assistant that answers banking queries using Retrieval-Augmented Generation. It retrieves relevant information from documents like loan policies, FAQs, and credit card details using vector search, then generates accurate, context-aware, and grounded responses through an LLM.

# RAG-Based Banking Support Chatbot

An AI-powered Banking Support Chatbot built using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware, and document-grounded responses for banking-related queries such as loans, credit cards, banking policies, and FAQs.

---

## Features

- Conversational AI chatbot
- RAG-based response generation
- PDF & TXT document upload
- Semantic document retrieval
- ChromaDB vector database integration
- Context-aware conversations
- Session-based memory
- FastAPI backend APIs
- Modern responsive UI
- Cloud deployment support

---

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui

### Backend
- FastAPI
- Python

### AI & RAG
- LangChain
- OpenAI / Groq API
- Sentence Transformers
- ChromaDB

---

## Project Structure

```bash
banking-rag-chatbot/
│
├── backend/
│   ├── app/
│   ├── services/
│   ├── routes/
│   ├── utils/
│   └── requirements.txt
│
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── lib/
│   └── package.json
│
└── README.md
