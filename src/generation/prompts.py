EDUCATION_QA_PROMPT = """You are an AI tutor for medical students.
Based on the following educational content, answer the student's question.
Cite sources as [1], [2], etc. If the content doesn't fully answer
the question, say so clearly.

Context:
{context}

Student question: {question}

Provide a clear, educational answer with citations."""

STUDY_GUIDE_PROMPT = """Based on the following content,
create a concise study guide covering the key concepts.
Use bullet points, highlight definitions, and note
common exam topics.

Content:
{context}

Topic: {topic}"""

SUMMARIZATION_PROMPT = """Summarize the following medical
education content for a student reviewing for boards.
Focus on clinically relevant points.

Content:
{context}"""
