from langchain.prompts import PromptTemplate

prompt_templates = {
    # Existing youtube_sum_3.py academic/insight prompts
    "insights": PromptTemplate(
        input_variables=["text"],
        template="""
You are an academic assistant. Given the following section of a document, extract the following:

1. Definitions of important terms (in bullet points).
2. List of technical terms (as a flat list).
3. Main ideas (summarize in a few bullet points).
4. Important takeaways or findings.
5. Any conclusions or implications.
6. Recurring themes or patterns.
7. High-level insights or summaries.

Section:
{text}
"""
    ),
    "citations": PromptTemplate(
        input_variables=["text"],
        template="""
You are an academic assistant. Analyze the following text section and extract any references, citations, or sources it includes.

Return:

1. In-text citations or reference markers (e.g., [1], (Smith, 2022), etc.)
2. Bibliographic references (if any full or partial references are present).
3. URLs or DOIs.
4. Mentioned author names, publication titles, or institutions.
5. Any inferred sources that are mentioned (e.g., "a study from Harvard" or "a report by WHO").

Section:
{text}
"""
    ),
    "summary": PromptTemplate(
        input_variables=["text"],
        template="Summarize this academic content clearly and concisely:\n\n{text}"
    ),
    "default": PromptTemplate(
        input_variables=["text"],
        template="Extract relevant academic information from the section below:\n\n{text}"
    ), 
    "final_insight": PromptTemplate(
        input_variables=['text'], 
        template="""
You are an expert analyst specializing in extracting and refining insights from large documents and transcripts.

Below is a list of insights extracted from various sections. These may contain overlaps, redundancies, or fragmented points.

---
{text}
Your task is to:
- Eliminate redundant or overlapping insights
- Merge similar ideas into unified, clear insights
- Retain all unique and meaningful information
- Present the final insights in a clean, concise bullet-point format
- Ensure the output is easy to read and focused only on essential takeaways without repeating information

Final refined insights:
"""
    ),

    # Prompts from prompts.py module
    "theme_title": PromptTemplate(
        input_variables=["summaries", "existing_titles"],
        template="""Given the following research paper summaries, generate a concise, human-readable, and unique theme title that:
- Clearly distinguishes this theme from others in this batch.
- Highlights the most specific technical innovation or topic.
- Avoids generic phrases like 'Advancements' or 'Introduction'.
- Do NOT repeat or closely match these other theme titles: {existing_titles}

Summaries:
{summaries}

Theme Title:
"""
    ),
    "final_theme": PromptTemplate(
        input_variables=["theme_summaries"],
        template="""
Given the following theme summaries, synthesize and group them into 3-5 final themes. For each theme, provide:
- Title
- Summary (4-6 sentences, capturing consensus, controversy, innovation, and future directions)

Theme Summaries:
{theme_summaries}
"""
    ),
    "final_conclusion": PromptTemplate(
        input_variables=["all_summaries"],
        template="""
Given all theme, cluster, and synthesis summaries, write a 1-2 paragraph final conclusion that holistically synthesizes the research landscape, consensus, open challenges and future directions.

All Summaries:
{all_summaries}
"""
    ),
    "takeaway": PromptTemplate(
        input_variables=["takeaways"],
        template="""
Given the following list of takeaways, synthesize and distill them into 3-6 final actionable takeaways or key insights for researchers and practitioners. Bullet point format.

Takeaways:
{takeaways}
"""
    ),
    "introduction": PromptTemplate(
        input_variables=["user_query", "all_summaries"],
        template="""
Write a 1 paragraph introduction to the following research synthesis report, providing context for the query, scope, and relevance.

User Query:
{user_query}

All Summaries:
{all_summaries}
"""
    ),
    "tldr": PromptTemplate(
        input_variables=["all_summaries"],
        template="""
Given all theme, cluster, and synthesis summaries, produce a 2-3 sentence TLDR suitable for an executive or busy reader.

All Summaries:
{all_summaries}
"""
    ),
    "key_terms": PromptTemplate(
        input_variables=["summaries"],
        template="""
You are an academic research assistant. Given the following research theme, cluster, and synthesis summaries, extract and return a dictionary of key technical terms and concepts found in the text.

For each key term, provide a concise definition (1-2 sentences) that explains its meaning in the context of the research. Only include terms that are specific, non-generic, and relevant to the technical context.

Format the output as valid JSON: {{"key_terms": [{{"term": "...", "definition": "..."}}]}}

Summaries:
{summaries}

Key Technical Terms (JSON):
"""
    ),
}