from src.generation.gemini_client import generate_json, generate_text

print(generate_text(
    "In one sentence, what is retrieval-augmented generation?",
    system_instruction="You are a concise technical tutor.",
))

result = generate_json(
    'Split this question into independent sub-questions and return JSON like '
    '{"subquestions": ["...", "..."]}: '
    '"What is the learning rate used, and how does it compare to the baseline?"',
    thinking_budget=0,
)
print(result)
print(type(result))