import google.generativeai as genai

# Paste your API key here
genai.configure(api_key="AQ.Ab8RN6IdUl8VOAajZFhawZBZsI1dsOcj6xnkw8NgcE2CY9ppOw")

print("Available Models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)