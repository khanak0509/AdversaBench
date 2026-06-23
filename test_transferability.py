import json
import os
import time
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    with open("dataset_verified.json") as f:
        data = json.load(f)
    

    failed = [r for r in data if r.get("target_failed")]
    test_set = failed[:15]
    

    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        max_tokens=1000,
        temperature=0.0,
    )
    
    results = []
    
    for i, row in enumerate(test_set):

        prompt = row.get("prompt", row.get("adversarial_prompt"))
        
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer accurately and completely."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
            preview = content[:100].replace('\n', ' ')

            
            results.append({
                "seed_id": row["seed_id"],
                "category": row["category"],
                "adversarial_prompt": prompt,
                "expected_behavior": row["expected_behavior"],
                "response_70b": content
            })
        except Exception:
            pass
            
        time.sleep(1)
        
    with open("transferability_results.json", "w") as f:
        json.dump(results, f, indent=2)
        


if __name__ == "__main__":
    main()
