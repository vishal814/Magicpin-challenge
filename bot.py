import os
import time
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
START = time.time()

# Set up OpenAI Client
# Ensure OPENAI_API_KEY is set in environment variables
# For the challenge, the key might be provided by the user via env var
client = AsyncOpenAI() 

# In-memory stores (use Redis/SQLite for production-grade)
contexts: dict[tuple[str, str], dict] = {}    # (scope, context_id) -> {version, payload}
conversations: dict[str, list] = {}           # conversation_id -> [turns]


@app.get("/")
async def root():
    return {"message": "magicpin Vera Bot is running! The judge API is located at /v1/"}

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Alpha",
        "team_members": ["Assistant"],
        "model": "gpt-4o",
        "approach": "single-prompt composer with structured JSON output",
        "contact_email": "team@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    async def process_trigger(trg_id):
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: return None
        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload") if merchant else None
        if not (merchant and category): return None
        
        # Check if customer context exists
        customer_id = trg.get("customer_id")
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None

        # Build context for LLM
        prompt_context = {
            "category": category,
            "merchant": merchant,
            "trigger": trg,
            "customer": customer
        }

        system_prompt = f"""You are 'Vera', an AI marketing assistant for magicpin. Your goal is to text merchants to help them grow their business, OR text their customers on their behalf.
You must return a JSON response matching the following schema:
{{
    "action": "send" | "skip",
    "body": "The actual WhatsApp message text (empty if skip)",
    "cta": "YES/STOP, open_ended, or none (empty if skip)",
    "rationale": "A short explanation of why this message will compel them to reply, or why you skipped"
}}

Rules:
1. SPECIFICITY: Anchor on a verifiable fact (number, date, headline, exact price). Do not use generic "10% off".
2. CATEGORY FIT: Match the tone of the business. (e.g. Dentists = clinical peer, Salons = warm).
3. MERCHANT FIT: Personalize to this specific merchant's data, offers, and language preference. Mix Hindi-English if their language preference allows it.
4. TRIGGER RELEVANCE: Clearly communicate "why now" using the trigger data.
5. SINGLE CTA: Only one clear action, preferably a binary YES/STOP choice.
6. URLs: Allowed only if they add clear value.
7. NO HALLUCINATION: Only use facts from the context.
8. RESTRAINT: If the trigger is irrelevant, or the merchant is hostile/over-messaged, choose "skip".
"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a message based on this context:\n{json.dumps(prompt_context, indent=2)}"}
                ],
                response_format={ "type": "json_object" },
                temperature=0.2
            )
            
            output = json.loads(response.choices[0].message.content)
            
            if output.get("action") == "skip":
                return None
            
            send_as = "merchant_on_behalf" if customer else "vera"
            
            return {
                "conversation_id": f"conv_{merchant_id}_{trg_id}",
                "merchant_id": merchant_id, 
                "customer_id": customer_id,
                "send_as": send_as, 
                "trigger_id": trg_id,
                "template_name": "vera_generic_v1",
                "template_params": [],
                "body": output.get("body", "Hello!"), 
                "cta": output.get("cta", "open_ended"),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": output.get("rationale", "Generated by LLM")
            }
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

    tasks = [process_trigger(tid) for tid in body.available_triggers]
    results = await asyncio.gather(*tasks)
    actions = [res for res in results if res is not None]
    
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_history = conversations.setdefault(body.conversation_id, [])
    conv_history.append({"from": body.from_role, "msg": body.message})
    
    # Retrieve merchant and category
    merchant = contexts.get(("merchant", body.merchant_id), {}).get("payload") if body.merchant_id else None
    category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload") if merchant else None
    customer = contexts.get(("customer", body.customer_id), {}).get("payload") if body.customer_id else None
    
    system_prompt = """You are 'Vera', an AI marketing assistant for magicpin. You are in the middle of a WhatsApp conversation.
You must return a JSON response determining the next action.
Schema:
{
    "action": "send" | "wait" | "end",
    "body": "The reply text if action is 'send'. Empty otherwise.",
    "wait_seconds": 1800, // Only if action is 'wait'
    "cta": "YES/STOP, open_ended, or none",
    "rationale": "Why you chose this action"
}

Rules for deciding action:
1. AUTO-REPLY DETECTION: If the user sends a canned corporate auto-reply for the first time, return "action": "wait" with wait_seconds 14400. If they repeat it multiple times, return "action": "end".
2. INTENT TRANSITION: If they say "yes", "let's do it", "go ahead" - immediately switch from pitching to ACTION mode (e.g. "Done! I have updated it."). Do not qualify further.
3. HOSTILE: If they say "stop spamming", "not interested", apologize briefly in body and return "action": "end" or just "end".
4. WAIT: If they ask for time ("ask me tomorrow"), return "action": "wait".
5. Otherwise, return "action": "send" with a helpful response that matches their tone and language preference.
"""

    context_str = f"Category: {json.dumps(category)}\nMerchant: {json.dumps(merchant)}\nCustomer: {json.dumps(customer)}\n"
    history_str = "\n".join([f"[{t['from']}] {t['msg']}" for t in conv_history])
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context_str}\nConversation History:\n{history_str}\n\nWhat is your next move?"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.2
        )
        
        output = json.loads(response.choices[0].message.content)
        
        action = output.get("action", "send")
        reply_body = output.get("body", "")
        
        if action == "send":
            conv_history.append({"from": "vera", "msg": reply_body})
            return {
                "action": "send", 
                "body": reply_body, 
                "cta": output.get("cta", "open_ended"),
                "rationale": output.get("rationale", "")
            }
        elif action == "wait":
            return {
                "action": "wait",
                "wait_seconds": output.get("wait_seconds", 1800),
                "rationale": output.get("rationale", "")
            }
        else:
            return {
                "action": "end",
                "rationale": output.get("rationale", "")
            }
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"action": "end", "rationale": "Error connecting to LLM, failing gracefully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
