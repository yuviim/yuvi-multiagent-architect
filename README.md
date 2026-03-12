**Yuvi's Multi-Agent AI Architect
**
Built by Yuvaraj M — Senior ML Architect  
Pattern: Multi-Agent Supervisor + Human-in-the-Loop (HITL)  
Stack: LangGraph + LangChain + Anthropic Claude + Streamlit

## What This Does

Type a client problem in plain English.  
Four AI agents collaborate to produce a C-suite ready solution brief.  
You review the architecture before the final brief is written.

## Setup

1. Clone the repo
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file: `ANTHROPIC_API_KEY=your_key_here`
5. Run: `streamlit run app.py`

## Agent Pipeline
```
Problem → Researcher → Architect → YOU (review) → Writer → Brief
```

## Token Cost

- Model: Claude Haiku (claude-haiku-4-5-20251001)
- Cost per full run: ~$0.0008
- Total tokens per run: ~5,000-7,000

## Sample Problems

**Banking Fraud:**
A leading bank processes 2M transactions daily. Fraud losses up 40% YoY. Need real-time detection under 200ms with explainable decisions.

**Healthcare:**
Hospital network needs patient history across 12 systems in under 10 seconds. Currently takes 20 minutes. HIPAA compliance required.

**Exasol NL-to-SQL:**
Investment bank needs natural language queries on 10TB trading data returning results in under 30 seconds for intraday margin calls.
