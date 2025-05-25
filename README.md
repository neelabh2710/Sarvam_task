🧠 Task-Specific LLM Evaluation and Bot Development
📌 Overview
This project involves the development and evaluation of task-specific bots using both small and large language models (LLMs). The focus is on analyzing their behavior, especially in tool usage during multi-turn conversations, and proposing strategies to improve small LLM performance.

🤖 Bots Developed
1. Job Search Bot
Uses user metadata and preferences to fetch job listings.

Involves multiple tool calls such as search_jobs.

2. Credit Card Assistant Bot
Manages credit card details.

Handles reminders and payment notifications.

📊 Tool Call Comparison
Smaller Model (LLaMA 3B): 40% incorrect tool calls.

Larger Model (LLaMA 70B): ~5% incorrect tool calls.

Ground truth manually constructed to validate comparison.

📚 Literature Review Highlights
1. LLMs Get Lost in Multi-Turn Conversations
Degradation due to over-eagerness and unreliability.

2. LightPlanner
Small LLMs lack deep reasoning and struggle with execution feedback.

3. Attention Satisfies
Smaller models hallucinate due to poor attention mechanisms.

🔍 Key Findings
1. Reliability Challenges
Small LLMs generate inaccurate tool outputs (e.g., invalid job links or irrelevant arguments).

2. Tool Selection Limitations
Incorrect tool invocation in multi-turn scenarios.

Fails to maintain user context across turns.

3. Constraint Adherence Failures
Issues in following constraints like semantic accuracy, factuality, and consistency.

Mismatch between tool call and textual output.

💡 Proposed Solutions
1. Fine-Tuning with Synthetic Tool Call Data
Inspired by ToolFlow to improve coherence and constraint adherence.

2. Modular Multi-LLM Architecture
Inspired by “Small LLMs Are Weak Tool Learners” paper.

Use specialized components: planner, tool caller, response generator.

3. ReAct Prompting with Self-Refinement
Iterative self-improvement of reasoning and tool calling behavior.

📄 References
LLMs Get Lost in Multi-Turn Conversations

LightPlanner

Attention Satisfies

Small LLMs Are Weak Tool Learners

Less is More: Function Calling for Edge Devices
