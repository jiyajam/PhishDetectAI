# PhishDetectAI - A Phishing Detection System

PhishDetectAI is an AI-driven system designed to automatically classify emails and messages as legitimate or phishing. The system uses machine learning, fine-tuning and retrieval-augmented generation(RAG) to detect fraudulent communication patterns and improve cybersecurity.

## Problem Summary

Phishing attacks are increasing globally, targeting individuals and organizations. It is common cybersecurity threat where attackers use fraudulent emails or messages to trick users into revealing sensitive data (such as passwords, banking information or personal data), performing unsafe action and clicking malicious links.

Manual detection is unreliable and existing filters often miss advanced phishing attempts. There is a need for an AI-powered system that can analyze message/email content, detect phishing patterns, retrieve relevant threat intelligence and provide accurate classification.

## Project Goal

The goal of this project is to investigate different AI and machine learning approaches that can classify messages as legitimate or phishing using:

- Fine-tuning a Qwen 1B model
- Implementing RAG workflow
- Comparing modern retrieval methods with TF-IDF

The project will be completed in two sprints:

- Sprint 1: Prepare a dataset and fine‑tune a Qwen 1B model for phishing classification.
- Sprint 2: Implement a Retrieval‑Augmented Generation (RAG) system and compare its performance with the fine‑tuned model and a traditional TF‑IDF‑based machine‑learning approach.

The purpose is not to assume that one method is better than another, but to experimentally determine which approach is most appropriate for the selected phishing detection problem.

## Target User

The system is intended for users who want a quick and easy way to check whether a suspicious email or message might be phishing. It is designed for everyday individuals, students, employees, and small organizations who need a simple tool to verify the legitimacy of messages without requiring technical cybersecurity knowledge.


## How It Works

1. **Preprocessing** — raw emails are cleaned, normalized, and split into train/test sets
2. **Embedding** — email text is embedded into vectors
3. **Fine-tuning** — a Qwen 1B model is fine-tuned (LoRA) on labeled phishing/legitimate emails
4. **Retrieval** — for each new email, the most similar examples are retrieved from a Chroma vector DB
5. **Classification** — the model predicts legitimate vs. phishing, optionally using retrieved context (RAG)
6. **Comparison** — RAG + fine-tuned vs. fine-tuned alone vs. TF-IDF baseline

## Data Attribution

This project uses the **Phishing Email Dataset**.

- **Title:** Phishing Email Dataset
- **Author:** Naser Abdullah Alam and 1 collaborator
- **Source:** https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **License:** [FILL IN FROM KAGGLE PAGE — likely CC BY-SA 4.0]
- **Modifications:** Filtered phishing/spam emails, cleaned and normalized text, split into train/test sets, converted to JSONL format.

**Required citation:**

Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024).
*Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection.*
arXiv. https://arxiv.org/abs/2405.11619

Because the dataset is licensed under CC BY-SA 4.0, the model weights and vector database
in this repository are derivative artifacts. They are made available under the same
CC BY-SA 4.0 terms to the extent required by that license.

