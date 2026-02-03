# 🚀 Quick Start Guide

## Step 1: Install Dependencies ✅

Already completed! All required packages are installed.

## Step 2: Get Your OpenAI API Key 🔑

1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy the key (it starts with `sk-`)

## Step 3: Run the App 🎮

Open your terminal and run:

```bash
# go to the directory where app.py is located
# run the app
streamlit run app.py
```

The app will open in your browser automatically at `http://localhost:8501`

## Step 4: Use the App 📚

1. **Enter API Key**: Paste your OpenAI API key in the sidebar
2. **Upload Document**: Click "Browse files" and select a PDF, DOCX, or TXT file
3. **Ask Questions**: Type your question in the text box and click "Ask"
4. **Get Answers**: The AI will respond based on your document content

## Features to Try 🎯

### Cache Management

- Check cache statistics in the sidebar
- Clear cache to free up space
- Cached responses load instantly!

### Conversation History

- Ask follow-up questions
- The AI remembers context
- Start a new conversation anytime

### Model Selection

- Choose between GPT-3.5 (faster) or GPT-4 (more accurate)
- Adjust based on your needs

## Example Workflow 💡

1. Upload a research paper (PDF)
2. Ask: "What is this paper about?"
3. Follow up: "What are the main findings?"
4. Ask: "Can you explain the methodology?"
5. All responses are cached for instant replay!

## Tips for Best Results 🌟

- **Upload clear, text-based documents** (not scanned images)
- **Ask specific questions** for better answers
- **Use GPT-3.5** for quick queries, **GPT-4** for complex analysis
- **Monitor cache** to track API usage savings

## Troubleshooting 🔧

### App won't start?

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### API errors?

- Check your API key is correct
- Verify you have credits in your OpenAI account
- Ensure internet connection is stable

### Document not processing?

- Check file format (PDF, DOCX, TXT only)
- Try a different document
- Clear cache and retry

## Need Help? 📖

Check the full README.md for detailed documentation!

---

**Ready to chat with your documents? Run the app now!** 🚀
