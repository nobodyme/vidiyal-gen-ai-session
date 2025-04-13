## Day 2

Create API key you require from [Google](https://aistudio.google.com/app/apikey). We can use Gemini models to learn, Google is offering them at free of cost upto a certain limit.

Make an API call to gemini model to test connectivity using in the basic section, it is referenced from [gemini's getting started page](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)


### How to run

Install depedencies
```python3
pip3 install -r requirements.txt
```
Run
```python3
python3 basic/index.py > output.md
```

Similarly, once langchain code is obtained, copy paste relevant content from output.md into langchain/index.py, and run,

```python3
python3 langchain/index.py
```