This is an experimental application in Python to post-editing machine translations by prompting an LLM through API calls.
The code outputs a final post-edited translation TXT file and an Excel file with the several corrections made by the LLM to the initial translation based on the following post-editing steps: accuracy, terminology, fluency, style, and inclusive language.
Instructions for each step are included in their respective TXT files in the Files folder.
The Files folder already includes sample source and target files, but you can save your own source (original) and target (translation) files to the Files folder before running the application.
Then run elsa.py
