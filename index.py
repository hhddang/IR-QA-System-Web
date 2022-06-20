from flask import Flask, render_template, request, url_for, redirect
from QA import DocumentRetrieval, AnswerExtraction, QASystem

# Init QA system
documentRetrieval = DocumentRetrieval()
answerExtraction = AnswerExtraction("deepset/bert-base-cased-squad2")
qaSystem = QASystem(documentRetrieval, answerExtraction)

# Init server
app=Flask("app", template_folder='templates')

@app.route('/', methods=["POST", "GET"])
def main():
    if request.method=="POST":
        question=request.form["question"]
        answer, top_page = qaSystem.answer(question)
        return render_template('home.html', question=question, top_page=top_page, answer=answer)
    else:
        return render_template('home.html', question="", top_page="", answer="")

# Run server
if __name__ == "__main__":
    app.run(debug=True)