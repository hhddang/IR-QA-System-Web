import wikipedia as wiki
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from collections import OrderedDict

class DocumentRetrieval:
    # Init 
    def __init__(self):
        pass
    # Extract context
    def retrieveContext(self, question):
        print("Retrieving context...")
        pages = wiki.search(question)
        print(len(pages), " page(s) found: ", pages)
        top_page = pages[0]
        print("Return top page '", top_page,"' as context.")
        context = wiki.page(title=top_page, auto_suggest=False).content
        return top_page, context


class AnswerExtraction:
    # Init
    def __init__(self, model_name,):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_len = 510  #self.model.config.max_position_embeddings
        self.chunked = False
    # Some essential methods
    def __tokenize__(self, question, context):
        self.inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]
        if len(self.input_ids) > self.max_len:
            self.inputs = self.__chunkify__()
            self.chunked = True

    def __chunkify__(self):
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0]
        chunked_input = OrderedDict()
        for k,v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)
            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}
                thing = torch.cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))
                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def __convert_ids_to_string__(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))
    
    # Extract answer
    def extractAnswer(self, question, context):
        print("Extracting answer...")
        self.__tokenize__(question, context)
        if self.chunked:
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk, return_dict=False)
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1
                answer = self.__convert_ids_to_string__(chunk['input_ids'][0][answer_start:answer_end])
                if answer != '[CLS]':
                    return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            return self.__convert_ids_to_string__(self.inputs['input_ids'][0][answer_start:answer_end])


class QASystem:
    # Init QA system
    def __init__(self, documentRetrieval, answerExtraction):
        self.documentRetrieval = documentRetrieval      # Document retrieval process
        self.answerExtraction = answerExtraction        # Question extraction process
    # Answer the input question
    def answer(self, question):
        print(" # QUESTION: ", question)
        top_page, context = self.documentRetrieval.retrieveContext(question)
        answer = self.answerExtraction.extractAnswer(question, context)
        print(" # ANSWER: ", answer)
        return answer, top_page

# TEST

# question = "What is Machine learning ?"

# docRtv = DocumentRetrieval()
# ansExt = AnswerExtraction("deepset/bert-base-cased-squad2")

# qa = QASystem(docRtv, ansExt)
# answer, top_page = qa.answer(question)
# print(answer)
