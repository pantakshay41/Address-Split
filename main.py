from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TFAutoModelForQuestionAnswering
import tensorflow as tf

model = TFAutoModelForQuestionAnswering.from_pretrained("Address-Split-Bert/Address-QA-model")
tokenizer = AutoTokenizer.from_pretrained("Address-Split-Bert/Address-QA-Tokenizer")

context = "9700 Amherst Avenue Margate City New Jersey 08402"


def split_address(address):
    street = "What is the Street?"
    city = "What is the City?"
    state = "What is the State?"
    zip_code = "What is the Zip Code?"
    question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)
    print(address)
    return (question_answerer(question=street, context=address)['answer']
            , question_answerer(question=city, context=address)['answer']
            , question_answerer(question=state, context=address)['answer']
            , question_answerer(question=zip_code, context=address)['answer'])


print(split_address(context))
